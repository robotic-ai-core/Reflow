"""
Metric management for progress bar callbacks.

This module handles all metric-related operations:
- Pattern matching for metric filtering
- Metric caching and refresh
- Metric value formatting
"""

import fnmatch
import torch
import logging
from typing import Dict, List, Optional, Set, Any

import lightning.pytorch as pl

logger = logging.getLogger(__name__)

# Constants
METRIC_FORMAT_FLOAT = "{:.4f}"
METRIC_FORMAT_PRECISE = "{:.6f}"
METRIC_FORMAT_SCIENTIFIC = "{:.2e}"
SCIENTIFIC_THRESHOLD = 1e-6


class MetricManager:
    """
    Manages metric matching, caching, and formatting for progress bars.

    Handles:
    - Pattern-based metric filtering (glob patterns like '*lr*')
    - Caching of matched metric keys for efficiency
    - Metric value formatting with appropriate precision
    - Special metrics (epoch, step, batch_idx)

    Args:
        global_bar_patterns: Metric patterns for global progress bar
        interval_bar_patterns: Metric patterns for interval progress bar

    Example:
        manager = MetricManager(['*lr*'], ['loss'])
        manager.update_metrics(trainer)
        global_metrics = manager.get_global_metrics()
    """

    def __init__(
        self,
        global_bar_patterns: Optional[List[str]] = None,
        interval_bar_patterns: Optional[List[str]] = None
    ):
        self.global_bar_patterns = global_bar_patterns or ['*lr*']
        self.interval_bar_patterns = interval_bar_patterns or ['loss']

        # Raw metrics from trainer
        self._prog_bar_metrics: Dict[str, str] = {}

        # Cached matched metrics
        self._global_metrics: Optional[Dict[str, str]] = None
        self._interval_metrics: Optional[Dict[str, str]] = None

        # Cached matched keys
        self._global_metric_keys_cache: Optional[Set[str]] = None
        self._interval_metric_keys_cache: Optional[Set[str]] = None
        self._available_metric_keys_cache: Optional[Set[str]] = None

        # Reference to trainer for special metrics
        self._trainer: Optional[pl.Trainer] = None
        self._current_batch_idx: Optional[int] = None

        # Async pinned-memory transfer state. Each metric name gets its own
        # pinned host buffer so we can ``non_blocking=True`` copy a scalar
        # off-GPU without forcing the training stream to sync.  Keyed by
        # metric name so the buffer survives across steps and can be reused
        # without re-allocation (pin_memory allocation is expensive).
        #
        # See feat/async-metric-display: reading ``tensor.detach().cpu()`` on
        # the in-flight loss tensor blocks until the CUDA stream drains
        # (~28 ms per call on compute-heavy runs).  Pinned-memory copy +
        # ``float()`` of the host buffer lets training kernels keep running.
        self._pinned_buffers: Dict[str, torch.Tensor] = {}
        # Cache of the most recently observed scalar value per metric.  Used
        # as a fallback when the async copy hasn't completed yet (rare but
        # possible if the bar callback fires twice in quick succession).
        self._last_known_values: Dict[str, float] = {}

    def set_trainer(self, trainer: pl.Trainer) -> None:
        """Set trainer reference for special metrics."""
        self._trainer = trainer

    def set_batch_idx(self, batch_idx: int) -> None:
        """Set current batch index for special metrics."""
        self._current_batch_idx = batch_idx

    def reset_caches(self) -> None:
        """Reset all metric caches (call at epoch start)."""
        self._global_metrics = None
        self._interval_metrics = None
        self._global_metric_keys_cache = None
        self._interval_metric_keys_cache = None
        self._available_metric_keys_cache = None

    def get_global_metrics(self) -> Optional[Dict[str, str]]:
        """Get metrics for global progress bar."""
        return self._global_metrics

    def get_interval_metrics(self) -> Optional[Dict[str, str]]:
        """Get metrics for interval progress bar."""
        return self._interval_metrics

    def format_metrics_postfix(self, metrics: Dict[str, str]) -> str:
        """Format metrics dictionary into postfix string."""
        return ", ".join([f"{k}={v}" for k, v in metrics.items()])

    def update_metrics(self, trainer: pl.Trainer, logging_interval: str = "step") -> None:
        """
        Update metrics from trainer's callback_metrics.

        Args:
            trainer: PyTorch Lightning trainer
            logging_interval: Logging interval for LR stats extraction
        """
        if not hasattr(trainer, 'callback_metrics'):
            return

        metrics_dict = {}
        for k, v in trainer.callback_metrics.items():
            if isinstance(v, torch.Tensor):
                # Async pinned-memory path for scalar tensors avoids the
                # blocking CUDA sync that ``.cpu()`` would otherwise force.
                if v.numel() == 1:
                    v = self._sync_scalar_to_host(k, v)
                else:
                    # Multi-element tensors are uncommon in progress-bar
                    # display.  Preserve legacy behaviour (blocking copy)
                    # rather than maintaining shape-sensitive pinned buffers.
                    v = v.detach().cpu()

            # Scale loss metrics to account for Lightning's normalization
            if k == 'loss' and trainer.accumulate_grad_batches > 1:
                v = v * trainer.accumulate_grad_batches

            if isinstance(v, float):
                metrics_dict[k] = METRIC_FORMAT_FLOAT.format(v)
            else:
                metrics_dict[k] = f"{v}"

        # Try to extract LR stats
        try:
            from lightning.pytorch.callbacks import LearningRateMonitor
            # Create a temporary LR monitor to extract stats
            # This is a workaround since we can't call parent method directly
            if hasattr(self, '_lr_monitor') and self._lr_monitor:
                stats = self._lr_monitor._extract_stats(trainer, logging_interval)
                if stats:
                    for key, value in stats.items():
                        if isinstance(value, torch.Tensor):
                            # LR stats are scalars; route through the async
                            # path under the ``lr:`` prefix so it doesn't
                            # clash with the loss buffer.
                            if value.numel() == 1:
                                value = self._sync_scalar_to_host(
                                    f"_lrstat::{key}", value
                                )
                            else:
                                value = value.detach().cpu()
                        if isinstance(value, float):
                            if abs(value) < SCIENTIFIC_THRESHOLD:
                                metrics_dict[key] = METRIC_FORMAT_SCIENTIFIC.format(value)
                            else:
                                metrics_dict[key] = METRIC_FORMAT_PRECISE.format(value)
                        else:
                            metrics_dict[key] = f"{value}"
        except (AttributeError, TypeError):
            pass

        self._prog_bar_metrics.update(metrics_dict)

    def _sync_scalar_to_host(self, name: str, tensor: torch.Tensor) -> float:
        """Copy a scalar tensor to a pinned host buffer without stalling.

        For CUDA tensors the copy is issued with ``non_blocking=True`` into a
        per-metric pinned buffer.  The training stream therefore does not
        wait on the H2D-side traffic; only the subsequent ``float()`` read
        of the pinned buffer can stall, and only by however much of the copy
        is still in flight at that moment (typically a small fraction of the
        full sync cost since the kernel work overlaps with the copy).

        For CPU tensors the function falls through to a plain ``float()`` —
        no pin_memory is needed.

        Args:
            name: Stable per-metric key (used to size/cache the pinned buffer).
            tensor: Scalar tensor (``numel() == 1``).  Multi-element tensors
                must be handled by the caller.

        Returns:
            The current scalar value as a Python ``float``.
        """
        # Defensive: only the scalar fast path goes through here.
        if tensor.numel() != 1:
            raise ValueError(
                f"_sync_scalar_to_host expects a scalar tensor; got numel="
                f"{tensor.numel()} for metric {name!r}"
            )

        detached = tensor.detach()

        # CPU-only path: no pinning needed, no async benefit available.
        if not detached.is_cuda:
            value = float(detached)
            self._last_known_values[name] = value
            return value

        # Lazy alloc / re-alloc on dtype change.  Pin once, reuse forever.
        buf = self._pinned_buffers.get(name)
        if buf is None or buf.dtype != detached.dtype or buf.numel() != 1:
            try:
                buf = torch.empty(
                    (1,), dtype=detached.dtype, device='cpu', pin_memory=True
                )
            except RuntimeError:
                # pin_memory can fail (e.g. CUDA disabled at runtime, OOM in
                # pinned region).  Fall back to a blocking copy rather than
                # crashing the bar.
                logger.debug(
                    "Pinned-memory alloc failed for metric %r; falling back "
                    "to blocking copy.", name
                )
                value = float(detached.cpu())
                self._last_known_values[name] = value
                return value
            self._pinned_buffers[name] = buf

        # The win lives here: non_blocking=True for a pinned destination
        # means the copy is issued on the current CUDA stream and the host
        # call returns immediately.  Training kernels keep running.
        try:
            buf.view(()).copy_(detached, non_blocking=True)
            value = float(buf.view(()))
        except RuntimeError:
            # Stream/device hiccup — fall back to blocking copy so the
            # progress bar never crashes training.  This branch is also
            # exercised on the rare path where pin_memory was nominally
            # allocated but the device backend disagrees at copy time.
            logger.debug(
                "Async pinned copy failed for metric %r; falling back to "
                "blocking copy.", name,
            )
            value = float(detached.cpu())

        self._last_known_values[name] = value
        return value

    def populate_metrics_if_needed(self, force_refresh: bool = False) -> None:
        """
        Populate global and interval metrics from cached keys.

        Uses caching for efficiency - only recalculates when keys change.

        Args:
            force_refresh: Force recalculation of matched keys
        """
        if not self._trainer:
            return

        special_metrics = self._get_special_metrics()
        current_metric_keys = set(self._prog_bar_metrics.keys())
        current_special_keys = set(special_metrics.keys())

        keys_changed = (self._available_metric_keys_cache != current_metric_keys or force_refresh)

        if (not keys_changed and
            self._global_metric_keys_cache is not None and
            self._interval_metric_keys_cache is not None):
            # Just refresh values using cached keys
            self._global_metrics = self._refresh_metric_values_from_cached_keys(
                self._global_metric_keys_cache, special_metrics
            )
            self._interval_metrics = self._refresh_metric_values_from_cached_keys(
                self._interval_metric_keys_cache, special_metrics
            )
            return

        # Recalculate matched keys
        self._available_metric_keys_cache = current_metric_keys
        self._global_metric_keys_cache = self._get_matched_keys_for_patterns(
            self.global_bar_patterns, current_metric_keys, current_special_keys
        )
        self._interval_metric_keys_cache = self._get_matched_keys_for_patterns(
            self.interval_bar_patterns, current_metric_keys, current_special_keys
        )

        # Refresh values
        self._global_metrics = self._refresh_metric_values_from_cached_keys(
            self._global_metric_keys_cache, special_metrics
        )
        self._interval_metrics = self._refresh_metric_values_from_cached_keys(
            self._interval_metric_keys_cache, special_metrics
        )

    def _get_special_metrics(self) -> Dict[str, str]:
        """Get special metrics like epoch, step, batch_idx."""
        special_metrics = {}
        if self._trainer:
            special_metrics['epoch'] = str(self._trainer.current_epoch + 1)
            special_metrics['step'] = str(self._trainer.global_step)
        if self._current_batch_idx is not None:
            special_metrics['batch_idx'] = str(self._current_batch_idx)
        return special_metrics

    def _match_metrics_by_pattern(self, pattern: str, available_metrics: Dict[str, str]) -> Dict[str, str]:
        """Match metrics by glob pattern."""
        matched_metrics = {}
        if '*' in pattern or '?' in pattern:
            for metric_key, metric_value in available_metrics.items():
                if fnmatch.fnmatch(metric_key, pattern):
                    matched_metrics[metric_key] = metric_value
        else:
            if pattern in available_metrics:
                matched_metrics[pattern] = available_metrics[pattern]
        return matched_metrics

    def _get_matched_keys_for_pattern(self, pattern: str, available_keys: Set[str]) -> Set[str]:
        """Get matched keys for a single pattern."""
        matched_keys = set()
        if '*' in pattern or '?' in pattern:
            for metric_key in available_keys:
                if fnmatch.fnmatch(metric_key, pattern):
                    matched_keys.add(metric_key)
        else:
            if pattern in available_keys:
                matched_keys.add(pattern)
        return matched_keys

    def _get_matched_keys_for_patterns(
        self,
        metric_patterns: List[str],
        available_keys: Set[str],
        special_keys: Set[str]
    ) -> Set[str]:
        """Get matched keys for multiple patterns."""
        all_matched_keys = set()
        for pattern in metric_patterns:
            if pattern in special_keys:
                all_matched_keys.add(pattern)
                continue
            matched_keys = self._get_matched_keys_for_pattern(pattern, available_keys)
            all_matched_keys.update(matched_keys)
        return all_matched_keys

    def _refresh_metric_values_from_cached_keys(
        self,
        cached_keys: Set[str],
        special_metrics: Dict[str, str]
    ) -> Dict[str, str]:
        """Refresh metric values using cached keys."""
        refreshed_metrics = {}
        for key in cached_keys:
            if key in special_metrics:
                refreshed_metrics[key] = special_metrics[key]
            elif key in self._prog_bar_metrics:
                refreshed_metrics[key] = self._prog_bar_metrics[key]
        return refreshed_metrics

    def get_metrics_for_bar(
        self,
        metric_patterns: List[str],
        special_metrics: Dict[str, str]
    ) -> Dict[str, str]:
        """Get metrics matching patterns for a progress bar."""
        bar_metrics = {}
        for pattern in metric_patterns:
            if pattern in special_metrics:
                bar_metrics[pattern] = special_metrics[pattern]
                continue
            matched_metrics = self._match_metrics_by_pattern(pattern, self._prog_bar_metrics)
            bar_metrics.update(matched_metrics)
        return bar_metrics
