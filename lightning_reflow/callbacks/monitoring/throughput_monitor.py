"""
Throughput monitoring callback for PyTorch Lightning.

Tracks and logs training/validation throughput (samples/sec) and total sample counts
to the configured logger (e.g., W&B).
"""

import time
import logging
from typing import Any, Dict, Optional

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback

logger = logging.getLogger(__name__)

# Default smoothing factor for exponential moving average
DEFAULT_EMA_SMOOTHING = 0.1


class ThroughputMonitorCallback(Callback):
    """
    Callback to monitor and log training/validation throughput.

    Tracks (configurable):
    - train/throughput: Training samples per second (EMA smoothed)
    - train/total_samples: Cumulative training samples processed (optional)
    - train/epoch_throughput: Epoch-level training throughput (optional)
    - val/throughput: Validation samples per second (per epoch)
    - val/total_samples: Cumulative validation samples processed (optional)
    - val/epoch_samples: Samples in current validation epoch (optional)
    - val/run_count: Number of validation runs completed

    Throughput is calculated using wall-clock time without GPU synchronization
    to avoid performance overhead. This provides accurate end-to-end throughput
    including any GPU bottlenecks.

    Args:
        ema_smoothing: Smoothing factor for exponential moving average (0-1).
            Higher values give more weight to recent measurements.
            Default: 0.1
        log_on_step: Whether to log throughput on each logging step.
            Default: True
        log_on_epoch: Whether to log epoch-level throughput summary.
            Default: True
        log_total_samples: Whether to log cumulative sample counts.
            Default: True
        log_epoch_samples: Whether to log per-epoch sample counts.
            Default: True
    """

    def __init__(
        self,
        ema_smoothing: float = DEFAULT_EMA_SMOOTHING,
        log_on_step: bool = True,
        log_on_epoch: bool = True,
        log_total_samples: bool = True,
        log_epoch_samples: bool = True,
    ):
        super().__init__()
        self.ema_smoothing = ema_smoothing
        self.log_on_step = log_on_step
        self.log_on_epoch = log_on_epoch
        self.log_total_samples = log_total_samples
        self.log_epoch_samples = log_epoch_samples

        # Training state
        self._train_batch_start_time: Optional[float] = None
        self._train_throughput_ema: Optional[float] = None
        self._train_total_samples: int = 0
        self._train_epoch_samples: int = 0
        self._train_epoch_start_time: Optional[float] = None

        # Validation state
        self._val_batch_start_time: Optional[float] = None
        self._val_epoch_samples: int = 0
        self._val_epoch_start_time: Optional[float] = None
        self._val_total_samples: int = 0
        self._val_run_count: int = 0

    def _get_batch_size(self, batch: Any) -> int:
        """Extract batch size from various batch formats."""
        if isinstance(batch, dict):
            # Try common keys for batch data
            for key in ['observation.images', 'input', 'x', 'image', 'data']:
                if key in batch:
                    tensor = batch[key]
                    if hasattr(tensor, 'shape'):
                        return tensor.shape[0]
            # Fallback: get first tensor's batch size
            for value in batch.values():
                if hasattr(value, 'shape') and len(value.shape) > 0:
                    return value.shape[0]
        elif isinstance(batch, (list, tuple)):
            # Assume first element is input tensor
            if len(batch) > 0 and hasattr(batch[0], 'shape'):
                return batch[0].shape[0]
        elif hasattr(batch, 'shape'):
            # Direct tensor
            return batch.shape[0]

        logger.warning("Could not determine batch size, defaulting to 1")
        return 1

    def _update_ema(self, current_value: float, ema_value: Optional[float]) -> float:
        """Update exponential moving average."""
        if ema_value is None:
            return current_value
        return self.ema_smoothing * current_value + (1 - self.ema_smoothing) * ema_value

    # -------------------------------------------------------------------------
    # Training hooks
    # -------------------------------------------------------------------------

    def on_train_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """Reset epoch counters at the start of each training epoch."""
        self._train_epoch_samples = 0
        self._train_epoch_start_time = time.perf_counter()

    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Record batch start time."""
        self._train_batch_start_time = time.perf_counter()

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Calculate throughput and update counters."""
        if self._train_batch_start_time is None:
            return

        elapsed = time.perf_counter() - self._train_batch_start_time
        batch_size = self._get_batch_size(batch)

        # Update counters
        self._train_total_samples += batch_size
        self._train_epoch_samples += batch_size

        # Calculate instantaneous throughput
        if elapsed > 0:
            instant_throughput = batch_size / elapsed
            self._train_throughput_ema = self._update_ema(
                instant_throughput, self._train_throughput_ema
            )

        # Log at trainer's configured frequency
        if self.log_on_step and self._should_log(trainer):
            self._log_train_metrics(pl_module, batch_size)

    def _should_log(self, trainer: "pl.Trainer") -> bool:
        """Check if we should log based on trainer's log_every_n_steps."""
        log_every_n_steps = getattr(trainer, 'log_every_n_steps', 1)
        return trainer.global_step % log_every_n_steps == 0

    def _log_train_metrics(
        self, pl_module: "pl.LightningModule", batch_size: int
    ) -> None:
        """Log training throughput metrics."""
        if self._train_throughput_ema is not None:
            pl_module.log(
                "train/throughput",
                self._train_throughput_ema,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                batch_size=batch_size,
            )

        if self.log_total_samples:
            pl_module.log(
                "train/total_samples",
                float(self._train_total_samples),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                batch_size=batch_size,
            )

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """Log epoch-level training throughput."""
        if not self.log_on_epoch or self._train_epoch_start_time is None:
            return

        if not self.log_epoch_samples:
            return

        elapsed = time.perf_counter() - self._train_epoch_start_time
        if elapsed > 0 and self._train_epoch_samples > 0:
            epoch_throughput = self._train_epoch_samples / elapsed
            pl_module.log(
                "train/epoch_throughput",
                epoch_throughput,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )

    # -------------------------------------------------------------------------
    # Validation hooks
    # -------------------------------------------------------------------------

    def on_validation_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """Reset validation epoch counters."""
        self._val_epoch_samples = 0
        self._val_epoch_start_time = time.perf_counter()

    def on_validation_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Record validation batch start time."""
        self._val_batch_start_time = time.perf_counter()

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Update validation sample counters."""
        batch_size = self._get_batch_size(batch)
        self._val_epoch_samples += batch_size
        self._val_total_samples += batch_size

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """Log validation throughput metrics."""
        if self._val_epoch_start_time is None:
            return

        # Increment validation run count
        self._val_run_count += 1

        elapsed = time.perf_counter() - self._val_epoch_start_time

        if elapsed > 0 and self._val_epoch_samples > 0:
            val_throughput = self._val_epoch_samples / elapsed
            pl_module.log(
                "val/throughput",
                val_throughput,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

        if self.log_total_samples:
            pl_module.log(
                "val/total_samples",
                float(self._val_total_samples),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

        if self.log_epoch_samples:
            pl_module.log(
                "val/epoch_samples",
                float(self._val_epoch_samples),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

        pl_module.log(
            "val/run_count",
            float(self._val_run_count),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

    # -------------------------------------------------------------------------
    # State persistence
    # -------------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        """Save callback state for checkpointing."""
        return {
            "train_total_samples": self._train_total_samples,
            "val_total_samples": self._val_total_samples,
            "val_run_count": self._val_run_count,
            "train_throughput_ema": self._train_throughput_ema,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore callback state from checkpoint."""
        self._train_total_samples = state_dict.get("train_total_samples", 0)
        self._val_total_samples = state_dict.get("val_total_samples", 0)
        self._val_run_count = state_dict.get("val_run_count", 0)
        self._train_throughput_ema = state_dict.get("train_throughput_ema", None)

        logger.info(
            f"Restored ThroughputMonitorCallback state: "
            f"train_samples={self._train_total_samples}, "
            f"val_samples={self._val_total_samples}, "
            f"val_runs={self._val_run_count}"
        )
