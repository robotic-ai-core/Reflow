import lightning.pytorch as pl
import torch
import fnmatch
import sys
import logging
from typing import Any, Dict, List, Optional, Set
from tqdm import tqdm
from lightning.pytorch.callbacks import LearningRateMonitor

logger = logging.getLogger(__name__)

# Note: BaseSampleGeneratorCallback is part of the main Yggdrasil project
# For the minimal lightning_reflow framework, we handle this dependency gracefully

# Constants
SMOOTHING_FACTOR = 0.05
METRIC_FORMAT_FLOAT = "{:.4f}"
METRIC_FORMAT_PRECISE = "{:.6f}"
METRIC_FORMAT_SCIENTIFIC = "{:.2e}"
SCIENTIFIC_THRESHOLD = 1e-6


class FlowProgressBarCallback(LearningRateMonitor):
    def __init__(self, 
                 refresh_rate: int = 1, 
                 process_position: int = 0, 
                 bar_colour: Optional[str] = None,
                 global_bar_metrics: Optional[List[str]] = None,
                 interval_bar_metrics: Optional[List[str]] = None,
                 logging_interval: str = "step"):
        super().__init__(logging_interval=logging_interval, log_momentum=False)
        self._refresh_rate = refresh_rate
        self._process_position = process_position
        self._bar_colour = bar_colour
        self._enabled = True
        self.total_steps_bar: Optional[tqdm] = None
        self.current_interval_bar: Optional[tqdm] = None
        self._trainer: Optional[pl.Trainer] = None

        self._current_batch_idx: Optional[int] = None
        self._prog_bar_metrics = {}
        self.global_bar_metrics = global_bar_metrics if global_bar_metrics is not None else ['*lr*']
        self.interval_bar_metrics = interval_bar_metrics if interval_bar_metrics is not None else ['loss']
        self._global_metrics: Optional[Dict[str, str]] = None
        self._interval_metrics: Optional[Dict[str, str]] = None
        self._global_metric_keys_cache: Optional[Set[str]] = None
        self._interval_metric_keys_cache: Optional[Set[str]] = None
        self._available_metric_keys_cache: Optional[Set[str]] = None
        self._last_validation_batch: int = 0  # Track when validation actually occurred (in training batches)
        self._validation_count: int = 0  # Track how many validations have completed
        
        # Cache for validation interval calculation
        self._cached_val_interval_steps: Optional[int] = None
        self._dataloader_length_cache: Optional[int] = None
        
        # Progress bar initialization tracking
        self._progress_bar_initialized = False
        
        # Register for manager state persistence
        self._register_for_state_persistence()

    def _match_metrics_by_pattern(self, pattern: str, available_metrics: Dict[str, str]) -> Dict[str, str]:
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
        matched_keys = set()
        if '*' in pattern or '?' in pattern:
            for metric_key in available_keys:
                if fnmatch.fnmatch(metric_key, pattern):
                    matched_keys.add(metric_key)
        else:
            if pattern in available_keys:
                matched_keys.add(pattern)
        return matched_keys

    def _get_matched_keys_for_patterns(self, metric_patterns: List[str], available_keys: Set[str], special_metrics: Set[str]) -> Set[str]:
        all_matched_keys = set()
        for pattern in metric_patterns:
            if pattern in special_metrics:
                all_matched_keys.add(pattern)
                continue
            matched_keys = self._get_matched_keys_for_pattern(pattern, available_keys)
            all_matched_keys.update(matched_keys)
        return all_matched_keys

    def _refresh_metric_values_from_cached_keys(self, cached_keys: Set[str], special_metrics: Dict[str, str]) -> Dict[str, str]:
        refreshed_metrics = {}
        for key in cached_keys:
            if key in special_metrics:
                refreshed_metrics[key] = special_metrics[key]
            elif key in self._prog_bar_metrics:
                refreshed_metrics[key] = self._prog_bar_metrics[key]
        return refreshed_metrics

    def _get_metrics_for_bar(self, metric_patterns: List[str], special_metrics: Dict[str, str]) -> Dict[str, str]:
        bar_metrics = {}
        for pattern in metric_patterns:
            if pattern in special_metrics:
                bar_metrics[pattern] = special_metrics[pattern]
                continue
            matched_metrics = self._match_metrics_by_pattern(pattern, self._prog_bar_metrics)
            bar_metrics.update(matched_metrics)
        return bar_metrics

    def _get_special_metrics(self) -> Dict[str, str]:
        """Get special metrics like epoch, step, batch_idx."""
        special_metrics = {}
        if self._trainer:
            special_metrics['epoch'] = str(self._trainer.current_epoch + 1)
            special_metrics['step'] = str(self._trainer.global_step)
        if self._current_batch_idx is not None:
            special_metrics['batch_idx'] = str(self._current_batch_idx)
        return special_metrics

    def _populate_metrics_if_needed(self, force_refresh: bool = False) -> None:
        if not self._trainer:
            return
        special_metrics = self._get_special_metrics()
        current_metric_keys = set(self._prog_bar_metrics.keys())
        current_special_keys = set(special_metrics.keys())
        keys_changed = (self._available_metric_keys_cache != current_metric_keys or force_refresh)
        if (not keys_changed and 
            self._global_metric_keys_cache is not None and 
            self._interval_metric_keys_cache is not None):
            self._global_metrics = self._refresh_metric_values_from_cached_keys(
                self._global_metric_keys_cache, special_metrics
            )
            self._interval_metrics = self._refresh_metric_values_from_cached_keys(
                self._interval_metric_keys_cache, special_metrics
            )
            return
        self._available_metric_keys_cache = current_metric_keys
        self._global_metric_keys_cache = self._get_matched_keys_for_patterns(
            self.global_bar_metrics, current_metric_keys, current_special_keys
        )
        self._interval_metric_keys_cache = self._get_matched_keys_for_patterns(
            self.interval_bar_metrics, current_metric_keys, current_special_keys
        )
        self._global_metrics = self._refresh_metric_values_from_cached_keys(
            self._global_metric_keys_cache, special_metrics
        )
        self._interval_metrics = self._refresh_metric_values_from_cached_keys(
            self._interval_metric_keys_cache, special_metrics
        )


    def _format_metrics_postfix(self, metrics: Dict[str, str]) -> str:
        """Format metrics dictionary into postfix string."""
        return ", ".join([f"{k}={v}" for k, v in metrics.items()])

    def _update_global_bar_postfix(self) -> None:
        """Update global progress bar postfix with metrics."""
        if self.total_steps_bar is None or not self._global_metrics:
            return
        self.total_steps_bar.set_postfix_str(self._format_metrics_postfix(self._global_metrics))
    
    def _update_interval_bar_postfix(self) -> None:
        """Update interval progress bar postfix with metrics."""
        if self.current_interval_bar is None or not self._interval_metrics:
            return
        
        # Get the base postfix with metrics
        base_postfix = self._format_metrics_postfix(self._interval_metrics)
        
        # Add steps until next validation if pause is scheduled
        if hasattr(self, '_state_machine') and hasattr(self, '_get_steps_until_next_validation'):
            # Check if we're in PauseCallback (has these attributes)
            if hasattr(self._state_machine, 'is_pause_scheduled') and self._state_machine.is_pause_scheduled():
                steps_until_val = self._get_steps_until_next_validation(self._trainer, self._current_batch_idx or 0)
                if steps_until_val is not None:
                    base_postfix += f" | Next Val: {steps_until_val} steps"
        
        self.current_interval_bar.set_postfix_str(base_postfix)

    def _get_pause_status_suffix(self) -> str:
        """Get status suffix for progress bar descriptions. Override in subclasses."""
        return ""
    
    def _get_global_pause_status_suffix(self) -> str:
        """Get pause status suffix for global progress bar. Override in subclasses."""
        return ""
    
    def _get_interval_pause_status_suffix(self) -> str:
        """Get pause status suffix for interval progress bar. Override in subclasses."""
        return self._get_pause_status_suffix()  # Default to general pause status

    @property
    def refresh_rate(self) -> int:
        return self._refresh_rate

    @property
    def process_position(self) -> int:
        return self._process_position

    @property
    def is_enabled(self) -> bool:
        """Check if progress bar is enabled."""
        # FlowProgressBarCallback is independent of Lightning's built-in progress bar
        # We only check our own internal _enabled flag
        return self._enabled

    @property
    def is_disabled(self) -> bool:
        return not self.is_enabled

    def enable(self) -> None:
        self._enabled = True
        if self.total_steps_bar is not None: self.total_steps_bar.disable = False
        if self.current_interval_bar is not None: self.current_interval_bar.disable = False

    def disable(self) -> None:
        self._enabled = False
        if self.total_steps_bar is not None: self.total_steps_bar.disable = True
        if self.current_interval_bar is not None: self.current_interval_bar.disable = True

    def _update_metrics(self) -> None:
        if not self.is_enabled or not self._trainer:
            return
        if hasattr(self._trainer, 'callback_metrics'):
            metrics_dict = {}
            for k, v in self._trainer.callback_metrics.items():
                if isinstance(v, torch.Tensor):
                    # Convert tensor to Python number, properly detaching from computation graph
                    v = float(v.detach().cpu()) if v.numel() == 1 else v.detach().cpu()
                
                # Scale loss metrics to account for Lightning's automatic normalization
                # Lightning normalizes losses by accumulate_grad_batches internally
                if k == 'loss' and self._trainer.accumulate_grad_batches > 1:
                    v = v * self._trainer.accumulate_grad_batches
                
                if isinstance(v, float):
                    metrics_dict[k] = METRIC_FORMAT_FLOAT.format(v)
                else:
                    metrics_dict[k] = f"{v}"
            try:
                stats = self._extract_stats(self._trainer, self.logging_interval)
                if stats:
                    for key, value in stats.items():
                        if isinstance(value, torch.Tensor):
                            value = value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu()
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

    def _log_lr(self, trainer: "pl.Trainer", interval: str) -> None:
        """Override parent method - learning rate logging handled through metrics system."""
        pass

    def _log_momentum(self, trainer: "pl.Trainer", interval: str) -> None:
        """Override parent method - momentum logging handled through metrics system."""
        pass

    def on_train_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int) -> None:
        """Hook called at the start of each training batch."""
        if not self.is_enabled: return
        
        self._current_batch_idx = batch_idx
        self._update_metrics()
        self._populate_metrics_if_needed(force_refresh=False)
        
        if batch_idx % self._refresh_rate != 0: 
            return
            
        self._update_total_steps_bar(trainer)
        self._update_interval_bar(trainer)
        
        # Update progress bars
        self._update_progress_bar(self.total_steps_bar, trainer.global_step)
        interval_progress = self._calculate_interval_progress(trainer, batch_idx)
        self._update_progress_bar(self.current_interval_bar, interval_progress)
        
        # Update postfix displays
        self._update_global_bar_postfix()
        self._update_interval_bar_postfix()

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self.is_enabled: 
            return
        
        self._current_batch_idx = None
        self._global_metrics = None
        self._interval_metrics = None
        self._global_metric_keys_cache = None
        self._interval_metric_keys_cache = None
        self._available_metric_keys_cache = None
        self._update_total_steps_bar(trainer)
        self._update_interval_bar(trainer)
        
        # Fix interval bar total now that dataloader is available
        self._fix_interval_bar_total_if_needed(trainer)

    def _fix_interval_bar_total_if_needed(self, trainer: "pl.Trainer") -> None:
        """Fix the interval bar total once dataloader is available in on_train_epoch_start."""
        if not self.current_interval_bar:
            return
            
        # Clear caches to force fresh calculation now that dataloader is available
        self._cached_val_interval_steps = None
        self._dataloader_length_cache = None
            
        # Get the correct validation interval steps now that dataloader is available
        val_interval_steps = self._get_val_check_interval_steps()
        
        if val_interval_steps and self.current_interval_bar.total != val_interval_steps:
            # Update the interval bar total
            self.current_interval_bar.total = val_interval_steps
            self.current_interval_bar.refresh()
            print(f"✅ Updated interval progress bar total to {val_interval_steps} steps (dataloader now available)")

    def _update_progress_bar(self, bar: Optional[tqdm], progress: int) -> None:
        """Update progress bar position and refresh."""
        if bar is not None:
            bar.n = progress
            bar.refresh()

    def _get_total_training_batches(self, trainer: "pl.Trainer", current_batch_idx: Optional[int] = None) -> int:
        """
        Calculate total training batches processed so far.
        
        Args:
            trainer: PyTorch Lightning trainer
            current_batch_idx: Current batch index within epoch (for real-time progress)
            
        Returns:
            Total number of training batches processed
        """
        # Use epoch-based calculation for accurate batch-by-batch progress
        if (current_batch_idx is not None and 
            hasattr(trainer, 'num_training_batches') and 
            trainer.num_training_batches != float('inf')):
            # Accurate calculation: epoch * batches_per_epoch + current_batch
            total_batches = trainer.current_epoch * trainer.num_training_batches + (current_batch_idx + 1)
            return total_batches
        else:
            # Fallback: use global_step approach (updates per gradient accumulation cycle)
            accumulate_grad_batches = getattr(trainer, 'accumulate_grad_batches', 1)
            return trainer.global_step * accumulate_grad_batches
    
    def _calculate_interval_progress(self, trainer: "pl.Trainer", batch_idx: int) -> int:
        """
        Calculate current progress within the validation interval.
        
        Uses modulo arithmetic for simple, accurate progress calculation.
        """
        val_interval_steps = self._get_val_check_interval_steps()
        
        if val_interval_steps:
            # Pass batch_idx for smooth batch-by-batch updates
            total_training_batches = self._get_total_training_batches(trainer, current_batch_idx=batch_idx)
            return total_training_batches % val_interval_steps
        else:
            # For pure epoch-based training without validation intervals
            # progress is the batch within the current epoch
            return batch_idx + 1

    def on_train_batch_end(
        self, 
        trainer: "pl.Trainer", 
        pl_module: "pl.LightningModule", 
        outputs: Optional[Dict[str, Any]], 
        batch: Any, 
        batch_idx: int
    ) -> None:
        if not self.is_enabled: 
            return
        
        self._current_batch_idx = batch_idx
        self._update_metrics()
        self._populate_metrics_if_needed(force_refresh=False)
        
        if batch_idx % self._refresh_rate == 0:
            self._update_total_steps_bar(trainer)
            self._update_interval_bar(trainer)
        
        # Update progress bars
        self._update_progress_bar(self.total_steps_bar, trainer.global_step)
        interval_progress = self._calculate_interval_progress(trainer, batch_idx)
        self._update_progress_bar(self.current_interval_bar, interval_progress)
        
        # Update postfix displays
        self._update_global_bar_postfix()
        self._update_interval_bar_postfix()

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self.is_enabled: 
            return
        
        # Track when validation actually occurs (for state persistence)
        self._last_validation_batch = self._get_total_training_batches(trainer)
        
        # Configure bar for validation - only update totals, not description
        if self.current_interval_bar is not None:
            validation_batch_count = self._get_validation_batch_count(trainer)
            if validation_batch_count:
                self._update_bar_total(self.current_interval_bar, validation_batch_count)
            self.current_interval_bar.set_description("Validating")
            self.current_interval_bar.reset() # Reset progress and timer
            self.current_interval_bar.set_postfix_str("") # Clear training postfix

        self._update_metrics()
        self._populate_metrics_if_needed(force_refresh=True)
        self._update_global_bar_postfix()
        self._update_interval_bar_postfix()
    
    def on_validation_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if not self.is_enabled: 
            return
            
        # Update validation progress
        self._update_progress_bar(self.current_interval_bar, batch_idx)
        
        if batch_idx % self._refresh_rate == 0:
            self._update_metrics()
            self._populate_metrics_if_needed(force_refresh=False)
            self._update_interval_bar_postfix()
    
    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if not self.is_enabled: 
            return
            
        # Update validation progress
        self._update_progress_bar(self.current_interval_bar, batch_idx + 1)
        
        if (batch_idx + 1) % self._refresh_rate == 0:
            self._update_metrics()
            self._populate_metrics_if_needed(force_refresh=False)
            self._update_interval_bar_postfix()
    
    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self.is_enabled: 
            return
        
        self._update_total_steps_bar(trainer)
        
        # Reset the interval bar for the next training interval
        if self.current_interval_bar is not None:
            # Clear the current bar completely
            self.current_interval_bar.clear()
            self.current_interval_bar.close()
            
            # Small delay to ensure terminal is ready
            import time
            time.sleep(0.01)
            
            # Recreate interval bar for next training interval
            val_interval_steps = self._get_val_check_interval_steps()
            
            if val_interval_steps:
                interval_total = val_interval_steps
                # Calculate interval BEFORE incrementing validation count
                # After completing validation N, we're starting interval N+1
                next_interval = self._validation_count + 1
                if trainer.val_check_interval:
                    interval_desc = f"Interval {next_interval} (Steps to Val)"
                else:
                    interval_desc = f"Interval {next_interval} (Steps to Sample)"
            else:
                interval_total = None
                if hasattr(trainer, 'num_training_batches') and trainer.num_training_batches != float('inf'):
                    interval_total = trainer.num_training_batches
                interval_desc = f"Epoch {trainer.current_epoch + 1}"
            
            # Create new interval bar - use tqdm.tqdm to ensure we get the right class
            # IMPORTANT: Always start from 0 after validation to show proper progress
            from tqdm import tqdm as tqdm_cls
            
            self.current_interval_bar = tqdm_cls(
                desc=interval_desc + self._get_interval_pause_status_suffix(),
                initial=0,  # Always start fresh after validation
                total=interval_total,
                position=self.process_position + 1,
                dynamic_ncols=True,
                colour=self._bar_colour,
                file=sys.stdout,
                leave=False,
                disable=self.is_disabled,
                smoothing=SMOOTHING_FACTOR,
                miniters=1,
                mininterval=0.1
            )
            
            # Note: No need to update _last_validation_batch since we use modulo for progress calculation
            
            # Only override bar format if we truly have no total
            if interval_total is None or interval_total == float('inf'):
                # For unknown total, hide the bar but keep the stats
                self.current_interval_bar.bar_format = '{desc}: {n_fmt} [{elapsed}, {rate_fmt}{postfix}]'
            else:
                # Explicitly ensure default format is used
                self.current_interval_bar.bar_format = None
            
            # Force a refresh to ensure the bar is displayed
            self.current_interval_bar.refresh()
            
            # Now increment validation count after creating the bar
            # This ensures the next call to _get_current_interval returns the correct value
            self._validation_count += 1

        self._update_metrics()
        self._populate_metrics_if_needed(force_refresh=True)
        self._update_global_bar_postfix()
        self._update_interval_bar_postfix()

    def _get_dataloader_length(self) -> Optional[int]:
        """Get training dataloader length using multiple fallback approaches."""
        if self._dataloader_length_cache is not None:
            return self._dataloader_length_cache
        
        if not self._trainer:
            return None
        
        # Method 1: Try trainer.datamodule.train_dataloader() (most reliable during resume)
        try:
            if hasattr(self._trainer, 'datamodule') and self._trainer.datamodule:
                train_dl = self._trainer.datamodule.train_dataloader()
                if hasattr(train_dl, '__len__'):
                    length = len(train_dl)
                    self._dataloader_length_cache = length
                    return length
        except (TypeError, AttributeError, ValueError):
            pass
        
        # Method 2: Try trainer.train_dataloader (might be available later)
        try:
            if hasattr(self._trainer, 'train_dataloader') and self._trainer.train_dataloader:
                train_dl = self._trainer.train_dataloader
                if hasattr(train_dl, '__len__'):
                    length = len(train_dl)
                    self._dataloader_length_cache = length
                    return length
        except (TypeError, AttributeError, ValueError):
            pass
        
        # Method 3: Try accessing via CLI reference (for resume scenarios)
        try:
            if hasattr(self._trainer, 'cli') and self._trainer.cli:
                if hasattr(self._trainer.cli, 'datamodule') and self._trainer.cli.datamodule:
                    train_dl = self._trainer.cli.datamodule.train_dataloader()
                    if hasattr(train_dl, '__len__'):
                        length = len(train_dl)
                        self._dataloader_length_cache = length
                        return length
        except (TypeError, AttributeError, ValueError):
            pass
        
        return None

    def _get_total_steps(self) -> Optional[int]:
        if not self._trainer: return None
        if self._is_step_based_training():
            return self._trainer.max_steps
        if self._trainer.max_epochs and self._trainer.max_epochs != -1:
            if hasattr(self._trainer, 'num_training_batches') and self._trainer.num_training_batches != float('inf') and self._trainer.num_training_batches > 0:
                return self._trainer.max_epochs * self._trainer.num_training_batches
            if not self._is_iterable_dataset():
                if hasattr(self._trainer, 'train_dataloader'):
                    try:
                        train_dl = self._trainer.train_dataloader
                        if hasattr(train_dl, '__len__'):
                            batches_per_epoch = len(train_dl)
                            return self._trainer.max_epochs * batches_per_epoch
                    except (TypeError, AttributeError, ValueError):
                        pass
        return None

    def _get_val_check_interval_steps(self) -> Optional[int]:
        # Return cached result if available
        if self._cached_val_interval_steps is not None:
            return self._cached_val_interval_steps
            
        if not self._trainer: 
            return None
        
            
        # Check if we have an explicit step-based val_check_interval
        val_check_interval = getattr(self._trainer, 'val_check_interval', None)
        check_val_every_n_epoch = getattr(self._trainer, 'check_val_every_n_epoch', None)
        
        # If val_check_interval is set to a meaningful value (not None, not 1.0)
        # AND it's less than the number of training batches, use it directly
        # This handles the case where Lightning sets both values
        if (val_check_interval is not None and 
            val_check_interval != 1.0 and 
            isinstance(val_check_interval, (int, float)) and
            val_check_interval > 0):
            # This is a user-specified step-based interval
            # Skip to the step-based logic below to handle it properly
            pass
        # Otherwise, check if Lightning has epoch-based validation configured 
        elif check_val_every_n_epoch:
            check_val_every_n_epoch = self._trainer.check_val_every_n_epoch
            
            # Get steps per epoch
            num_training_batches = None
            if hasattr(self._trainer, 'num_training_batches') and self._trainer.num_training_batches != float('inf'):
                num_training_batches = self._trainer.num_training_batches
            if num_training_batches is None:
                # Use the robust dataloader length getter
                num_training_batches = self._get_dataloader_length()
            
            if num_training_batches and num_training_batches > 0:
                val_interval_steps = num_training_batches * check_val_every_n_epoch
                # Cache the result for future calls
                self._cached_val_interval_steps = val_interval_steps
                return val_interval_steps
        
        # Second, check if Lightning has step-based validation configured
        if self._trainer.val_check_interval:
            val_check_interval = self._trainer.val_check_interval
            num_training_batches = None
            if hasattr(self._trainer, 'num_training_batches') and self._trainer.num_training_batches != float('inf'):
                num_training_batches = self._trainer.num_training_batches
            if num_training_batches is None:
                # Use the robust dataloader length getter
                num_training_batches = self._get_dataloader_length()
            if isinstance(val_check_interval, int) and val_check_interval > 0:
                # Lightning's val_check_interval counts training batches directly
                # Return the value as-is since we want to track training batches, not optimizer steps
                result = val_check_interval
                # Cache the result for future calls
                self._cached_val_interval_steps = result
                return result
            elif isinstance(val_check_interval, float) and val_check_interval > 0:
                # Handle floats: > 1.0 are absolute training batches, <= 1.0 are fractions of epoch
                if val_check_interval > 1.0:
                    # Lightning treats float > 1.0 as absolute training batch count
                    result = int(val_check_interval)
                    # Cache the result for future calls
                    self._cached_val_interval_steps = result
                    return result
                elif num_training_batches:
                    # For fractions, Lightning already handles accumulation correctly
                    result = int(val_check_interval * num_training_batches) // getattr(self._trainer, 'accumulate_grad_batches', 1)
                    # Cache the result for future calls
                    self._cached_val_interval_steps = result
                    return result
        
        # Third, look for sample generator callbacks if no Lightning validation is configured
        for callback in self._trainer.callbacks:
            # Check if callback has the expected sample generator interface
            if (hasattr(callback, 'has_step_based_sampling') and 
                hasattr(callback, 'sampling_interval_steps')):
                if callback.has_step_based_sampling:
                    return callback.sampling_interval_steps
        
        return None
    
    def _get_steps_until_next_validation(self, trainer: "pl.Trainer", batch_idx: int) -> Optional[int]:
        """Calculate actual steps remaining until the next validation."""
        check_val_every_n_epoch = getattr(trainer, 'check_val_every_n_epoch', None)
        
        if check_val_every_n_epoch:
            # For epoch-based validation
            current_epoch = trainer.current_epoch
            next_val_epoch = ((current_epoch // check_val_every_n_epoch) + 1) * check_val_every_n_epoch
            epochs_until_val = next_val_epoch - current_epoch
            
            # Get steps per epoch
            num_training_batches = None
            if hasattr(trainer, 'num_training_batches') and trainer.num_training_batches != float('inf'):
                num_training_batches = trainer.num_training_batches
            
            if num_training_batches:
                # Steps remaining in current epoch
                steps_left_in_epoch = num_training_batches - (batch_idx + 1)
                # Plus steps in remaining full epochs
                steps_in_full_epochs = (epochs_until_val - 1) * num_training_batches
                return steps_left_in_epoch + steps_in_full_epochs
        
        elif trainer.val_check_interval:
            # For step-based validation
            val_interval = trainer.val_check_interval
            if isinstance(val_interval, int):
                steps_since_last_val = trainer.global_step - self._last_validation_batch
                return val_interval - steps_since_last_val
            elif isinstance(val_interval, float) and val_interval > 1.0:
                steps_since_last_val = trainer.global_step - self._last_validation_batch
                return int(val_interval) - steps_since_last_val
        
        return None

    def _update_total_steps_bar(self, trainer: "pl.Trainer") -> None:
        if self.total_steps_bar is None or not self.is_enabled: 
            return
        total_steps_val = self._get_total_steps()
        if total_steps_val and total_steps_val != self.total_steps_bar.total:
            self.total_steps_bar.total = total_steps_val
            self.total_steps_bar.refresh()
            
        # Update description with pause status (global bar - typically no status)
        base_desc = "Global Steps"
        desc_with_status = base_desc + self._get_global_pause_status_suffix()
        if self.total_steps_bar.desc != desc_with_status:
            self.total_steps_bar.set_description(desc_with_status)

    def _get_validation_batch_count(self, trainer: "pl.Trainer") -> Optional[int]:
        """Extract the number of validation batches from trainer."""
        if not hasattr(trainer, 'num_val_batches') or trainer.num_val_batches is None:
            return None
            
        if isinstance(trainer.num_val_batches, list):
            # Multiple validation dataloaders - use the first one
            num_val_batches = trainer.num_val_batches[0] if trainer.num_val_batches else None
        else:
            num_val_batches = trainer.num_val_batches
        
        # Return None if the value is 0 or inf to avoid progress bar issues
        if num_val_batches and num_val_batches > 0 and num_val_batches != float('inf'):
            return num_val_batches
        return None

    def _update_bar_total(self, bar: tqdm, new_total: Optional[int]) -> None:
        """Update progress bar total if changed."""
        if new_total and new_total != float('inf') and new_total != bar.total:
            bar.total = new_total
            bar.refresh()

    def _get_current_interval(self, trainer: "pl.Trainer", interval_steps: int) -> int:
        """Calculate current interval number based on forward passes."""
        if not trainer or trainer.global_step == 0:
            return 1
        # The current interval is validation_count + 1
        # This represents which interval we're currently training in
        return self._validation_count + 1

    def _update_interval_bar(self, trainer: "pl.Trainer") -> None:
        """Update the interval progress bar based on current training state."""
        if self.current_interval_bar is None or not self.is_enabled:
            return
        
        # Skip updates during sanity check
        if trainer.state.stage == "sanity_check":
            return
        
        # Handle validation mode
        if trainer.validating:
            self._update_bar_total(self.current_interval_bar, self._get_validation_batch_count(trainer))
            self.current_interval_bar.set_description("Validating")
            return
        
        # Handle training mode
        val_interval_steps = self._get_val_check_interval_steps()
        
        # Update bar total
        if val_interval_steps:
            self._update_bar_total(self.current_interval_bar, val_interval_steps)
        elif hasattr(trainer, 'num_training_batches') and trainer.num_training_batches != float('inf'):
            self._update_bar_total(self.current_interval_bar, trainer.num_training_batches)
        
        # Update bar description
        if val_interval_steps:
            current_interval = self._get_current_interval(trainer, val_interval_steps)
            # Check if we have Lightning validation (either step-based or epoch-based)
            if trainer.val_check_interval or (hasattr(trainer, 'check_val_every_n_epoch') and trainer.check_val_every_n_epoch):
                base_desc = f"Interval {current_interval} (Steps to Val)"
            else:
                base_desc = f"Interval {current_interval} (Steps to Sample)"
        else:
            base_desc = f"Epoch {trainer.current_epoch + 1}"
            
        # Add pause status suffix (interval bar shows pause status)
        desc_with_status = base_desc + self._get_interval_pause_status_suffix()
        self.current_interval_bar.set_description(desc_with_status)

    def on_load_checkpoint(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]) -> None:
        """Hook called when checkpoint is loaded - Lightning handles state restoration."""
        logger.debug(f"FlowProgressBar: on_load_checkpoint called with checkpoint keys: {list(checkpoint.keys())}")
        if 'global_step' in checkpoint:
            logger.info(f"FlowProgressBar: Lightning loading checkpoint with global_step={checkpoint['global_step']}")

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # Only call parent's on_train_start if trainer has a logger
        # This prevents LearningRateMonitor from raising MisconfigurationException
        if trainer.logger is not None:
            super().on_train_start(trainer, pl_module)
        
        self._trainer = trainer
        self._prog_bar_metrics = {}
        
        if not self.is_enabled: 
            return
            
        # Initialize progress bars immediately in on_train_start
        # Now that resume command issue is fixed, global_step should be correct
        self._initialize_progress_bars(trainer)

    def _initialize_progress_bars(self, trainer: "pl.Trainer") -> None:
        """Initialize progress bars with trainer's current global_step."""
        if self._progress_bar_initialized:
            return
            
        # Initialize validation tracking (only if not already restored from checkpoint)
        # Note: _last_validation_batch is mainly kept for state persistence; 
        # actual progress calculation uses modulo arithmetic
        if not hasattr(self, '_validation_count') or self._validation_count is None:
            val_interval = self._get_val_check_interval_steps()
            if trainer.global_step > 0 and val_interval:
                # Simple initialization for state tracking (not used in progress calculation)
                total_training_batches = self._get_total_training_batches(trainer)
                self._last_validation_batch = (total_training_batches // val_interval) * val_interval
                self._validation_count = total_training_batches // val_interval
            else:
                # Initialize to 0 (new training or no validation interval)
                self._last_validation_batch = 0
                self._validation_count = 0

        total_steps_val = self._get_total_steps()
        logger.info(f"FlowProgressBar: Initializing with trainer.global_step={trainer.global_step}")
        
        self.total_steps_bar = tqdm(
            desc="Global Steps",
            initial=trainer.global_step, 
            total=total_steps_val,
            position=self.process_position,
            dynamic_ncols=True,
            colour=self._bar_colour,
            file=sys.stdout,
            leave=False,
            disable=self.is_disabled,
            smoothing=SMOOTHING_FACTOR,
            miniters=1,
            mininterval=0.1,
            bar_format=None
        )
        
        if total_steps_val is None or total_steps_val == 0:
            self.total_steps_bar.bar_format = '{desc}: {n_fmt} [{elapsed}, {rate_fmt}{postfix}]'

        # Initialize interval bar
        interval_total = None
        interval_desc = "Current Interval"
        val_interval_steps = self._get_val_check_interval_steps()
        
        if val_interval_steps:
            interval_total = val_interval_steps
            if self._trainer.val_check_interval or (hasattr(self._trainer, 'check_val_every_n_epoch') and self._trainer.check_val_every_n_epoch):
                check_val_every_n_epoch = getattr(self._trainer, 'check_val_every_n_epoch', None)
                if check_val_every_n_epoch:
                    current_epoch = self._trainer.current_epoch
                    epoch_in_cycle = (current_epoch % check_val_every_n_epoch) + 1
                    interval_desc = f"Validation Cycle {self._trainer.current_epoch // check_val_every_n_epoch + 1} - Epoch {epoch_in_cycle}/{check_val_every_n_epoch}"
                else:
                    interval_desc = f"Interval {self._trainer.current_epoch + 1} (Steps to Val)"
            else:
                interval_desc = f"Interval {self._trainer.current_epoch + 1} (Steps to Sample)"
        elif not self._is_iterable_dataset() and hasattr(trainer, 'num_training_batches') and trainer.num_training_batches != float('inf'):
            interval_total = trainer.num_training_batches
            interval_desc = f"Epoch {self._trainer.current_epoch + 1}"
        
        # Calculate initial position for interval bar during resume
        initial_interval_progress = 0
        if val_interval_steps and trainer.global_step > 0:
            # Use consistent calculation method
            total_training_batches = self._get_total_training_batches(trainer)
            initial_interval_progress = total_training_batches % val_interval_steps
        
        self.current_interval_bar = tqdm(
            desc=interval_desc,
            initial=initial_interval_progress,  # Start from current position in interval
            total=interval_total,
            position=self.process_position + 1, 
            dynamic_ncols=True,
            colour=self._bar_colour,
            file=sys.stdout,
            leave=False,
            disable=self.is_disabled,
            smoothing=SMOOTHING_FACTOR,
            miniters=1,
            mininterval=0.1,
            bar_format=None
        )
        
        if interval_total is None or interval_total == float('inf'):
            self.current_interval_bar.bar_format = '{desc}: {n_fmt} [{elapsed}, {rate_fmt}{postfix}]'
        else:
            self.current_interval_bar.bar_format = None
            
        self._progress_bar_initialized = True

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Cleanup progress bars when training ends."""
        if self.total_steps_bar is not None:
            self.total_steps_bar.close()
            self.total_steps_bar = None
        if self.current_interval_bar is not None:
            self.current_interval_bar.close()
            self.current_interval_bar = None
    
    def teardown(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        """Override for custom cleanup in subclasses."""
        pass

    def _is_iterable_dataset(self) -> bool:
        if not self._trainer: return False
        if hasattr(self._trainer, 'num_training_batches') and self._trainer.num_training_batches == float('inf'):
            return True
        if hasattr(self._trainer, '_data_connector'):
            if hasattr(self._trainer._data_connector, '_train_dataloader_source'):
                try:
                    return hasattr(self._trainer._data_connector._train_dataloader_source, '_dataset_is_iterable')
                except Exception:
                    pass
        try:
            if hasattr(self._trainer, 'train_dataloader'):
                train_dl = self._trainer.train_dataloader
                if train_dl is None:
                    return False
                if hasattr(train_dl, 'dataset'):
                    dataset = train_dl.dataset
                    if hasattr(dataset, '__iter__') and not hasattr(dataset, '__getitem__'):
                        return True
                    if hasattr(dataset, 'datasets') and isinstance(dataset.datasets, list) and dataset.datasets:
                        if hasattr(dataset.datasets[0], '__iter__') and not hasattr(dataset.datasets[0], '__getitem__'):
                            return True
        except Exception:
            pass
        return False
    
    def _is_step_based_training(self) -> bool:
        if not self._trainer: return False
        return bool(self._trainer.max_steps and self._trainer.max_steps != -1 and self._trainer.max_steps != float('inf'))
    
    def state_dict(self) -> Dict[str, Any]:
        """Save progress bar callback state for checkpointing (Lightning's standard method)."""
        return {
            'validation_count': self._validation_count,
            'last_validation_batch': self._last_validation_batch,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore progress bar callback state from checkpoint (Lightning's standard method)."""
        self._validation_count = state_dict.get('validation_count', 0)
        self._last_validation_batch = state_dict.get('last_validation_batch', state_dict.get('last_validation_step', 0))
        
        # Clear metric caches so they get rebuilt with correct state
        self._global_metric_keys_cache = None
        self._interval_metric_keys_cache = None
        self._available_metric_keys_cache = None
        
        print(f"✅ Restored FlowProgressBarCallback state via Lightning - validation_count: {self._validation_count}")
    

    
    def _register_for_state_persistence(self) -> None:
        """Register this callback for manager state persistence."""
        try:
            from lightning_reflow.utils.checkpoint.manager_state import register_manager, ManagerState
            import time
            from typing import Dict, Any
            
            class FlowProgressBarState(ManagerState):
                """Manager state for FlowProgressBarCallback."""
                
                def __init__(self, callback: 'FlowProgressBarCallback'):
                    self.callback = callback
                
                @property
                def manager_name(self) -> str:
                    return "flow_progress_bar"
                
                def capture_state(self) -> Dict[str, Any]:
                    """Capture FlowProgressBarCallback state for persistence."""
                    return {
                        'version': '1.0.0',
                        'validation_count': self.callback._validation_count,
                        'last_validation_batch': self.callback._last_validation_batch,
                        'configuration': {
                            'refresh_rate': self.callback._refresh_rate,
                            'global_bar_metrics': self.callback.global_bar_metrics,
                            'interval_bar_metrics': self.callback.interval_bar_metrics,
                            'bar_colour': self.callback._bar_colour
                        },
                        'timestamp': time.time()
                    }
                
                def restore_state(self, state: Dict[str, Any]) -> bool:
                    """Restore FlowProgressBarCallback state from persistence."""
                    try:
                        if not self.validate_state(state):
                            return False
                        
                        # Restore critical tracking state
                        self.callback._validation_count = state.get('validation_count', 0)
                        self.callback._last_validation_batch = state.get('last_validation_batch', state.get('last_validation_step', 0))
                        
                        # Clear metric caches so they get rebuilt with correct state
                        self.callback._global_metric_keys_cache = None
                        self.callback._interval_metric_keys_cache = None
                        self.callback._available_metric_keys_cache = None
                        
                        return True
                        
                    except Exception as e:
                        print(f"Failed to restore FlowProgressBarCallback state: {e}")
                        return False
                
                def validate_state(self, state: Dict[str, Any]) -> bool:
                    """Validate that the state is compatible."""
                    if not isinstance(state, dict):
                        return False
                    
                    version = state.get('version')
                    if version != '1.0.0':
                        print(f"FlowProgressBarCallback: Incompatible state version {version}, expected 1.0.0")
                        return False
                    
                    # Validate required fields (with backward compatibility)
                    if 'validation_count' not in state:
                        print(f"FlowProgressBarCallback: Missing required field in state: validation_count")
                        return False
                        
                    # Check for either new or old field name (backward compatibility)
                    if 'last_validation_batch' not in state and 'last_validation_step' not in state:
                        print(f"FlowProgressBarCallback: Missing required field in state: last_validation_batch (or legacy last_validation_step)")
                        return False
                    
                    return True
            
            # Register the state manager
            state_manager = FlowProgressBarState(self)
            register_manager(state_manager)
            # Note: Don't print here as this gets called during __init__
            
        except ImportError:
            # Manager state system not available - continue with Lightning's built-in state persistence
            pass
        except Exception as e:
            print(f"FlowProgressBarCallback: Failed to register for state persistence: {e}")
