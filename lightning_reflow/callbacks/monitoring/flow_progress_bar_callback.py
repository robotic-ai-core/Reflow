import lightning.pytorch as pl
import sys
import logging
from typing import Any, Dict, List, Optional
from tqdm import tqdm
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning_reflow.callbacks.monitoring.metric_manager import MetricManager
from lightning_reflow.callbacks.monitoring.validation_interval_calculator import ValidationIntervalCalculator

logger = logging.getLogger(__name__)

# Note: BaseSampleGeneratorCallback is part of the main Yggdrasil project
# For the minimal lightning_reflow framework, we handle this dependency gracefully

# Constants
SMOOTHING_FACTOR = 0.05


class FlowProgressBarCallback(LearningRateMonitor):
    """
    Progress bar callback for Lightning training with dual progress bars.
    
    IMPORTANT NOTE ON PYTORCH LIGHTNING'S INCONSISTENT UNITS:
    - Global progress bar shows OPTIMIZATION STEPS (gradient updates)
    - Interval progress bar shows TRAINING BATCHES (forward passes)
    
    This reflects PyTorch Lightning's internal inconsistency:
    - max_steps counts optimization steps
    - val_check_interval (when integer) counts training batches
    
    With gradient accumulation (e.g., accumulate_grad_batches=16):
    - 16 training batches = 1 optimization step
    - val_check_interval=1600 means validation every 1600 training batches
    - This equals only 100 optimization steps
    
    See the config file documentation for more details.
    """
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

        # Initialize helper classes
        self._metric_manager = MetricManager(
            global_bar_patterns=global_bar_metrics,
            interval_bar_patterns=interval_bar_metrics
        )
        self._interval_calculator = ValidationIntervalCalculator()

        # Expose pattern accessors for backward compatibility
        self.global_bar_metrics = self._metric_manager.global_bar_patterns
        self.interval_bar_metrics = self._metric_manager.interval_bar_patterns

        # Progress bar initialization tracking
        self._progress_bar_initialized = False

    def _populate_metrics_if_needed(self, force_refresh: bool = False) -> None:
        """Populate metrics using the metric manager."""
        self._metric_manager.populate_metrics_if_needed(force_refresh=force_refresh)

    def _format_metrics_postfix(self, metrics: Dict[str, str]) -> str:
        """Format metrics dictionary into postfix string."""
        return self._metric_manager.format_metrics_postfix(metrics)

    @property
    def _global_metrics(self) -> Optional[Dict[str, str]]:
        """Get global metrics from metric manager."""
        return self._metric_manager.get_global_metrics()

    @property
    def _interval_metrics(self) -> Optional[Dict[str, str]]:
        """Get interval metrics from metric manager."""
        return self._metric_manager.get_interval_metrics()

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
        """Update metrics from trainer's callback_metrics."""
        if not self.is_enabled or not self._trainer:
            return
        self._metric_manager.update_metrics(self._trainer, self.logging_interval)

    def _log_lr(self, trainer: "pl.Trainer", interval: str) -> None:
        """Override parent method - learning rate logging handled through metrics system."""
        pass

    def _log_momentum(self, trainer: "pl.Trainer", interval: str) -> None:
        """Override parent method - momentum logging handled through metrics system."""
        pass

    def on_train_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int) -> None:
        """Hook called at the start of each training batch."""
        if not self.is_enabled: return

        self._metric_manager.set_batch_idx(batch_idx)
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

        # Reset caches for new epoch
        self._metric_manager.reset_caches()
        self._update_total_steps_bar(trainer)
        self._update_interval_bar(trainer)

        # Fix interval bar total now that dataloader is available
        self._fix_interval_bar_total_if_needed(trainer)

    def _fix_interval_bar_total_if_needed(self, trainer: "pl.Trainer") -> None:
        """Fix the interval bar total once dataloader is available in on_train_epoch_start."""
        if not self.current_interval_bar:
            return

        # Clear caches to force fresh calculation now that dataloader is available
        self._interval_calculator.reset_caches()

        # Get the correct validation interval steps now that dataloader is available
        val_interval_steps = self._interval_calculator.get_val_check_interval_steps()

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
        """Calculate total training batches processed so far."""
        return self._interval_calculator.get_total_training_batches(current_batch_idx)

    def _calculate_interval_progress(self, trainer: "pl.Trainer", batch_idx: int) -> int:
        """Calculate current progress within the validation interval."""
        return self._interval_calculator.calculate_interval_progress(batch_idx)

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

        self._metric_manager.set_batch_idx(batch_idx)
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
        self._interval_calculator.update_last_validation_batch()
        
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
            val_interval_steps = self._interval_calculator.get_val_check_interval_steps()

            if val_interval_steps:
                interval_total = val_interval_steps
                # Calculate interval BEFORE incrementing validation count
                # After completing validation N, we're starting interval N+1
                next_interval = self._interval_calculator.get_validation_count() + 1
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
            self._interval_calculator.increment_validation_count()

        self._update_metrics()
        self._populate_metrics_if_needed(force_refresh=True)
        self._update_global_bar_postfix()
        self._update_interval_bar_postfix()

    def _get_total_steps(self) -> Optional[int]:
        """Get total training steps."""
        return self._interval_calculator.get_total_steps()

    def _get_val_check_interval_steps(self) -> Optional[int]:
        """Get validation check interval in steps."""
        return self._interval_calculator.get_val_check_interval_steps()

    def _get_steps_until_next_validation(self, trainer: "pl.Trainer", batch_idx: int) -> Optional[int]:
        """Calculate steps remaining until the next validation."""
        return self._interval_calculator.get_steps_until_next_validation(batch_idx)

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
        return self._interval_calculator.get_current_interval(interval_steps)

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

        # Set trainer on helper classes
        self._metric_manager.set_trainer(trainer)
        self._interval_calculator.set_trainer(trainer)

        if not self.is_enabled:
            return

        # Initialize progress bars immediately in on_train_start
        # Now that resume command issue is fixed, global_step should be correct
        self._initialize_progress_bars(trainer)

    def _initialize_progress_bars(self, trainer: "pl.Trainer") -> None:
        """Initialize progress bars with trainer's current global_step."""
        if self._progress_bar_initialized:
            return

        # Initialize validation tracking from checkpoint if resuming
        if self._interval_calculator.get_validation_count() == 0 and trainer.global_step > 0:
            self._interval_calculator.initialize_from_checkpoint()

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
        """Check if using an iterable dataset."""
        return self._interval_calculator._is_iterable_dataset()

    def state_dict(self) -> Dict[str, Any]:
        """Save progress bar callback state for checkpointing (Lightning's standard method)."""
        return {
            'validation_count': self._interval_calculator.get_validation_count(),
            'last_validation_batch': self._interval_calculator.get_last_validation_batch(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore progress bar callback state from checkpoint (Lightning's standard method)."""
        validation_count = state_dict.get('validation_count', 0)
        last_validation_batch = state_dict.get('last_validation_batch', state_dict.get('last_validation_step', 0))

        self._interval_calculator.set_validation_count(validation_count)
        self._interval_calculator.set_last_validation_batch(last_validation_batch)

        # Clear metric caches so they get rebuilt with correct state
        self._metric_manager.reset_caches()

        print(f"✅ Restored FlowProgressBarCallback state via Lightning - validation_count: {validation_count}")
    

    
