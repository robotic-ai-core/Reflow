"""
Validation interval calculation for progress bar callbacks.

This module handles all validation interval-related calculations:
- Dataloader length detection
- Validation interval step calculation
- Progress calculation within intervals
- Steps until next validation
"""

import logging
from typing import Optional

import lightning.pytorch as pl

logger = logging.getLogger(__name__)


class ValidationIntervalCalculator:
    """
    Calculates validation intervals and progress for progress bars.

    Handles:
    - Multiple methods for dataloader length detection
    - Both step-based and epoch-based validation intervals
    - Progress calculation within validation intervals
    - Caching for performance

    Example:
        calc = ValidationIntervalCalculator()
        calc.set_trainer(trainer)
        interval_steps = calc.get_val_check_interval_steps()
        progress = calc.calculate_interval_progress(batch_idx)
    """

    def __init__(self):
        self._trainer: Optional[pl.Trainer] = None

        # Caches
        self._cached_val_interval_steps: Optional[int] = None
        self._dataloader_length_cache: Optional[int] = None

        # State tracking
        self._last_validation_batch: int = 0
        self._validation_count: int = 0

    def set_trainer(self, trainer: pl.Trainer) -> None:
        """Set trainer reference."""
        self._trainer = trainer

    def reset_caches(self) -> None:
        """Reset all caches (call when dataloader becomes available)."""
        self._cached_val_interval_steps = None
        self._dataloader_length_cache = None

    def get_validation_count(self) -> int:
        """Get the number of completed validations."""
        return self._validation_count

    def set_validation_count(self, count: int) -> None:
        """Set the validation count (used for state restoration)."""
        self._validation_count = count

    def increment_validation_count(self) -> None:
        """Increment the validation count after completing a validation."""
        self._validation_count += 1

    def get_last_validation_batch(self) -> int:
        """Get the batch index of the last validation."""
        return self._last_validation_batch

    def set_last_validation_batch(self, batch: int) -> None:
        """Set the last validation batch (used for state restoration)."""
        self._last_validation_batch = batch

    def update_last_validation_batch(self) -> None:
        """Update last validation batch to current position."""
        if self._trainer:
            self._last_validation_batch = self.get_total_training_batches()

    def get_dataloader_length(self) -> Optional[int]:
        """
        Get training dataloader length using multiple fallback approaches.

        Tries multiple methods to get dataloader length:
        1. trainer.datamodule.train_dataloader()
        2. trainer.train_dataloader
        3. trainer.cli.datamodule.train_dataloader()

        Returns:
            Dataloader length if available, None otherwise
        """
        if self._dataloader_length_cache is not None:
            return self._dataloader_length_cache

        if not self._trainer:
            return None

        # Method 1: Try trainer.datamodule.train_dataloader()
        try:
            if hasattr(self._trainer, 'datamodule') and self._trainer.datamodule:
                train_dl = self._trainer.datamodule.train_dataloader()
                if hasattr(train_dl, '__len__'):
                    length = len(train_dl)
                    self._dataloader_length_cache = length
                    return length
        except (TypeError, AttributeError, ValueError):
            pass

        # Method 2: Try trainer.train_dataloader
        try:
            if hasattr(self._trainer, 'train_dataloader') and self._trainer.train_dataloader:
                train_dl = self._trainer.train_dataloader
                if hasattr(train_dl, '__len__'):
                    length = len(train_dl)
                    self._dataloader_length_cache = length
                    return length
        except (TypeError, AttributeError, ValueError):
            pass

        # Method 3: Try accessing via CLI reference
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

    def get_total_steps(self) -> Optional[int]:
        """
        Calculate total training steps.

        Returns:
            Total steps if calculable, None otherwise
        """
        if not self._trainer:
            return None

        if self._is_step_based_training():
            return self._trainer.max_steps

        if self._trainer.max_epochs and self._trainer.max_epochs != -1:
            if (hasattr(self._trainer, 'num_training_batches') and
                self._trainer.num_training_batches != float('inf') and
                self._trainer.num_training_batches > 0):
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

    def get_val_check_interval_steps(self) -> Optional[int]:
        """
        Get the validation check interval in training batches.

        Handles both step-based and epoch-based validation intervals.
        Also checks for sample generator callbacks.

        Returns:
            Validation interval in training batches, or None
        """
        if self._cached_val_interval_steps is not None:
            return self._cached_val_interval_steps

        if not self._trainer:
            return None

        val_check_interval = getattr(self._trainer, 'val_check_interval', None)
        check_val_every_n_epoch = getattr(self._trainer, 'check_val_every_n_epoch', None)

        # Check if val_check_interval is set to a meaningful value
        if (val_check_interval is not None and
            val_check_interval != 1.0 and
            isinstance(val_check_interval, (int, float)) and
            val_check_interval > 0):
            pass  # Skip to step-based logic below
        elif check_val_every_n_epoch:
            # Epoch-based validation
            num_training_batches = self._get_num_training_batches()
            if num_training_batches and num_training_batches > 0:
                val_interval_steps = num_training_batches * check_val_every_n_epoch
                self._cached_val_interval_steps = val_interval_steps
                return val_interval_steps

        # Step-based validation
        if self._trainer.val_check_interval:
            val_check_interval = self._trainer.val_check_interval
            num_training_batches = self._get_num_training_batches()

            if isinstance(val_check_interval, int) and val_check_interval > 0:
                self._cached_val_interval_steps = val_check_interval
                return val_check_interval
            elif isinstance(val_check_interval, float) and val_check_interval > 0:
                if val_check_interval > 1.0:
                    result = int(val_check_interval)
                    self._cached_val_interval_steps = result
                    return result
                elif num_training_batches:
                    accumulate = getattr(self._trainer, 'accumulate_grad_batches', 1)
                    result = int(val_check_interval * num_training_batches) // accumulate
                    self._cached_val_interval_steps = result
                    return result

        # Check for sample generator callbacks
        for callback in self._trainer.callbacks:
            if (hasattr(callback, 'has_step_based_sampling') and
                hasattr(callback, 'sampling_interval_steps')):
                if callback.has_step_based_sampling:
                    return callback.sampling_interval_steps

        return None

    def get_total_training_batches(self, current_batch_idx: Optional[int] = None) -> int:
        """
        Calculate total training batches processed so far.

        Args:
            current_batch_idx: Current batch index within epoch

        Returns:
            Total number of training batches processed
        """
        if not self._trainer:
            return 0

        if (current_batch_idx is not None and
            hasattr(self._trainer, 'num_training_batches') and
            self._trainer.num_training_batches != float('inf')):
            return self._trainer.current_epoch * self._trainer.num_training_batches + (current_batch_idx + 1)
        else:
            accumulate = getattr(self._trainer, 'accumulate_grad_batches', 1)
            return self._trainer.global_step * accumulate

    def calculate_interval_progress(self, batch_idx: int) -> int:
        """
        Calculate current progress within the validation interval.

        Args:
            batch_idx: Current batch index within epoch

        Returns:
            Progress within current interval
        """
        val_interval_steps = self.get_val_check_interval_steps()

        if val_interval_steps:
            total_training_batches = self.get_total_training_batches(current_batch_idx=batch_idx)
            return total_training_batches % val_interval_steps
        else:
            return batch_idx + 1

    def get_steps_until_next_validation(self, batch_idx: int) -> Optional[int]:
        """
        Calculate steps remaining until the next validation.

        Args:
            batch_idx: Current batch index within epoch

        Returns:
            Steps until next validation, or None if not calculable
        """
        if not self._trainer:
            return None

        check_val_every_n_epoch = getattr(self._trainer, 'check_val_every_n_epoch', None)

        if check_val_every_n_epoch:
            # Epoch-based validation
            current_epoch = self._trainer.current_epoch
            next_val_epoch = ((current_epoch // check_val_every_n_epoch) + 1) * check_val_every_n_epoch
            epochs_until_val = next_val_epoch - current_epoch

            num_training_batches = self._get_num_training_batches()
            if num_training_batches:
                steps_left_in_epoch = num_training_batches - (batch_idx + 1)
                steps_in_full_epochs = (epochs_until_val - 1) * num_training_batches
                return steps_left_in_epoch + steps_in_full_epochs

        elif self._trainer.val_check_interval:
            # Step-based validation
            val_interval = self._trainer.val_check_interval
            if isinstance(val_interval, int):
                steps_since_last_val = self._trainer.global_step - self._last_validation_batch
                return val_interval - steps_since_last_val
            elif isinstance(val_interval, float) and val_interval > 1.0:
                steps_since_last_val = self._trainer.global_step - self._last_validation_batch
                return int(val_interval) - steps_since_last_val

        return None

    def get_current_interval(self, interval_steps: int) -> int:
        """
        Calculate current interval number.

        Args:
            interval_steps: Steps per interval

        Returns:
            Current interval number (1-indexed)
        """
        return self._validation_count + 1

    def initialize_from_checkpoint(self) -> None:
        """Initialize validation tracking when resuming from checkpoint."""
        if not self._trainer:
            return

        val_interval = self.get_val_check_interval_steps()
        if self._trainer.global_step > 0 and val_interval:
            total_training_batches = self.get_total_training_batches()
            self._last_validation_batch = (total_training_batches // val_interval) * val_interval
            self._validation_count = total_training_batches // val_interval
        else:
            self._last_validation_batch = 0
            self._validation_count = 0

    def _get_num_training_batches(self) -> Optional[int]:
        """Get number of training batches per epoch."""
        if not self._trainer:
            return None

        if (hasattr(self._trainer, 'num_training_batches') and
            self._trainer.num_training_batches != float('inf')):
            return self._trainer.num_training_batches

        return self.get_dataloader_length()

    def _is_step_based_training(self) -> bool:
        """Check if training is step-based (max_steps set)."""
        if not self._trainer:
            return False
        return (self._trainer.max_steps is not None and
                self._trainer.max_steps > 0 and
                self._trainer.max_steps != -1)

    def _is_iterable_dataset(self) -> bool:
        """Check if using an iterable dataset."""
        if not self._trainer:
            return False
        try:
            if hasattr(self._trainer, 'datamodule') and self._trainer.datamodule:
                from torch.utils.data import IterableDataset
                train_dataset = getattr(self._trainer.datamodule, 'train_dataset', None)
                if train_dataset is not None:
                    return isinstance(train_dataset, IterableDataset)
        except (TypeError, AttributeError):
            pass
        return False
