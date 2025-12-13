"""
Unit tests for ValidationIntervalCalculator.

Tests validation interval calculation and progress tracking.
"""

import pytest
from unittest.mock import Mock, MagicMock, PropertyMock

from lightning_reflow.callbacks.monitoring.validation_interval_calculator import ValidationIntervalCalculator


class TestValidationIntervalCalculator:
    """Test ValidationIntervalCalculator functionality."""

    @pytest.fixture
    def calculator(self):
        """Create a calculator for testing."""
        return ValidationIntervalCalculator()

    @pytest.fixture
    def mock_trainer(self):
        """Create a mock trainer."""
        trainer = Mock()
        trainer.current_epoch = 5
        trainer.global_step = 100
        trainer.max_steps = 1000
        trainer.max_epochs = 10
        trainer.num_training_batches = 100
        trainer.val_check_interval = None
        trainer.check_val_every_n_epoch = None
        trainer.accumulate_grad_batches = 1
        trainer.callbacks = []
        return trainer

    def test_initialization(self, calculator):
        """Test calculator initialization."""
        assert calculator._trainer is None
        assert calculator._cached_val_interval_steps is None
        assert calculator._dataloader_length_cache is None
        assert calculator._validation_count == 0
        assert calculator._last_validation_batch == 0

    def test_set_trainer(self, calculator, mock_trainer):
        """Test setting trainer reference."""
        calculator.set_trainer(mock_trainer)
        assert calculator._trainer is mock_trainer

    def test_reset_caches(self, calculator):
        """Test cache reset."""
        calculator._cached_val_interval_steps = 100
        calculator._dataloader_length_cache = 50

        calculator.reset_caches()

        assert calculator._cached_val_interval_steps is None
        assert calculator._dataloader_length_cache is None

    def test_validation_count_tracking(self, calculator):
        """Test validation count tracking."""
        assert calculator.get_validation_count() == 0

        calculator.increment_validation_count()
        assert calculator.get_validation_count() == 1

        calculator.set_validation_count(5)
        assert calculator.get_validation_count() == 5


class TestTotalStepsCalculation:
    """Test total steps calculation."""

    @pytest.fixture
    def calculator(self):
        """Create a calculator for testing."""
        return ValidationIntervalCalculator()

    def test_get_total_steps_step_based(self, calculator):
        """Test total steps for step-based training."""
        trainer = Mock()
        trainer.max_steps = 1000
        trainer.max_epochs = None

        calculator.set_trainer(trainer)
        result = calculator.get_total_steps()

        assert result == 1000

    def test_get_total_steps_epoch_based(self, calculator):
        """Test total steps for epoch-based training."""
        trainer = Mock()
        trainer.max_steps = -1
        trainer.max_epochs = 10
        trainer.num_training_batches = 100

        calculator.set_trainer(trainer)
        result = calculator.get_total_steps()

        assert result == 1000  # 10 epochs * 100 batches

    def test_get_total_steps_no_trainer(self, calculator):
        """Test total steps without trainer."""
        result = calculator.get_total_steps()
        assert result is None


class TestValCheckIntervalSteps:
    """Test validation interval calculation."""

    @pytest.fixture
    def calculator(self):
        """Create a calculator for testing."""
        return ValidationIntervalCalculator()

    def test_step_based_interval(self, calculator):
        """Test step-based validation interval."""
        trainer = Mock()
        trainer.val_check_interval = 500
        trainer.check_val_every_n_epoch = None
        trainer.num_training_batches = 100
        trainer.callbacks = []

        calculator.set_trainer(trainer)
        result = calculator.get_val_check_interval_steps()

        assert result == 500

    def test_epoch_based_interval(self, calculator):
        """Test epoch-based validation interval."""
        trainer = Mock()
        trainer.val_check_interval = None
        trainer.check_val_every_n_epoch = 2
        trainer.num_training_batches = 100
        trainer.callbacks = []

        calculator.set_trainer(trainer)
        result = calculator.get_val_check_interval_steps()

        assert result == 200  # 2 epochs * 100 batches

    def test_no_validation_interval(self, calculator):
        """Test when no validation interval is set."""
        trainer = Mock()
        trainer.val_check_interval = None
        trainer.check_val_every_n_epoch = None
        trainer.callbacks = []

        calculator.set_trainer(trainer)
        result = calculator.get_val_check_interval_steps()

        assert result is None

    def test_interval_caching(self, calculator):
        """Test that interval is cached."""
        trainer = Mock()
        trainer.val_check_interval = 500
        trainer.check_val_every_n_epoch = None
        trainer.num_training_batches = 100
        trainer.callbacks = []

        calculator.set_trainer(trainer)

        # First call
        result1 = calculator.get_val_check_interval_steps()

        # Modify trainer (should not affect cached result)
        trainer.val_check_interval = 1000

        # Second call should return cached value
        result2 = calculator.get_val_check_interval_steps()

        assert result1 == result2 == 500


class TestIntervalProgress:
    """Test interval progress calculation."""

    @pytest.fixture
    def calculator(self):
        """Create a calculator for testing."""
        return ValidationIntervalCalculator()

    def test_calculate_interval_progress_with_interval(self, calculator):
        """Test progress calculation with validation interval."""
        trainer = Mock()
        trainer.val_check_interval = 100
        trainer.check_val_every_n_epoch = None
        trainer.current_epoch = 0
        trainer.global_step = 50
        trainer.num_training_batches = 200
        trainer.accumulate_grad_batches = 1
        trainer.callbacks = []

        calculator.set_trainer(trainer)
        result = calculator.calculate_interval_progress(49)  # batch_idx 49 = 50 batches

        assert result == 50  # 50 % 100 = 50

    def test_calculate_interval_progress_no_interval(self, calculator):
        """Test progress calculation without validation interval."""
        trainer = Mock()
        trainer.val_check_interval = None
        trainer.check_val_every_n_epoch = None
        trainer.current_epoch = 0
        trainer.global_step = 50
        trainer.num_training_batches = float('inf')
        trainer.accumulate_grad_batches = 1
        trainer.callbacks = []

        calculator.set_trainer(trainer)
        result = calculator.calculate_interval_progress(49)

        assert result == 50  # batch_idx + 1


class TestTotalTrainingBatches:
    """Test total training batches calculation."""

    @pytest.fixture
    def calculator(self):
        """Create a calculator for testing."""
        return ValidationIntervalCalculator()

    def test_with_batch_idx(self, calculator):
        """Test calculation with current batch index."""
        trainer = Mock()
        trainer.current_epoch = 2
        trainer.global_step = 250
        trainer.num_training_batches = 100
        trainer.accumulate_grad_batches = 1

        calculator.set_trainer(trainer)
        result = calculator.get_total_training_batches(current_batch_idx=49)

        # 2 * 100 + 50 = 250
        assert result == 250

    def test_without_batch_idx(self, calculator):
        """Test calculation without batch index (uses global_step)."""
        trainer = Mock()
        trainer.current_epoch = 2
        trainer.global_step = 200
        trainer.num_training_batches = float('inf')
        trainer.accumulate_grad_batches = 2

        calculator.set_trainer(trainer)
        result = calculator.get_total_training_batches()

        # global_step * accumulate_grad_batches = 200 * 2 = 400
        assert result == 400


class TestStepsUntilValidation:
    """Test steps until next validation calculation."""

    @pytest.fixture
    def calculator(self):
        """Create a calculator for testing."""
        return ValidationIntervalCalculator()

    def test_epoch_based_steps_until_val(self, calculator):
        """Test steps until validation for epoch-based."""
        trainer = Mock()
        trainer.current_epoch = 1
        trainer.val_check_interval = None
        trainer.check_val_every_n_epoch = 2
        trainer.num_training_batches = 100

        calculator.set_trainer(trainer)
        result = calculator.get_steps_until_next_validation(batch_idx=49)

        # Next val at epoch 2, currently at epoch 1 batch 50
        # Steps left in epoch 1: 100 - 50 = 50
        # Steps in remaining epochs: 0 (next val at end of epoch 1)
        assert result == 50

    def test_step_based_steps_until_val(self, calculator):
        """Test steps until validation for step-based."""
        trainer = Mock()
        trainer.current_epoch = 0
        trainer.global_step = 80
        trainer.val_check_interval = 100
        trainer.check_val_every_n_epoch = None

        calculator.set_trainer(trainer)
        calculator._last_validation_batch = 0

        result = calculator.get_steps_until_next_validation(batch_idx=79)

        # 100 - 80 = 20
        assert result == 20

    def test_no_validation_configured(self, calculator):
        """Test when no validation is configured."""
        trainer = Mock()
        trainer.val_check_interval = None
        trainer.check_val_every_n_epoch = None

        calculator.set_trainer(trainer)
        result = calculator.get_steps_until_next_validation(batch_idx=50)

        assert result is None


class TestCurrentInterval:
    """Test current interval calculation."""

    @pytest.fixture
    def calculator(self):
        """Create a calculator for testing."""
        return ValidationIntervalCalculator()

    def test_get_current_interval(self, calculator):
        """Test getting current interval number."""
        calculator._validation_count = 5

        result = calculator.get_current_interval(interval_steps=100)

        assert result == 6  # validation_count + 1


class TestCheckpointRestore:
    """Test checkpoint restoration."""

    @pytest.fixture
    def calculator(self):
        """Create a calculator for testing."""
        return ValidationIntervalCalculator()

    def test_initialize_from_checkpoint(self, calculator):
        """Test initialization from checkpoint."""
        trainer = Mock()
        trainer.current_epoch = 5
        trainer.global_step = 500
        trainer.val_check_interval = 100
        trainer.check_val_every_n_epoch = None
        trainer.num_training_batches = 100
        trainer.accumulate_grad_batches = 1
        trainer.callbacks = []

        calculator.set_trainer(trainer)
        calculator.initialize_from_checkpoint()

        # 500 global_step * 1 accumulate = 500 batches
        # 500 // 100 = 5 validations
        assert calculator._validation_count == 5
        assert calculator._last_validation_batch == 500


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
