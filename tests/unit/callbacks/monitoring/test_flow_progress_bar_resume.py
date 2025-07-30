#!/usr/bin/env python3
"""
Unit tests for FlowProgressBarCallback resume functionality.

This test ensures that the progress bar correctly preserves global_step
when resuming from checkpoint.
"""

import pytest
import torch
import lightning.pytorch as pl
from unittest.mock import Mock, MagicMock

from lightning_reflow.callbacks.monitoring import FlowProgressBarCallback


class TestFlowProgressBarResume:
    """Test that FlowProgressBarCallback handles resume correctly."""
    
    def test_progress_bar_immediate_initialization(self):
        """Test that progress bar initialization happens immediately in on_train_start."""
        callback = FlowProgressBarCallback()
        
        # Mock trainer with global_step = 0 initially
        trainer = Mock()
        trainer.global_step = 0
        trainer.logger = None
        trainer.current_epoch = 0
        trainer.val_check_interval = None
        trainer.check_val_every_n_epoch = None
        trainer.max_steps = 1000
        trainer.callback_metrics = {}
        trainer.num_val_batches = None
        trainer.num_training_batches = float('inf')
        trainer.accumulate_grad_batches = 1
        trainer.state = Mock()
        trainer.state.stage = None
        trainer.validating = False
        
        # Mock methods that will be called during initialization
        callback._get_total_steps = Mock(return_value=1000)
        callback._get_val_check_interval_steps = Mock(return_value=None)
        callback._is_iterable_dataset = Mock(return_value=True)
        
        # Call on_train_start - should initialize progress bars immediately
        callback.on_train_start(trainer, Mock())
        
        # Check that progress bars are initialized
        assert hasattr(callback, '_progress_bar_initialized') and callback._progress_bar_initialized
        assert callback.total_steps_bar is not None
        assert callback.current_interval_bar is not None
    
    def test_progress_bar_initialization_with_resumed_global_step(self):
        """Test that progress bars are initialized with correct global_step when resuming."""
        callback = FlowProgressBarCallback()
        
        # Mock trainer simulating resumed state
        trainer = Mock()
        trainer.global_step = 1234  # Simulating resumed state
        trainer.logger = None
        trainer.current_epoch = 5
        trainer.val_check_interval = None
        trainer.check_val_every_n_epoch = None
        trainer.max_steps = 2000
        trainer.callback_metrics = {}  # Mock empty metrics
        trainer.num_val_batches = None  # No validation batches
        trainer.num_training_batches = float('inf')  # Unknown number of batches
        trainer.accumulate_grad_batches = 1
        trainer.state = Mock()
        trainer.state.stage = None
        trainer.validating = False
        
        # Mock that we can get total steps
        callback._get_total_steps = Mock(return_value=2000)
        callback._get_val_check_interval_steps = Mock(return_value=None)
        callback._is_iterable_dataset = Mock(return_value=True)
        
        # Call on_train_start - should initialize bars immediately
        callback.on_train_start(trainer, Mock())
        
        # Verify bars are initialized with correct global_step
        assert callback.total_steps_bar is not None
        assert callback.total_steps_bar.n == 1234  # tqdm sets n to initial value
        assert callback.total_steps_bar.initial == 1234  # Initial should be set to global_step
        assert callback._progress_bar_initialized is True
        
        # Now call on_train_batch_start - bars should already be initialized
        callback.on_train_batch_start(trainer, Mock(), Mock(), 0)
        
        # Verify progress bars remain initialized with same values
        assert callback.total_steps_bar is not None
        assert callback.total_steps_bar.initial == 1234  # Initial value preserved
    
    def test_progress_bar_not_reinitialized_on_subsequent_batches(self):
        """Test that progress bars are not re-initialized on subsequent batches."""
        callback = FlowProgressBarCallback()
        
        # Setup
        trainer = Mock()
        trainer.global_step = 100
        trainer.logger = None
        trainer.current_epoch = 1
        trainer.val_check_interval = None
        trainer.check_val_every_n_epoch = None
        trainer.max_steps = 1000
        trainer.callback_metrics = {}  # Mock empty metrics
        trainer.num_val_batches = None
        trainer.num_training_batches = float('inf')
        
        callback._trainer = trainer
        callback._get_total_steps = Mock(return_value=1000)
        callback._get_val_check_interval_steps = Mock(return_value=None)
        callback._is_iterable_dataset = Mock(return_value=True)
        
        # Initialize
        callback.on_train_start(trainer, Mock())
        callback.on_train_batch_start(trainer, Mock(), Mock(), 0)
        
        # Capture the progress bar reference
        original_bar = callback.total_steps_bar
        
        # Call on_train_batch_start again
        trainer.global_step = 101
        callback.on_train_batch_start(trainer, Mock(), Mock(), 1)
        
        # Verify same progress bar instance (not re-initialized)
        assert callback.total_steps_bar is original_bar
        
        # Verify that the progress bar was updated on the second batch
        assert callback.total_steps_bar.n == 101  # Should be updated to new global_step
    
    def test_progress_bar_handles_disabled_state(self):
        """Test that progress bar handles disabled state correctly."""
        callback = FlowProgressBarCallback()
        callback._enabled = False  # Disable the callback
        
        trainer = Mock()
        trainer.global_step = 500
        trainer.logger = None
        
        # Set up the callback's internal trainer reference
        callback._trainer = trainer
        
        # Should not crash when disabled
        callback.on_train_start(trainer, Mock())
        
        # Manually check if progress bars would be initialized when disabled
        # The is_enabled property should return False when _is_disabled is True
        assert not callback.is_enabled
        
        # on_train_batch_start should return early when disabled
        callback.on_train_batch_start(trainer, Mock(), Mock(), 0)
        
        # Progress bars should not be created when disabled
        assert callback.total_steps_bar is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 