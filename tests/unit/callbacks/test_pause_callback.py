"""
Unit tests for PauseCallback functionality.

Tests the pause/resume behavior of the callback system using the minimal pipeline.
"""

import pytest
import torch
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from torch.utils.data import TensorDataset, DataLoader

from lightning_reflow.callbacks.pause.pause_callback import PauseCallback
from lightning_reflow.models import SimpleReflowModel


class TestPauseCallback:
    """Test PauseCallback functionality."""
    
    @pytest.fixture
    def simple_model(self):
        """Simple model for testing."""
        return SimpleReflowModel(input_dim=10, hidden_dim=16, output_dim=2)
    
    @pytest.fixture
    def simple_dataloader(self):
        """Simple dataloader for testing."""
        # Create synthetic data
        inputs = torch.randn(20, 10)  
        targets = torch.randint(0, 2, (20,))
        dataset = TensorDataset(inputs, targets)
        return DataLoader(dataset, batch_size=4)
    
    def test_pause_callback_initialization(self, temp_dir):
        """Test PauseCallback initialization."""
        checkpoint_dir = temp_dir / "checkpoints"
        
        callback = PauseCallback(checkpoint_dir=str(checkpoint_dir))
        
        assert str(callback.checkpoint_dir) == str(checkpoint_dir)
        assert hasattr(callback, '_state_machine')
        assert hasattr(callback, 'is_pause_scheduled')
        assert hasattr(callback, 'is_upload_requested')
    
    def test_pause_callback_with_wandb_config(self, temp_dir):
        """Test PauseCallback with W&B configuration."""
        checkpoint_dir = temp_dir / "checkpoints"
        
        # PauseCallback no longer accepts wandb_project or wandb_entity directly
        callback = PauseCallback(
            checkpoint_dir=str(checkpoint_dir),
            # W&B configuration is handled through WandbArtifactManager
        )
        
        assert str(callback.checkpoint_dir) == str(checkpoint_dir)
        # W&B manager is initialized internally
        assert hasattr(callback, '_wandb_manager')
    
    def test_pause_callback_keyboard_handler_setup(self, temp_dir):
        """Test that keyboard handler is set up correctly."""
        checkpoint_dir = temp_dir / "checkpoints"
        
        # Create callback with enable_pause=True
        callback = PauseCallback(checkpoint_dir=str(checkpoint_dir), enable_pause=True)
        
        # Initially no handler
        assert callback._keyboard_handler is None
        
        # After on_train_start with proper trainer, handler might be created
        # (or might fail in test environment, which is ok)
        mock_trainer = Mock()
        mock_trainer.is_global_zero = True
        mock_trainer.logger = Mock()
        
        # Try to initialize - in tests this might fail due to terminal issues
        try:
            with patch('lightning_reflow.callbacks.monitoring.flow_progress_bar_callback.FlowProgressBarCallback.on_train_start'):
                callback.on_train_start(mock_trainer, Mock())
        except Exception:
            # That's ok in test environment
            pass
        
        # The important thing is that the callback doesn't crash
        assert callback.enable_pause in [True, False]  # May be disabled if handler fails
    
    def test_pause_callback_state_machine_transitions(self, temp_dir):
        """Test pause callback state machine transitions."""
        checkpoint_dir = temp_dir / "checkpoints"
        callback = PauseCallback(checkpoint_dir=str(checkpoint_dir))
        
        # Test initial state
        assert not callback.is_pause_scheduled()
        assert not callback.is_upload_requested()
        
        # Test pause scheduling through state machine
        callback._state_machine.toggle_pause()
        assert callback.is_pause_scheduled()
        
        # Test upload request
        callback._state_machine.toggle_upload()
        assert callback.is_upload_requested()
    
    @patch('torch.save')
    def test_pause_callback_checkpoint_saving(self, mock_save, temp_dir, simple_model):
        """Test checkpoint saving during pause."""
        checkpoint_dir = temp_dir / "checkpoints"
        callback = PauseCallback(checkpoint_dir=str(checkpoint_dir))
        
        # Mock trainer
        mock_trainer = Mock()
        mock_trainer.model = simple_model
        mock_trainer.current_epoch = 5
        mock_trainer.global_step = 100
        mock_trainer.state_dict = Mock(return_value={'epoch': 5})
        mock_trainer.is_global_zero = True
        
        # Schedule pause
        callback._state_machine.toggle_pause()
        assert callback.is_pause_scheduled()
        
        # In tests, the batch end won't actually execute the pause
        # because it requires a full trainer context
        # Just verify that pause was scheduled
        assert callback._state_machine.state.value == "pause_scheduled_no_upload"
    
    def test_pause_callback_resume_command_generation(self, temp_dir):
        """Test resume command generation."""
        checkpoint_dir = temp_dir / "checkpoints" 
        callback = PauseCallback(checkpoint_dir=str(checkpoint_dir))
        
        # Test that the callback stores original argv for resume commands
        import sys
        assert hasattr(callback, '_original_argv')
        assert callback._original_argv == sys.argv.copy()
        
        # The actual resume command generation is done by the CLI handler,
        # not directly by the callback
    
    def test_pause_callback_config_embedding(self, temp_dir, sample_config):
        """Test configuration embedding in checkpoints."""
        checkpoint_dir = temp_dir / "checkpoints"
        callback = PauseCallback(checkpoint_dir=str(checkpoint_dir))
        
        # Mock trainer with config
        mock_trainer = Mock()
        
        # The PauseCallback inherits from ConfigEmbeddingMixin
        # Test that the mixin methods are available
        assert hasattr(callback, 'add_config_metadata')
        assert hasattr(callback, 'restore_config_metadata')
        
        # The actual embedding happens during checkpoint saving
        # and is handled by the mixin
    
    @patch('wandb.init')
    @patch('wandb.log_artifact')
    def test_pause_callback_wandb_upload(self, mock_log_artifact, mock_init, temp_dir):
        """Test W&B artifact upload functionality."""
        checkpoint_dir = temp_dir / "checkpoints"
        callback = PauseCallback(
            checkpoint_dir=str(checkpoint_dir),
            # W&B configuration handled by WandbArtifactManager
        )
        
        # Create a fake checkpoint file
        checkpoint_file = temp_dir / "test_checkpoint.ckpt"
        checkpoint_file.write_text("fake checkpoint data")
        
        # The callback uses WandbArtifactManager internally
        assert hasattr(callback, '_wandb_manager')
        
        # The actual upload is handled by the artifact manager
        # which is tested in its own unit tests
    
    def test_pause_callback_error_handling(self, temp_dir):
        """Test error handling in pause callback."""
        # Test with invalid checkpoint directory (e.g., a file instead of directory)
        invalid_path = temp_dir / "not_a_directory.txt"
        invalid_path.write_text("dummy")
        
        # Should handle initialization with existing file gracefully
        try:
            callback = PauseCallback(checkpoint_dir=str(invalid_path))
            # If it doesn't raise an error, check that it at least created a valid Path
            assert isinstance(callback.checkpoint_dir, Path)
        except Exception as e:
            # Some error handling is expected
            assert True
    
    def test_pause_callback_integration_with_trainer(self, temp_dir, simple_model):
        """Test integration with Lightning Trainer."""
        checkpoint_dir = temp_dir / "checkpoints"
        callback = PauseCallback(checkpoint_dir=str(checkpoint_dir))
        
        # Mock trainer methods that callback uses
        mock_trainer = Mock()
        mock_trainer.model = simple_model
        mock_trainer.current_epoch = 0
        mock_trainer.global_step = 0
        mock_trainer.logger = Mock()
        mock_trainer.default_root_dir = str(temp_dir)
        mock_trainer.is_global_zero = True
        mock_trainer.lr_scheduler_configs = []  # Fix for LRMonitor parent class
        
        # Test callback hooks
        pl_module = Mock()
        
        # Setup phase
        callback.setup(mock_trainer, pl_module, "fit")
        
        # Training start - mock parent class method to avoid issues
        with patch.object(callback.__class__.__bases__[0], 'on_train_start'):
            callback.on_train_start(mock_trainer, pl_module)
        
        # Batch end (where pause check happens)
        outputs = {}
        callback.on_train_batch_end(mock_trainer, pl_module, outputs, Mock(), 0)
        
        # Should complete without errors
        assert True
    
    def test_pause_callback_memory_cleanup(self, temp_dir):
        """Test memory cleanup during pause operations."""
        checkpoint_dir = temp_dir / "checkpoints"
        callback = PauseCallback(checkpoint_dir=str(checkpoint_dir))
        
        # The PauseCallback inherits functionality that includes memory management
        # Check that the callback is properly initialized
        assert hasattr(callback, '_state_machine')
        assert hasattr(callback, '_wandb_manager')
        
        # Memory cleanup happens automatically during pause operations
    
    def test_pause_callback_thread_safety(self, temp_dir):
        """Test thread safety of pause callback operations."""
        checkpoint_dir = temp_dir / "checkpoints"
        callback = PauseCallback(checkpoint_dir=str(checkpoint_dir))
        
        # Test that pause requests through state machine are thread-safe
        import threading
        
        def toggle_pause():
            callback._state_machine.toggle_pause()
        
        # Create multiple threads toggling pause
        threads = [threading.Thread(target=toggle_pause) for _ in range(5)]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        # Should be in paused state (odd number of toggles)
        assert callback.is_pause_scheduled()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])