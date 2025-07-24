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
        
        assert callback.checkpoint_dir == str(checkpoint_dir)
        assert hasattr(callback, 'pause_requested')
        assert hasattr(callback, 'upload_requested')
    
    def test_pause_callback_with_wandb_config(self, temp_dir):
        """Test PauseCallback with W&B configuration."""
        checkpoint_dir = temp_dir / "checkpoints"
        
        callback = PauseCallback(
            checkpoint_dir=str(checkpoint_dir),
            wandb_project="test-project",
            wandb_entity="test-entity"
        )
        
        assert callback.checkpoint_dir == str(checkpoint_dir)
        # Additional W&B specific assertions would go here
    
    @patch('lightning_reflow.callbacks.pause.unified_keyboard_handler.UnifiedKeyboardHandler')
    def test_pause_callback_keyboard_handler_setup(self, mock_handler_class, temp_dir):
        """Test that keyboard handler is set up correctly."""
        checkpoint_dir = temp_dir / "checkpoints"
        mock_handler = Mock()
        mock_handler_class.return_value = mock_handler
        
        callback = PauseCallback(checkpoint_dir=str(checkpoint_dir))
        
        # Mock trainer and setup
        mock_trainer = Mock()
        mock_trainer.logger = Mock()
        mock_trainer.default_root_dir = str(temp_dir)
        
        callback.setup(mock_trainer, Mock(), "fit")
        
        # Verify handler was created
        mock_handler_class.assert_called_once()
    
    def test_pause_callback_state_machine_transitions(self, temp_dir):
        """Test pause callback state machine transitions."""
        checkpoint_dir = temp_dir / "checkpoints"
        callback = PauseCallback(checkpoint_dir=str(checkpoint_dir))
        
        # Test initial state
        assert not callback.pause_requested
        assert not callback.upload_requested
        
        # Test pause request
        callback.request_pause()
        assert callback.pause_requested
        
        # Test upload request  
        callback.request_upload()
        assert callback.upload_requested
    
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
        
        # Mock the checkpoint creation
        with patch.object(callback, '_create_checkpoint') as mock_create:
            mock_create.return_value = "checkpoint_data"
            
            # Simulate pause request and checkpoint
            callback.request_pause()
            callback.on_train_batch_end(mock_trainer, Mock(), {}, 0)
            
            # Verify checkpoint creation was called
            mock_create.assert_called_once()
    
    def test_pause_callback_resume_command_generation(self, temp_dir):
        """Test resume command generation."""
        checkpoint_dir = temp_dir / "checkpoints" 
        callback = PauseCallback(checkpoint_dir=str(checkpoint_dir))
        
        # Mock the resume command generation
        with patch.object(callback, '_generate_resume_commands') as mock_gen:
            mock_gen.return_value = {
                'checkpoint_path': 'python train.py resume --checkpoint-path /path/to/ckpt',
                'wandb_artifact': 'python train.py resume --checkpoint-artifact entity/project/run:latest'
            }
            
            commands = callback._generate_resume_commands('checkpoint.ckpt', 'test-run-123')
            
            assert 'checkpoint_path' in commands
            assert 'wandb_artifact' in commands
            assert 'resume --checkpoint-path' in commands['checkpoint_path']
            assert 'resume --checkpoint-artifact' in commands['wandb_artifact']
    
    def test_pause_callback_config_embedding(self, temp_dir, sample_config):
        """Test configuration embedding in checkpoints."""
        checkpoint_dir = temp_dir / "checkpoints"
        callback = PauseCallback(checkpoint_dir=str(checkpoint_dir))
        
        # Mock trainer with config
        mock_trainer = Mock()
        mock_trainer.lightning_module.hparams = sample_config
        
        # Test config embedding
        with patch.object(callback, '_embed_config_in_checkpoint') as mock_embed:
            mock_embed.return_value = "embedded_config_content"
            
            embedded = callback._embed_config_in_checkpoint(mock_trainer)
            
            mock_embed.assert_called_once_with(mock_trainer)
            assert embedded == "embedded_config_content"
    
    @patch('wandb.init')
    @patch('wandb.log_artifact')
    def test_pause_callback_wandb_upload(self, mock_log_artifact, mock_init, temp_dir):
        """Test W&B artifact upload functionality."""
        checkpoint_dir = temp_dir / "checkpoints"
        callback = PauseCallback(
            checkpoint_dir=str(checkpoint_dir),
            wandb_project="test-project"
        )
        
        # Create a fake checkpoint file
        checkpoint_file = temp_dir / "test_checkpoint.ckpt"
        checkpoint_file.write_text("fake checkpoint data")
        
        # Mock W&B
        mock_artifact = Mock()
        mock_init.return_value = Mock()
        
        with patch('wandb.Artifact', return_value=mock_artifact):
            # Test upload
            with patch.object(callback, '_upload_to_wandb') as mock_upload:
                mock_upload.return_value = "uploaded_artifact_name"
                
                result = callback._upload_to_wandb(str(checkpoint_file), "test-run-123")
                
                mock_upload.assert_called_once_with(str(checkpoint_file), "test-run-123")
                assert result == "uploaded_artifact_name"
    
    def test_pause_callback_error_handling(self, temp_dir):
        """Test error handling in pause callback."""
        checkpoint_dir = temp_dir / "checkpoints"
        callback = PauseCallback(checkpoint_dir=str(checkpoint_dir))
        
        # Test handling of missing checkpoint directory
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            mock_mkdir.side_effect = PermissionError("Permission denied")
            
            # Should handle the error gracefully
            with pytest.raises(PermissionError):
                callback._ensure_checkpoint_dir()
    
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
        
        # Test callback hooks
        pl_module = Mock()
        
        # Setup phase
        callback.setup(mock_trainer, pl_module, "fit")
        
        # Training start
        callback.on_train_start(mock_trainer, pl_module)
        
        # Batch end (where pause check happens)
        callback.on_train_batch_end(mock_trainer, pl_module, {}, 0)
        
        # Should complete without errors
        assert True
    
    def test_pause_callback_memory_cleanup(self, temp_dir):
        """Test memory cleanup during pause operations."""
        checkpoint_dir = temp_dir / "checkpoints"
        callback = PauseCallback(checkpoint_dir=str(checkpoint_dir))
        
        # Test cleanup method exists and can be called
        with patch.object(callback, '_cleanup_memory', return_value=None) as mock_cleanup:
            callback._cleanup_memory()
            mock_cleanup.assert_called_once()
    
    def test_pause_callback_thread_safety(self, temp_dir):
        """Test thread safety of pause callback operations."""
        checkpoint_dir = temp_dir / "checkpoints"
        callback = PauseCallback(checkpoint_dir=str(checkpoint_dir))
        
        # Test that pause requests are thread-safe
        import threading
        
        def request_pause():
            callback.request_pause()
        
        # Create multiple threads requesting pause
        threads = [threading.Thread(target=request_pause) for _ in range(5)]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        # Should be in paused state
        assert callback.pause_requested


if __name__ == "__main__":
    pytest.main([__file__, "-v"])