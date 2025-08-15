"""
Test suite for WandbArtifactCheckpoint callback.

This module tests the WandbArtifactCheckpoint functionality including:
- Basic instantiation and configuration
- PauseCallback integration
- Emergency checkpoint creation
- Upload scheduling logic
- State persistence
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, PropertyMock
import lightning.pytorch as pl
import torch

# Add the project to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from lightning_reflow.callbacks import WandbArtifactCheckpoint


class TestWandbArtifactCheckpointBasics:
    """Test basic functionality of WandbArtifactCheckpoint."""
    
    def test_instantiation_with_defaults(self):
        """Test that WandbArtifactCheckpoint can be instantiated with defaults."""
        callback = WandbArtifactCheckpoint()
        
        assert callback.upload_best_model is True
        assert callback.upload_last_model is True
        assert callback.upload_all_checkpoints is False
        assert callback.monitor_pause_checkpoints is True
        assert callback.use_compression is True
        assert callback.wandb_verbose is True
        
    def test_instantiation_with_custom_config(self):
        """Test WandbArtifactCheckpoint with custom configuration."""
        callback = WandbArtifactCheckpoint(
            upload_best_model=False,
            upload_last_model=True,
            upload_every_n_epoch=10,
            upload_every_n_validation=5,
            min_training_minutes=30.0,
            use_compression=False,
            upload_best_last_only_at_end=True,
            periodic_upload_pattern="timestamped"
        )
        
        assert callback.upload_best_model is False
        assert callback.upload_every_n_epoch == 10
        assert callback.upload_every_n_validation == 5
        assert callback.min_training_minutes == 30.0
        assert callback.use_compression is False
        assert callback.upload_best_last_only_at_end is True
        assert callback.periodic_upload_pattern == "timestamped"
    
    def test_required_methods_exist(self):
        """Test that all required Lightning callback methods exist."""
        callback = WandbArtifactCheckpoint()
        
        # Check Lightning callback methods
        assert hasattr(callback, 'on_fit_start')
        assert hasattr(callback, 'on_fit_end')
        assert hasattr(callback, 'on_exception')
        assert hasattr(callback, 'on_save_checkpoint')
        assert hasattr(callback, 'on_load_checkpoint')
        assert hasattr(callback, 'on_validation_end')
        assert hasattr(callback, 'on_train_epoch_end')
        assert hasattr(callback, 'on_train_batch_end')
        assert hasattr(callback, 'teardown')
        
        # Check custom methods
        assert hasattr(callback, 'upload_pause_checkpoint')
        assert hasattr(callback, '_create_emergency_checkpoint')
        assert hasattr(callback, '_save_comprehensive_checkpoint')


class TestWandbArtifactCheckpointPauseIntegration:
    """Test integration with PauseCallback."""
    
    def test_upload_pause_checkpoint_method_signature(self):
        """Test that upload_pause_checkpoint has the correct signature for PauseCallback."""
        callback = WandbArtifactCheckpoint()
        
        import inspect
        sig = inspect.signature(callback.upload_pause_checkpoint)
        params = list(sig.parameters.keys())
        
        # Should have trainer, pl_module, checkpoint_path parameters
        assert 'trainer' in params
        assert 'pl_module' in params
        assert 'checkpoint_path' in params
    
    @patch('lightning_reflow.callbacks.wandb.wandb_artifact_checkpoint.Path')
    def test_upload_pause_checkpoint_no_wandb_run(self, mock_path):
        """Test upload_pause_checkpoint returns None when no W&B run available."""
        callback = WandbArtifactCheckpoint(monitor_pause_checkpoints=True)
        callback._wandb_run_ref = None
        
        trainer = Mock(spec=pl.Trainer)
        pl_module = Mock(spec=pl.LightningModule)
        
        result = callback.upload_pause_checkpoint(trainer, pl_module, "/fake/checkpoint.ckpt")
        
        assert result is None
    
    def test_upload_pause_checkpoint_disabled(self):
        """Test upload_pause_checkpoint returns None when monitoring is disabled."""
        callback = WandbArtifactCheckpoint(monitor_pause_checkpoints=False)
        
        trainer = Mock(spec=pl.Trainer)
        pl_module = Mock(spec=pl.LightningModule)
        
        result = callback.upload_pause_checkpoint(trainer, pl_module, "/fake/checkpoint.ckpt")
        
        assert result is None
    
    @patch('lightning_reflow.callbacks.wandb.wandb_artifact_checkpoint.Path')
    def test_upload_pause_checkpoint_file_not_exists(self, mock_path_class):
        """Test upload_pause_checkpoint returns None when checkpoint doesn't exist."""
        mock_path_class.return_value.exists.return_value = False
        
        callback = WandbArtifactCheckpoint(monitor_pause_checkpoints=True)
        callback._wandb_run_ref = Mock()
        
        trainer = Mock(spec=pl.Trainer)
        pl_module = Mock(spec=pl.LightningModule)
        
        result = callback.upload_pause_checkpoint(trainer, pl_module, "/fake/checkpoint.ckpt")
        
        assert result is None


class TestWandbArtifactCheckpointEmergency:
    """Test emergency checkpoint functionality."""
    
    def test_create_emergency_checkpoint_basic(self):
        """Test basic emergency checkpoint creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = WandbArtifactCheckpoint(create_emergency_checkpoints=True)
            callback._model_checkpoint_ref = Mock()
            callback._model_checkpoint_ref.dirpath = tmpdir
            
            trainer = Mock(spec=pl.Trainer)
            trainer.current_epoch = 5
            trainer.global_step = 1000
            
            pl_module = Mock(spec=pl.LightningModule)
            
            # Mock the save_comprehensive_checkpoint method to create a file
            def mock_save(trainer, pl_module, path, reason, extra_metadata=None):
                # Create the file so the check passes
                Path(path).touch()
                Path(path).write_text("mock checkpoint content")
            
            with patch.object(callback, '_save_comprehensive_checkpoint', side_effect=mock_save):
                result = callback._create_emergency_checkpoint(trainer, pl_module, "test_reason")
                
                # Check that a path was returned
                assert result is not None
                assert "emergency-test_reason-epoch=5-step=1000.ckpt" in result
    
    def test_emergency_checkpoint_disabled(self):
        """Test that emergency checkpoints are not created when disabled."""
        callback = WandbArtifactCheckpoint(create_emergency_checkpoints=False)
        
        trainer = Mock(spec=pl.Trainer)
        trainer.is_global_zero = True
        pl_module = Mock(spec=pl.LightningModule)
        
        # This should not be called when create_emergency_checkpoints=False
        # The logic is in _upload_checkpoints_if_needed
        callback._has_uploaded = False
        callback._wandb_run_ref = Mock()
        callback._model_checkpoint_ref = Mock()
        callback._model_checkpoint_ref.best_model_path = "/fake/best.ckpt"
        callback._model_checkpoint_ref.last_model_path = "/fake/last.ckpt"
        callback._model_checkpoint_ref.best_model_score = torch.tensor(0.95)  # Use actual tensor
        callback._training_start_time = 0
        
        with patch.object(callback, '_create_emergency_checkpoint') as mock_create:
            with patch.object(callback, '_upload_single_checkpoint'):
                with patch('pathlib.Path.exists', return_value=True):
                    callback._upload_checkpoints_if_needed(trainer, pl_module, "normal completion")
                
                # Should not have called _create_emergency_checkpoint for normal completion
                mock_create.assert_not_called()


class TestWandbArtifactCheckpointUploadScheduling:
    """Test upload scheduling logic."""
    
    def test_upload_every_n_epoch_logic(self):
        """Test that upload_every_n_epoch works correctly."""
        callback = WandbArtifactCheckpoint(
            upload_every_n_epoch=5,
            upload_every_n_validation=None
        )
        
        trainer = Mock(spec=pl.Trainer)
        trainer.current_epoch = 4
        
        # First 4 epochs should not trigger upload
        callback._epoch_count = 4
        assert callback._should_upload_epoch(trainer) is False
        
        # 5th epoch should trigger
        callback._epoch_count = 4
        callback._next_checkpoint_epoch = 5
        trainer.current_epoch = 5
        assert callback._should_upload_epoch(trainer) is True
    
    def test_upload_every_n_validation_logic(self):
        """Test that upload_every_n_validation works correctly."""
        callback = WandbArtifactCheckpoint(
            upload_every_n_validation=10,
            upload_periodic_checkpoints=False
        )
        
        trainer = Mock(spec=pl.Trainer)
        trainer.global_step = 1000
        
        # Set up for 10th validation
        callback._validation_count = 9
        callback._next_checkpoint_step = 1000
        
        assert callback._should_upload_validation(trainer) is True
        
        # Not at checkpoint step yet
        trainer.global_step = 999
        assert callback._should_upload_validation(trainer) is False
    
    def test_upload_best_last_only_at_end(self):
        """Test that upload_best_last_only_at_end prevents uploads during training."""
        callback = WandbArtifactCheckpoint(
            upload_best_last_only_at_end=True,
            periodic_upload_pattern="timestamped"
        )
        
        # This configuration should use timestamped pattern for periodic uploads
        # and only upload best/last at the very end
        assert callback.upload_best_last_only_at_end is True
        assert callback.periodic_upload_pattern == "timestamped"


class TestWandbArtifactCheckpointStatePersistence:
    """Test state persistence for resume functionality."""
    
    def test_state_saved_in_checkpoint(self):
        """Test that callback state is saved in checkpoint."""
        callback = WandbArtifactCheckpoint()
        callback._validation_count = 15
        callback._epoch_count = 3
        callback._last_uploaded_epoch = 2
        callback._next_checkpoint_epoch = 5
        
        trainer = Mock(spec=pl.Trainer)
        trainer.current_epoch = 3
        trainer.global_step = 600
        
        pl_module = Mock(spec=pl.LightningModule)
        checkpoint = {}
        
        callback.on_save_checkpoint(trainer, pl_module, checkpoint)
        
        # Check that state was saved
        assert 'wandb_artifact_checkpoint_state' in checkpoint
        state = checkpoint['wandb_artifact_checkpoint_state']
        assert state['epoch_count'] == 3
        assert state['validation_count'] == 15
        assert state['next_checkpoint_epoch'] == 5
    
    def test_state_restored_from_checkpoint(self):
        """Test that callback state is restored from checkpoint."""
        callback = WandbArtifactCheckpoint()
        
        trainer = Mock(spec=pl.Trainer)
        pl_module = Mock(spec=pl.LightningModule)
        
        checkpoint = {
            'wandb_artifact_checkpoint_state': {
                'validation_count': 20,
                'last_checkpoint_epoch': 4,
                'last_checkpoint_step': 800,
                'next_checkpoint_epoch': 10,
                'next_checkpoint_step': 2000
            },
            'epoch': 4
        }
        
        callback.on_load_checkpoint(trainer, pl_module, checkpoint)
        
        # Check that state was restored
        assert callback._validation_count == 20
        assert callback._last_checkpoint_epoch == 4
        assert callback._last_checkpoint_step == 800
        assert callback._next_checkpoint_epoch == 10
        assert callback._next_checkpoint_step == 2000
        assert callback._epoch_count == 4  # From checkpoint['epoch']


class TestWandbArtifactCheckpointIntegration:
    """Integration tests with mocked components."""
    
    @patch('wandb.sdk.wandb_run.Run')
    @patch('lightning_reflow.callbacks.wandb.wandb_artifact_checkpoint.WandbArtifactManager')
    def test_full_training_lifecycle(self, mock_manager_class, mock_run_class):
        """Test callback through a simulated training lifecycle."""
        # Setup mocks
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        mock_manager.get_wandb_run.return_value = Mock()
        mock_manager.upload_checkpoint_artifact.return_value = "entity/project/artifact:v1"
        
        # Create callback
        callback = WandbArtifactCheckpoint(
            upload_best_model=True,
            upload_last_model=True,
            min_training_minutes=0.0  # Disable time threshold for testing
        )
        
        # Setup trainer and module
        trainer = Mock(spec=pl.Trainer)
        trainer.is_global_zero = True
        trainer.current_epoch = 0
        trainer.global_step = 0
        trainer.max_epochs = 10
        trainer.callbacks = [callback]
        trainer.logger = None
        
        pl_module = Mock(spec=pl.LightningModule)
        
        # Setup ModelCheckpoint mock
        model_checkpoint = Mock()
        model_checkpoint.best_model_path = "/fake/best.ckpt"
        model_checkpoint.last_model_path = "/fake/last.ckpt"
        model_checkpoint.best_model_score = torch.tensor(0.95)
        callback._model_checkpoint_ref = model_checkpoint
        
        # Simulate training lifecycle
        callback.on_fit_start(trainer, pl_module)
        assert callback._training_start_time is not None
        
        # Simulate some training progress
        trainer.current_epoch = 5
        trainer.global_step = 1000
        
        # Test validation end (should not upload if not scheduled)
        callback.on_validation_end(trainer, pl_module)
        
        # Simulate training end
        with patch.object(callback, '_is_training_complete', return_value=True):
            with patch.object(callback, '_is_pause_context', return_value=False):
                with patch('pathlib.Path.exists', return_value=True):
                    callback.on_fit_end(trainer, pl_module)
        
        # Verify upload was attempted
        assert mock_manager.upload_checkpoint_artifact.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])