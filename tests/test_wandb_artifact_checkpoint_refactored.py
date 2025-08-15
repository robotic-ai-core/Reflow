"""
Comprehensive tests for the refactored WandbArtifactCheckpoint callback.

This test suite ensures that the refactored version maintains 100% feature parity
with the original implementation while testing the new dataclass-based architecture.
"""

import pytest
import tempfile
import torch
import wandb
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, PropertyMock, call
from dataclasses import dataclass
import time
import gzip

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

# Import the refactored callback
from lightning_reflow.callbacks.wandb.wandb_artifact_checkpoint import (
    WandbArtifactCheckpoint,
    WandbCheckpointConfig,
    UploadState,
    UploadReason
)


class TestWandbCheckpointConfig:
    """Test the configuration dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = WandbCheckpointConfig()
        
        assert config.upload_best_model is True
        assert config.upload_last_model is True
        assert config.upload_all_checkpoints is False
        assert config.monitor_pause_checkpoints is True
        assert config.min_training_minutes == 5.0
        assert config.use_compression is True
        assert config.create_emergency_checkpoints is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = WandbCheckpointConfig(
            upload_best_model=False,
            upload_last_model=False,
            upload_all_checkpoints=True,
            min_training_minutes=10.0,
            upload_every_n_validation=5
        )
        
        assert config.upload_best_model is False
        assert config.upload_last_model is False
        assert config.upload_all_checkpoints is True
        assert config.min_training_minutes == 10.0
        assert config.upload_every_n_validation == 5


class TestUploadState:
    """Test the upload state dataclass."""
    
    def test_default_state(self):
        """Test default state values."""
        state = UploadState()
        
        assert state.has_uploaded is False
        assert state.training_start_time is None
        assert state.uploaded_pause_checkpoints == {}
        assert state.validation_count == 0
        assert state.epoch_count == 0
        assert state.last_uploaded_epoch == -1
    
    def test_state_modification(self):
        """Test state modification."""
        state = UploadState()
        
        state.has_uploaded = True
        state.training_start_time = 123.456
        state.uploaded_pause_checkpoints["checkpoint1.ckpt"] = "artifact1"
        state.validation_count = 5
        
        assert state.has_uploaded is True
        assert state.training_start_time == 123.456
        assert "checkpoint1.ckpt" in state.uploaded_pause_checkpoints
        assert state.validation_count == 5


class TestWandbArtifactCheckpointRefactored:
    """Test the main refactored callback functionality."""
    
    @pytest.fixture
    def mock_trainer(self):
        """Create a mock trainer."""
        trainer = Mock(spec=pl.Trainer)
        trainer.global_step = 100
        trainer.current_epoch = 5
        trainer.max_epochs = 10
        trainer.max_steps = 1000
        trainer.is_global_zero = True
        trainer.callbacks = []
        trainer.default_root_dir = "/tmp/test"
        trainer.logger = Mock(spec=WandbLogger)
        return trainer
    
    @pytest.fixture
    def mock_pl_module(self):
        """Create a mock Lightning module."""
        module = Mock(spec=pl.LightningModule)
        return module
    
    @pytest.fixture
    def mock_model_checkpoint(self):
        """Create a mock ModelCheckpoint callback."""
        checkpoint = Mock(spec=ModelCheckpoint)
        checkpoint.best_model_path = "/tmp/best.ckpt"
        checkpoint.last_model_path = "/tmp/last.ckpt"
        checkpoint.best_model_score = torch.tensor(0.95)
        checkpoint.dirpath = "/tmp/checkpoints"
        return checkpoint
    
    @pytest.fixture
    def callback_with_mocks(self, mock_trainer, mock_model_checkpoint):
        """Create callback with mocked dependencies."""
        callback = WandbArtifactCheckpoint(
            upload_best_model=True,
            upload_last_model=True,
            min_training_minutes=0.0  # Disable time check for tests
        )
        
        # Mock the model checkpoint reference
        callback._model_checkpoint_ref = mock_model_checkpoint
        
        # Mock the W&B run
        mock_run = Mock()
        mock_run.id = "test_run_id"
        callback._wandb_run_ref = mock_run
        
        # Mock the W&B manager
        callback._wandb_manager = Mock()
        callback._wandb_manager.upload_checkpoint_artifact = Mock(return_value="artifact_path")
        callback._wandb_manager.extract_score_from_trainer = Mock(return_value=0.95)
        callback._wandb_manager.get_wandb_run = Mock(return_value=mock_run)
        
        return callback
    
    def test_initialization(self):
        """Test callback initialization."""
        callback = WandbArtifactCheckpoint(
            upload_best_model=False,
            upload_last_model=True,
            upload_every_n_epoch=3
        )
        
        assert isinstance(callback.config, WandbCheckpointConfig)
        assert isinstance(callback.state, UploadState)
        assert callback.config.upload_best_model is False
        assert callback.config.upload_last_model is True
        assert callback.config.upload_every_n_epoch == 3
    
    def test_on_fit_start(self, callback_with_mocks, mock_trainer, mock_pl_module):
        """Test on_fit_start hook."""
        with patch('time.time', return_value=1000.0):
            callback_with_mocks.on_fit_start(mock_trainer, mock_pl_module)
        
        assert callback_with_mocks.state.training_start_time == 1000.0
        assert callback_with_mocks._wandb_run_ref is not None
    
    def test_on_fit_end_normal(self, callback_with_mocks, mock_trainer, mock_pl_module):
        """Test on_fit_end uploads checkpoints."""
        callback_with_mocks.state.training_start_time = time.time() - 400  # 6+ minutes
        mock_trainer.global_step = 999  # Near max_steps
        
        with patch.object(callback_with_mocks, '_upload_checkpoints') as mock_upload:
            callback_with_mocks.on_fit_end(mock_trainer, mock_pl_module)
            mock_upload.assert_called_once_with(
                mock_trainer, mock_pl_module, UploadReason.NORMAL_COMPLETION
            )
    
    def test_on_exception_handling(self, callback_with_mocks, mock_trainer, mock_pl_module):
        """Test exception handling with emergency checkpoint."""
        callback_with_mocks.config.upload_on_exception = True
        exception = RuntimeError("Test exception")
        
        with patch.object(callback_with_mocks, '_upload_checkpoints') as mock_upload:
            callback_with_mocks.on_exception(mock_trainer, mock_pl_module, exception)
            mock_upload.assert_called_once_with(
                mock_trainer, mock_pl_module, UploadReason.EXCEPTION
            )
    
    def test_upload_pause_checkpoint(self, callback_with_mocks, mock_trainer, mock_pl_module):
        """Test pause checkpoint upload."""
        checkpoint_path = "/tmp/pause.ckpt"
        
        # Create a temporary file to simulate checkpoint
        with tempfile.NamedTemporaryFile(suffix='.ckpt', delete=False) as f:
            f.write(b"checkpoint_data")
            checkpoint_path = f.name
        
        try:
            result = callback_with_mocks.upload_pause_checkpoint(
                mock_trainer, mock_pl_module, checkpoint_path
            )
            
            assert result == "artifact_path"
            assert checkpoint_path in callback_with_mocks.state.uploaded_pause_checkpoints
            
            # Test cached upload (should not upload again)
            callback_with_mocks._wandb_manager.upload_checkpoint_artifact.reset_mock()
            result2 = callback_with_mocks.upload_pause_checkpoint(
                mock_trainer, mock_pl_module, checkpoint_path
            )
            assert result2 == "artifact_path"
            callback_with_mocks._wandb_manager.upload_checkpoint_artifact.assert_not_called()
            
        finally:
            Path(checkpoint_path).unlink(missing_ok=True)
    
    def test_periodic_validation_upload(self, callback_with_mocks, mock_trainer, mock_pl_module):
        """Test periodic validation uploads."""
        callback_with_mocks.config.upload_every_n_validation = 2
        callback_with_mocks.config.upload_periodic_checkpoints = False  # Disable to test every_n logic
        
        with patch.object(callback_with_mocks, '_upload_periodic_checkpoints') as mock_upload:
            # First validation - no upload
            callback_with_mocks.on_validation_end(mock_trainer, mock_pl_module)
            mock_upload.assert_not_called()
            
            # Second validation - should upload because count % 2 == 0
            callback_with_mocks.on_validation_end(mock_trainer, mock_pl_module)
            mock_upload.assert_called_once_with(
                mock_trainer, mock_pl_module, UploadReason.PERIODIC_VALIDATION
            )
    
    def test_periodic_epoch_upload(self, callback_with_mocks, mock_trainer, mock_pl_module):
        """Test periodic epoch uploads."""
        callback_with_mocks.config.upload_every_n_epoch = 3
        
        with patch.object(callback_with_mocks, '_upload_periodic_checkpoints') as mock_upload:
            # Simulate 3 epochs
            for _ in range(2):
                callback_with_mocks.on_train_epoch_end(mock_trainer, mock_pl_module)
            mock_upload.assert_not_called()
            
            # Third epoch - should upload
            callback_with_mocks.on_train_epoch_end(mock_trainer, mock_pl_module)
            mock_upload.assert_called_once()
    
    def test_compression_enabled(self, callback_with_mocks):
        """Test checkpoint compression."""
        callback_with_mocks.config.use_compression = True
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.ckpt', delete=False) as f:
            f.write(b"uncompressed_data" * 100)
            filepath = f.name
        
        try:
            compressed_path = callback_with_mocks._prepare_upload_path(filepath)
            
            assert compressed_path != filepath
            assert compressed_path.endswith('.ckpt.gz')
            
            # Verify it's actually compressed
            with gzip.open(compressed_path, 'rb') as f:
                data = f.read()
                assert b"uncompressed_data" in data
            
        finally:
            Path(filepath).unlink(missing_ok=True)
            if compressed_path != filepath:
                Path(compressed_path).unlink(missing_ok=True)
    
    def test_compression_disabled(self, callback_with_mocks):
        """Test no compression when disabled."""
        callback_with_mocks.config.use_compression = False
        
        filepath = "/tmp/test.ckpt"
        result = callback_with_mocks._prepare_upload_path(filepath)
        assert result == filepath
    
    def test_upload_all_checkpoints(self, callback_with_mocks, mock_trainer, mock_pl_module):
        """Test uploading all checkpoints."""
        callback_with_mocks.config.upload_all_checkpoints = True
        
        # Create mock checkpoint files
        checkpoint_dir = Path("/tmp/checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        ckpt_files = []
        for i in range(3):
            ckpt_path = checkpoint_dir / f"checkpoint-{i}.ckpt"
            ckpt_path.touch()
            ckpt_files.append(ckpt_path)
        
        try:
            with patch.object(Path, 'glob', return_value=ckpt_files):
                with patch.object(Path, 'exists', return_value=True):
                    uploaded = callback_with_mocks._upload_all_checkpoints(
                        mock_trainer, mock_pl_module, UploadReason.NORMAL_COMPLETION
                    )
                    
                    # Should have uploaded all checkpoints
                    assert len(uploaded) == 3
        
        finally:
            for f in ckpt_files:
                f.unlink(missing_ok=True)
    
    def test_emergency_checkpoint_creation(self, callback_with_mocks, mock_trainer, mock_pl_module):
        """Test emergency checkpoint creation."""
        callback_with_mocks.config.create_emergency_checkpoints = True
        
        with patch('lightning_reflow.utils.checkpoint.checkpoint_utils.save_comprehensive_checkpoint') as mock_save:
            emergency_path = callback_with_mocks._create_emergency_checkpoint(
                mock_trainer, mock_pl_module, "test_reason"
            )
            
            # Should have called save_comprehensive_checkpoint
            mock_save.assert_called_once()
            call_args = mock_save.call_args
            assert "emergency" in call_args[0][2]  # checkpoint path
            assert call_args[1]['reason'] == "emergency_test_reason"
    
    def test_state_persistence(self, callback_with_mocks, mock_trainer, mock_pl_module):
        """Test state saving and loading from checkpoint."""
        # Set some state
        callback_with_mocks.state.has_uploaded = True
        callback_with_mocks.state.training_start_time = 123.456
        callback_with_mocks.state.validation_count = 10
        callback_with_mocks.state.uploaded_pause_checkpoints["test.ckpt"] = "artifact1"
        
        # Save checkpoint
        checkpoint = {}
        callback_with_mocks.on_save_checkpoint(mock_trainer, mock_pl_module, checkpoint)
        
        assert 'wandb_artifact_checkpoint_state' in checkpoint
        saved_state = checkpoint['wandb_artifact_checkpoint_state']
        assert saved_state['version'] == '2.0'
        assert saved_state['state']['has_uploaded'] is True
        assert saved_state['state']['validation_count'] == 10
        
        # Create a new callback and restore state
        new_callback = WandbArtifactCheckpoint()
        new_callback.on_load_checkpoint(mock_trainer, mock_pl_module, checkpoint)
        
        assert new_callback.state.has_uploaded is True
        assert new_callback.state.training_start_time == 123.456
        assert new_callback.state.validation_count == 10
        assert "test.ckpt" in new_callback.state.uploaded_pause_checkpoints
    
    def test_pause_context_detection(self, callback_with_mocks, mock_trainer):
        """Test detection of pause context."""
        # No pause callback
        assert callback_with_mocks._is_pause_context(mock_trainer) is False
        
        # Add a mock pause callback
        pause_callback = Mock()
        pause_callback.__class__.__name__ = 'PauseCallback'
        pause_callback.is_pausing = Mock(return_value=True)
        pause_callback.is_pause_scheduled = Mock(return_value=False)
        mock_trainer.callbacks = [pause_callback]
        
        assert callback_with_mocks._is_pause_context(mock_trainer) is True
        
        # Test with pause scheduled
        pause_callback.is_pausing = Mock(return_value=False)
        pause_callback.is_pause_scheduled = Mock(return_value=True)
        assert callback_with_mocks._is_pause_context(mock_trainer) is True
    
    def test_training_complete_detection(self, callback_with_mocks, mock_trainer):
        """Test detection of training completion."""
        # Not complete
        mock_trainer.global_step = 500
        mock_trainer.current_epoch = 5
        assert callback_with_mocks._is_training_complete(mock_trainer) is False
        
        # Complete by steps
        mock_trainer.global_step = 999
        assert callback_with_mocks._is_training_complete(mock_trainer) is True
        
        # Complete by epochs
        mock_trainer.global_step = 500
        mock_trainer.current_epoch = 9
        assert callback_with_mocks._is_training_complete(mock_trainer) is True
    
    def test_duplicate_upload_prevention(self, callback_with_mocks):
        """Test prevention of duplicate uploads."""
        uploaded = [
            {'filepath': '/tmp/checkpoint1.ckpt', 'type': 'best'},
            {'filepath': '/tmp/checkpoint2.ckpt', 'type': 'latest'}
        ]
        
        assert callback_with_mocks._is_duplicate_upload('/tmp/checkpoint1.ckpt', uploaded) is True
        assert callback_with_mocks._is_duplicate_upload('/tmp/checkpoint3.ckpt', uploaded) is False
    
    def test_storage_optimization_mode(self, callback_with_mocks, mock_trainer, mock_pl_module):
        """Test storage optimization with periodic uploads."""
        callback_with_mocks.config.upload_best_last_only_at_end = True
        callback_with_mocks.config.periodic_upload_pattern = "timestamped"
        
        checkpoints = callback_with_mocks._get_periodic_checkpoints_to_upload()
        
        # Should return timestamped checkpoint
        assert len(checkpoints) == 1
        assert checkpoints[0][1].startswith("periodic_")
    
    def test_alias_creation(self, callback_with_mocks):
        """Test alias creation for different upload reasons."""
        # Normal completion
        aliases = callback_with_mocks._create_aliases("best", UploadReason.NORMAL_COMPLETION)
        assert "best" in aliases
        assert "latest" in aliases
        
        # Exception
        aliases = callback_with_mocks._create_aliases("emergency", UploadReason.EXCEPTION)
        assert "emergency" in aliases
        assert "crash_recovery" in aliases
        assert "latest" in aliases
        
        # Pause
        aliases = callback_with_mocks._create_aliases("pause", UploadReason.PAUSE_REQUESTED)
        assert "pause" in aliases
        assert "latest" in aliases
        
        # Periodic
        aliases = callback_with_mocks._create_aliases("periodic", UploadReason.PERIODIC_VALIDATION)
        assert "periodic" in aliases
        assert "latest" in aliases
    
    def test_metadata_generation(self, callback_with_mocks):
        """Test metadata generation for uploads."""
        callback_with_mocks.state.training_start_time = time.time() - 600  # 10 minutes ago
        callback_with_mocks.config.model_checkpoint_monitor_metric = "val_loss"
        
        metadata = callback_with_mocks._get_extra_metadata(UploadReason.NORMAL_COMPLETION)
        
        assert metadata['upload_reason'] == "normal_completion"
        assert metadata['artifact_type'] == "model"
        assert metadata['compressed'] is True
        assert metadata['monitored_metric'] == "val_loss"
        assert 9.5 < metadata['training_duration_minutes'] < 10.5  # Approximately 10 minutes
    
    def test_epoch_step_resolution(self, callback_with_mocks, mock_trainer):
        """Test extraction of epoch and step from checkpoint paths."""
        # Test with standard naming
        epoch, step = callback_with_mocks._resolve_epoch_step(
            "/tmp/checkpoint-epoch=3-step=1000.ckpt", mock_trainer
        )
        assert epoch == 3
        assert step == 1000
        
        # Test with underscore separator
        epoch, step = callback_with_mocks._resolve_epoch_step(
            "/tmp/checkpoint-epoch_5-step_2000.ckpt", mock_trainer
        )
        assert epoch == 5
        assert step == 2000
        
        # Test fallback to trainer values
        epoch, step = callback_with_mocks._resolve_epoch_step(
            "/tmp/checkpoint.ckpt", mock_trainer
        )
        assert epoch == mock_trainer.current_epoch
        assert step == mock_trainer.global_step
    
    def test_minimum_training_time_check(self, callback_with_mocks):
        """Test minimum training time requirements."""
        # No training start time - should allow
        assert callback_with_mocks._has_sufficient_training_time(UploadReason.NORMAL_COMPLETION) is True
        
        # Set training start time
        callback_with_mocks.state.training_start_time = time.time()
        
        # Too short for normal completion
        callback_with_mocks.config.min_training_minutes = 5.0
        assert callback_with_mocks._has_sufficient_training_time(UploadReason.NORMAL_COMPLETION) is False
        
        # Exception should use different threshold
        callback_with_mocks.config.min_training_minutes_for_exceptions = 0.0
        assert callback_with_mocks._has_sufficient_training_time(UploadReason.EXCEPTION) is True
        
        # Simulate enough time passed
        callback_with_mocks.state.training_start_time = time.time() - 360  # 6 minutes ago
        assert callback_with_mocks._has_sufficient_training_time(UploadReason.NORMAL_COMPLETION) is True


class TestIntegrationWithLightningReflow:
    """Test integration with Lightning Reflow components."""
    
    def test_automatic_addition_with_wandb_logger(self):
        """Test automatic addition when W&B logger is present."""
        from lightning_reflow.core.shared_config import ensure_essential_callbacks
        
        trainer = Mock()
        trainer.logger = Mock(spec=WandbLogger)
        
        callbacks = []
        updated_callbacks = ensure_essential_callbacks(callbacks, trainer)
        
        # Should have added WandbArtifactCheckpoint
        wandb_callbacks = [cb for cb in updated_callbacks 
                          if isinstance(cb, WandbArtifactCheckpoint)]
        assert len(wandb_callbacks) == 1
        
        # Check default configuration
        wandb_cb = wandb_callbacks[0]
        assert wandb_cb.config.upload_best_model is True
        assert wandb_cb.config.upload_last_model is True
        assert wandb_cb.config.monitor_pause_checkpoints is True
    
    def test_no_duplicate_addition(self):
        """Test that callback isn't added twice."""
        from lightning_reflow.core.shared_config import ensure_essential_callbacks
        
        trainer = Mock()
        trainer.logger = Mock(spec=WandbLogger)
        
        # Already has WandbArtifactCheckpoint
        existing_callback = WandbArtifactCheckpoint()
        callbacks = [existing_callback]
        
        updated_callbacks = ensure_essential_callbacks(callbacks, trainer)
        
        # Should still have only one WandbArtifactCheckpoint
        wandb_callbacks = [cb for cb in updated_callbacks 
                          if isinstance(cb, WandbArtifactCheckpoint)]
        assert len(wandb_callbacks) == 1
        assert wandb_callbacks[0] is existing_callback


class TestManagerStateIntegration:
    """Test integration with manager state system."""
    
    def test_manager_registration(self, monkeypatch):
        """Test registration with manager state system."""
        mock_register = Mock()
        
        with patch('lightning_reflow.utils.checkpoint.manager_state.register_manager', mock_register):
            callback = WandbArtifactCheckpoint()
            
            # Should have called register_manager
            mock_register.assert_called_once()
            
            # Get the registered state object
            state_obj = mock_register.call_args[0][0]
            assert state_obj.manager_name == "wandb_artifact_checkpoint"
    
    def test_manager_state_capture_restore(self):
        """Test state capture and restore through manager interface."""
        with patch('lightning_reflow.utils.checkpoint.manager_state.register_manager'):
            # Import at the top of the test function to avoid scope issues
            from lightning_reflow.callbacks.wandb.wandb_artifact_checkpoint import WandbArtifactCheckpoint
            
            callback = WandbArtifactCheckpoint()
            
            # Set some state
            callback.state.has_uploaded = True
            callback.state.validation_count = 15
            callback.config.upload_every_n_epoch = 3
            
            # The state manager is created in _register_for_state_persistence
            # We'll test it through the checkpoint save/load mechanism
            trainer = Mock()
            pl_module = Mock()
            checkpoint = {}
            
            # Save state
            callback.on_save_checkpoint(trainer, pl_module, checkpoint)
            
            # Create new callback and restore
            new_callback = WandbArtifactCheckpoint()
            new_callback.on_load_checkpoint(trainer, pl_module, checkpoint)
            
            # Verify state was restored
            assert new_callback.state.has_uploaded is True
            assert new_callback.state.validation_count == 15
            # Note: config is not restored, only state


if __name__ == "__main__":
    pytest.main([__file__, "-v"])