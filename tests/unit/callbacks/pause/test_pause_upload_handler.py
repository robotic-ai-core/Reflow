"""
Unit tests for PauseUploadHandler.

Tests W&B artifact upload, fallback handling, and error scenarios.
"""

import pytest
import torch
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from lightning_reflow.callbacks.pause.pause_upload_handler import PauseUploadHandler
from lightning_reflow.utils.wandb.wandb_artifact_manager import WandbArtifactManager


class TestPauseUploadHandler:
    """Test PauseUploadHandler functionality."""

    @pytest.fixture
    def mock_wandb_manager(self):
        """Create a mock WandbArtifactManager."""
        manager = Mock(spec=WandbArtifactManager)
        manager.get_wandb_run.return_value = Mock()
        manager.upload_checkpoint_artifact.return_value = "entity/project/artifact:v1"
        return manager

    @pytest.fixture
    def upload_handler(self, mock_wandb_manager):
        """Create an upload handler for testing."""
        return PauseUploadHandler(mock_wandb_manager)

    @pytest.fixture
    def mock_trainer(self):
        """Create a mock trainer for testing."""
        trainer = Mock()
        trainer.current_epoch = 5
        trainer.global_step = 100
        trainer.callbacks = []
        trainer.lightning_module = Mock()
        return trainer

    @pytest.fixture
    def checkpoint_file(self, temp_dir):
        """Create a dummy checkpoint file."""
        checkpoint_path = temp_dir / "test_checkpoint.ckpt"
        checkpoint = {
            'state_dict': {'layer.weight': torch.randn(10, 10)},
            'epoch': 5,
            'global_step': 100,
        }
        torch.save(checkpoint, checkpoint_path)
        return str(checkpoint_path)

    def test_initialization(self, mock_wandb_manager):
        """Test upload handler initialization."""
        handler = PauseUploadHandler(mock_wandb_manager)
        assert handler._wandb_manager is mock_wandb_manager

    def test_handle_wandb_upload_no_callback(self, upload_handler, mock_trainer, checkpoint_file):
        """Test upload when no W&B callback is available."""
        mock_trainer.callbacks = []

        result = upload_handler.handle_wandb_upload(mock_trainer, Mock(), checkpoint_file)

        assert result is None

    def test_handle_wandb_upload_with_upload_pause_checkpoint(self, upload_handler, mock_trainer, checkpoint_file):
        """Test upload with callback that has upload_pause_checkpoint method."""
        wandb_callback = Mock()
        wandb_callback.upload_pause_checkpoint.return_value = "entity/project/artifact:v1"
        mock_trainer.callbacks = [wandb_callback]

        result = upload_handler.handle_wandb_upload(mock_trainer, Mock(), checkpoint_file)

        assert result == "entity/project/artifact:v1"
        wandb_callback.upload_pause_checkpoint.assert_called_once()

    def test_handle_wandb_upload_with_legacy_method(self, upload_handler, mock_trainer, checkpoint_file, mock_wandb_manager):
        """Test upload with callback that has legacy _upload_pause_checkpoint_artifact method."""
        wandb_callback = Mock(spec=['_upload_pause_checkpoint_artifact'])
        # Remove upload_pause_checkpoint to force legacy path
        del wandb_callback.upload_pause_checkpoint
        mock_trainer.callbacks = [wandb_callback]

        result = upload_handler.handle_wandb_upload(mock_trainer, Mock(), checkpoint_file)

        assert result == "entity/project/artifact:v1"
        mock_wandb_manager.upload_checkpoint_artifact.assert_called_once()

    def test_handle_wandb_upload_none_trainer(self, upload_handler, checkpoint_file):
        """Test upload fails with None trainer."""
        with pytest.raises(ValueError, match="Trainer cannot be None"):
            upload_handler.handle_wandb_upload(None, Mock(), checkpoint_file)

    def test_handle_wandb_upload_no_callbacks_attribute(self, upload_handler, checkpoint_file):
        """Test upload fails when trainer has no callbacks attribute."""
        trainer = Mock(spec=[])  # No callbacks attribute

        with pytest.raises(ValueError, match="Trainer must have callbacks list"):
            upload_handler.handle_wandb_upload(trainer, Mock(), checkpoint_file)

    def test_handle_wandb_upload_callback_raises_value_error(self, upload_handler, mock_trainer, checkpoint_file):
        """Test upload gracefully handles ValueError from callback."""
        wandb_callback = Mock()
        wandb_callback.upload_pause_checkpoint.side_effect = ValueError("Test error")
        mock_trainer.callbacks = [wandb_callback]

        result = upload_handler.handle_wandb_upload(mock_trainer, Mock(), checkpoint_file)

        assert result is None

    def test_handle_wandb_upload_callback_raises_runtime_error(self, upload_handler, mock_trainer, checkpoint_file):
        """Test upload gracefully handles RuntimeError from callback."""
        wandb_callback = Mock()
        wandb_callback.upload_pause_checkpoint.side_effect = RuntimeError("Test error")
        mock_trainer.callbacks = [wandb_callback]

        result = upload_handler.handle_wandb_upload(mock_trainer, Mock(), checkpoint_file)

        assert result is None

    def test_handle_wandb_upload_callback_raises_unexpected_error(self, upload_handler, mock_trainer, checkpoint_file):
        """Test upload re-raises unexpected exceptions."""
        wandb_callback = Mock()
        wandb_callback.upload_pause_checkpoint.side_effect = Exception("Unexpected error")
        mock_trainer.callbacks = [wandb_callback]

        with pytest.raises(RuntimeError, match="Unexpected error during W&B upload"):
            upload_handler.handle_wandb_upload(mock_trainer, Mock(), checkpoint_file)


class TestUploadPauseCheckpointArtifact:
    """Test _upload_pause_checkpoint_artifact method."""

    @pytest.fixture
    def mock_wandb_manager(self):
        """Create a mock WandbArtifactManager."""
        manager = Mock(spec=WandbArtifactManager)
        manager.get_wandb_run.return_value = Mock()
        manager.upload_checkpoint_artifact.return_value = "entity/project/artifact:v1"
        return manager

    @pytest.fixture
    def upload_handler(self, mock_wandb_manager):
        """Create an upload handler for testing."""
        return PauseUploadHandler(mock_wandb_manager)

    @pytest.fixture
    def mock_trainer(self):
        """Create a mock trainer for testing."""
        trainer = Mock()
        trainer.current_epoch = 5
        trainer.global_step = 100
        trainer.lightning_module = Mock()
        return trainer

    @pytest.fixture
    def checkpoint_file(self, temp_dir):
        """Create a dummy checkpoint file."""
        checkpoint_path = temp_dir / "test_checkpoint.ckpt"
        torch.save({'state_dict': {}}, checkpoint_path)
        return str(checkpoint_path)

    def test_upload_artifact_success(self, upload_handler, mock_trainer, checkpoint_file, mock_wandb_manager):
        """Test successful artifact upload."""
        result = upload_handler._upload_pause_checkpoint_artifact(
            Mock(), mock_trainer, checkpoint_file
        )

        assert result == "entity/project/artifact:v1"
        mock_wandb_manager.upload_checkpoint_artifact.assert_called_once()

    def test_upload_artifact_none_trainer(self, upload_handler, checkpoint_file):
        """Test upload fails with None trainer."""
        with pytest.raises(ValueError, match="Trainer cannot be None"):
            upload_handler._upload_pause_checkpoint_artifact(Mock(), None, checkpoint_file)

    def test_upload_artifact_nonexistent_checkpoint(self, upload_handler, mock_trainer):
        """Test upload fails with non-existent checkpoint."""
        with pytest.raises(ValueError, match="Checkpoint path does not exist"):
            upload_handler._upload_pause_checkpoint_artifact(
                Mock(), mock_trainer, "/nonexistent/path.ckpt"
            )

    def test_upload_artifact_no_wandb_run(self, upload_handler, mock_trainer, checkpoint_file, mock_wandb_manager):
        """Test upload fails when no W&B run is active."""
        mock_wandb_manager.get_wandb_run.return_value = None

        with pytest.raises(RuntimeError, match="No active W&B run found"):
            upload_handler._upload_pause_checkpoint_artifact(
                Mock(), mock_trainer, checkpoint_file
            )

    def test_upload_artifact_no_lightning_module(self, upload_handler, checkpoint_file, mock_wandb_manager):
        """Test upload fails when trainer has no lightning_module."""
        trainer = Mock()
        trainer.lightning_module = None
        mock_wandb_manager.get_wandb_run.return_value = Mock()

        with pytest.raises(ValueError, match="Trainer must have a valid lightning_module"):
            upload_handler._upload_pause_checkpoint_artifact(
                Mock(), trainer, checkpoint_file
            )

    def test_upload_artifact_returns_none(self, upload_handler, mock_trainer, checkpoint_file, mock_wandb_manager):
        """Test upload fails when artifact manager returns None."""
        mock_wandb_manager.upload_checkpoint_artifact.return_value = None

        with pytest.raises(RuntimeError, match="Artifact upload returned None"):
            upload_handler._upload_pause_checkpoint_artifact(
                Mock(), mock_trainer, checkpoint_file
            )


class TestHandleUploadWithFallback:
    """Test handle_upload_with_fallback method."""

    @pytest.fixture
    def mock_wandb_manager(self):
        """Create a mock WandbArtifactManager."""
        return Mock(spec=WandbArtifactManager)

    @pytest.fixture
    def upload_handler(self, mock_wandb_manager):
        """Create an upload handler for testing."""
        return PauseUploadHandler(mock_wandb_manager)

    @pytest.fixture
    def mock_trainer(self):
        """Create a mock trainer for testing."""
        trainer = Mock()
        trainer.callbacks = []
        return trainer

    def test_fallback_success(self, upload_handler, mock_trainer):
        """Test fallback returns artifact path on success."""
        wandb_callback = Mock()
        wandb_callback.upload_pause_checkpoint.return_value = "entity/project/artifact:v1"
        mock_trainer.callbacks = [wandb_callback]

        result = upload_handler.handle_upload_with_fallback(
            mock_trainer, Mock(), "checkpoint.ckpt"
        )

        assert result == "entity/project/artifact:v1"

    def test_fallback_no_callback(self, upload_handler, mock_trainer):
        """Test fallback returns None when no callback."""
        result = upload_handler.handle_upload_with_fallback(
            mock_trainer, Mock(), "checkpoint.ckpt"
        )

        assert result is None

    def test_fallback_handles_value_error(self, upload_handler, mock_trainer):
        """Test fallback handles ValueError gracefully."""
        wandb_callback = Mock()
        wandb_callback.upload_pause_checkpoint.side_effect = ValueError("Test")
        mock_trainer.callbacks = [wandb_callback]

        result = upload_handler.handle_upload_with_fallback(
            mock_trainer, Mock(), "checkpoint.ckpt"
        )

        assert result is None

    def test_fallback_handles_runtime_error(self, upload_handler, mock_trainer):
        """Test fallback handles RuntimeError gracefully."""
        wandb_callback = Mock()
        wandb_callback.upload_pause_checkpoint.side_effect = RuntimeError("Test")
        mock_trainer.callbacks = [wandb_callback]

        result = upload_handler.handle_upload_with_fallback(
            mock_trainer, Mock(), "checkpoint.ckpt"
        )

        assert result is None

    def test_fallback_handles_unexpected_error(self, upload_handler, mock_trainer):
        """Test fallback handles unexpected errors gracefully."""
        with patch.object(upload_handler, 'handle_wandb_upload', side_effect=Exception("Unexpected")):
            result = upload_handler.handle_upload_with_fallback(
                mock_trainer, Mock(), "checkpoint.ckpt"
            )

        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
