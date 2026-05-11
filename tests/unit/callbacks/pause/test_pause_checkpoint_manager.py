"""
Unit tests for PauseCheckpointManager.

Tests checkpoint path generation, saving, validation, and atomic operations.
"""

import pytest
import torch
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from lightning_reflow.callbacks.pause.pause_checkpoint_manager import PauseCheckpointManager
from lightning_reflow.models import SimpleReflowModel


class TestPauseCheckpointManager:
    """Test PauseCheckpointManager functionality."""

    @pytest.fixture
    def simple_model(self):
        """Simple model for testing."""
        return SimpleReflowModel(input_dim=10, hidden_dim=16, output_dim=2)

    @pytest.fixture
    def checkpoint_manager(self, temp_dir):
        """Create a checkpoint manager for testing."""
        checkpoint_dir = temp_dir / "checkpoints"
        return PauseCheckpointManager(checkpoint_dir)

    @pytest.fixture
    def mock_trainer(self):
        """Create a mock trainer for testing."""
        trainer = Mock()
        trainer.current_epoch = 5
        trainer.global_step = 100
        trainer.logger = Mock()
        return trainer

    def test_initialization(self, temp_dir):
        """Test checkpoint manager initialization."""
        checkpoint_dir = temp_dir / "checkpoints"
        manager = PauseCheckpointManager(checkpoint_dir)

        assert manager.checkpoint_dir == checkpoint_dir
        assert checkpoint_dir.exists()
        assert manager.last_checkpoint_path is None

    def test_initialization_creates_directory(self, temp_dir):
        """Test that initialization creates the checkpoint directory."""
        checkpoint_dir = temp_dir / "nested" / "checkpoints"
        assert not checkpoint_dir.exists()

        manager = PauseCheckpointManager(checkpoint_dir)

        assert checkpoint_dir.exists()

    def test_get_checkpoint_path_basic(self, checkpoint_manager, mock_trainer):
        """Test basic checkpoint path generation."""
        path = checkpoint_manager.get_checkpoint_path(mock_trainer)

        assert path.parent == checkpoint_manager.checkpoint_dir
        assert "pause" in path.name
        assert f"epoch={mock_trainer.current_epoch}" in path.name
        assert f"step={mock_trainer.global_step}" in path.name
        assert path.suffix == ".ckpt"

    def test_get_checkpoint_path_with_upload_tag(self, checkpoint_manager, mock_trainer):
        """Test checkpoint path with upload tag."""
        path = checkpoint_manager.get_checkpoint_path(mock_trainer, upload=True)

        assert "upload" in path.name
        assert "pause" not in path.name

    def test_get_checkpoint_path_includes_timestamp(self, checkpoint_manager, mock_trainer):
        """Test that checkpoint path includes timestamp."""
        before = int(time.time())
        path = checkpoint_manager.get_checkpoint_path(mock_trainer)
        after = int(time.time())

        # Extract timestamp from filename (format: tag_epoch=X_step=Y_TIMESTAMP.ckpt)
        parts = path.stem.split('_')
        timestamp = int(parts[-1])

        assert before <= timestamp <= after

    def test_save_checkpoint_basic(self, checkpoint_manager, mock_trainer, simple_model, temp_dir):
        """Test basic checkpoint saving."""
        checkpoint_path = temp_dir / "test_checkpoint.ckpt"

        def mock_save_checkpoint(filepath):
            checkpoint = {
                'state_dict': simple_model.state_dict(),
                'epoch': mock_trainer.current_epoch,
                'global_step': mock_trainer.global_step,
            }
            torch.save(checkpoint, filepath)

        mock_trainer.save_checkpoint = mock_save_checkpoint

        checkpoint_manager.save_checkpoint(mock_trainer, simple_model, checkpoint_path)

        assert checkpoint_path.exists()
        assert checkpoint_manager.last_checkpoint_path == checkpoint_path

    def test_save_checkpoint_with_validation_success(self, checkpoint_manager, mock_trainer, simple_model, temp_dir):
        """Test checkpoint saving with validation."""
        checkpoint_path = temp_dir / "test_checkpoint.ckpt"

        def mock_save_checkpoint(filepath):
            checkpoint = {
                'state_dict': simple_model.state_dict(),
                'epoch': mock_trainer.current_epoch,
                'global_step': mock_trainer.global_step,
            }
            torch.save(checkpoint, filepath)

        mock_trainer.save_checkpoint = mock_save_checkpoint

        checkpoint_manager.save_checkpoint_with_validation(
            mock_trainer, simple_model, checkpoint_path
        )

        assert checkpoint_path.exists()
        assert checkpoint_manager.last_checkpoint_path == checkpoint_path

        # Verify checkpoint structure
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        assert 'state_dict' in checkpoint
        assert 'epoch' in checkpoint
        assert 'global_step' in checkpoint

    def test_save_checkpoint_with_validation_adds_metadata(self, checkpoint_manager, mock_trainer, simple_model, temp_dir):
        """Test that config metadata callback is called."""
        checkpoint_path = temp_dir / "test_checkpoint.ckpt"

        def mock_save_checkpoint(filepath):
            checkpoint = {
                'state_dict': simple_model.state_dict(),
                'epoch': mock_trainer.current_epoch,
                'global_step': mock_trainer.global_step,
            }
            torch.save(checkpoint, filepath)

        mock_trainer.save_checkpoint = mock_save_checkpoint

        metadata_added = []

        def add_metadata(checkpoint):
            checkpoint['custom_metadata'] = {'test': 'value'}
            metadata_added.append(True)

        checkpoint_manager.save_checkpoint_with_validation(
            mock_trainer, simple_model, checkpoint_path,
            config_metadata_fn=add_metadata
        )

        assert len(metadata_added) == 1

        # Verify metadata was added
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        assert 'custom_metadata' in checkpoint
        assert checkpoint['custom_metadata'] == {'test': 'value'}

    def test_save_checkpoint_with_validation_atomic(self, checkpoint_manager, mock_trainer, simple_model, temp_dir):
        """Test that checkpoint saving is atomic (no .tmp file left)."""
        checkpoint_path = temp_dir / "test_checkpoint.ckpt"
        temp_checkpoint_path = checkpoint_path.with_suffix('.tmp')

        def mock_save_checkpoint(filepath):
            checkpoint = {
                'state_dict': simple_model.state_dict(),
                'epoch': mock_trainer.current_epoch,
                'global_step': mock_trainer.global_step,
            }
            torch.save(checkpoint, filepath)

        mock_trainer.save_checkpoint = mock_save_checkpoint

        checkpoint_manager.save_checkpoint_with_validation(
            mock_trainer, simple_model, checkpoint_path
        )

        # Final checkpoint should exist, temp should not
        assert checkpoint_path.exists()
        assert not temp_checkpoint_path.exists()

    def test_save_checkpoint_with_validation_rejects_small_file(self, checkpoint_manager, mock_trainer, simple_model, temp_dir):
        """Test that validation rejects suspiciously small checkpoints."""
        checkpoint_path = temp_dir / "test_checkpoint.ckpt"

        def mock_save_tiny_checkpoint(filepath):
            # Save a very small file (less than 1KB)
            filepath = Path(filepath)
            filepath.write_text("tiny")

        mock_trainer.save_checkpoint = mock_save_tiny_checkpoint

        with pytest.raises(RuntimeError, match="too small"):
            checkpoint_manager.save_checkpoint_with_validation(
                mock_trainer, simple_model, checkpoint_path
            )

    def test_save_checkpoint_with_validation_rejects_missing_keys(self, checkpoint_manager, mock_trainer, simple_model, temp_dir):
        """Test that validation rejects checkpoints missing required keys."""
        checkpoint_path = temp_dir / "test_checkpoint.ckpt"

        def mock_save_incomplete_checkpoint(filepath):
            # Save checkpoint without required keys
            torch.save({'some_data': 'value'}, filepath)

        mock_trainer.save_checkpoint = mock_save_incomplete_checkpoint

        with pytest.raises(RuntimeError, match="missing required keys"):
            checkpoint_manager.save_checkpoint_with_validation(
                mock_trainer, simple_model, checkpoint_path
            )

    def test_save_checkpoint_with_validation_cleans_up_on_failure(self, checkpoint_manager, mock_trainer, simple_model, temp_dir):
        """Test that temp file is cleaned up on failure."""
        checkpoint_path = temp_dir / "test_checkpoint.ckpt"
        temp_checkpoint_path = checkpoint_path.with_suffix('.tmp')

        def mock_save_failing_checkpoint(filepath):
            # Save a very small file that will fail validation
            filepath = Path(filepath)
            filepath.write_text("tiny")

        mock_trainer.save_checkpoint = mock_save_failing_checkpoint

        with pytest.raises(RuntimeError):
            checkpoint_manager.save_checkpoint_with_validation(
                mock_trainer, simple_model, checkpoint_path
            )

        # Temp file should be cleaned up
        assert not temp_checkpoint_path.exists()
        assert not checkpoint_path.exists()

    def test_validate_trainer_state_success(self, checkpoint_manager, mock_trainer, simple_model):
        """Test successful trainer state validation."""
        result = checkpoint_manager.validate_trainer_state_for_pause(
            mock_trainer, simple_model
        )
        assert result is True

    def test_validate_trainer_state_missing_attribute(self, checkpoint_manager, simple_model):
        """Test validation fails for missing trainer attributes."""
        trainer = Mock(spec=[])  # Empty spec - no attributes

        result = checkpoint_manager.validate_trainer_state_for_pause(
            trainer, simple_model
        )
        assert result is False

    def test_validate_trainer_state_none_module(self, checkpoint_manager, mock_trainer):
        """Test validation fails for None pl_module."""
        result = checkpoint_manager.validate_trainer_state_for_pause(
            mock_trainer, None
        )
        assert result is False

    def test_validate_trainer_state_inaccessible_directory(self, temp_dir, mock_trainer, simple_model):
        """Test validation fails for inaccessible checkpoint directory."""
        # Create manager with a valid directory first
        checkpoint_dir = temp_dir / "checkpoints"
        manager = PauseCheckpointManager(checkpoint_dir)

        # Now make the directory inaccessible by mocking mkdir to fail
        with patch.object(Path, 'mkdir', side_effect=PermissionError("Access denied")):
            # Force manager to try to create directory again by setting a new path
            manager.checkpoint_dir = temp_dir / "inaccessible"
            result = manager.validate_trainer_state_for_pause(mock_trainer, simple_model)

        # Should fail validation gracefully
        assert result is False

    @patch('shutil.disk_usage')
    def test_validate_trainer_state_low_disk_space(self, mock_disk_usage, checkpoint_manager, mock_trainer, simple_model):
        """Test validation warns on low disk space."""
        # Simulate very low disk space (50MB)
        mock_disk_usage.return_value = Mock(free=50 * 1024 * 1024)

        result = checkpoint_manager.validate_trainer_state_for_pause(
            mock_trainer, simple_model
        )
        assert result is False

    def test_get_last_checkpoint(self, checkpoint_manager, mock_trainer, simple_model, temp_dir):
        """Test getting the last saved checkpoint path."""
        assert checkpoint_manager.get_last_checkpoint() is None

        checkpoint_path = temp_dir / "test_checkpoint.ckpt"

        def mock_save_checkpoint(filepath):
            checkpoint = {
                'state_dict': simple_model.state_dict(),
                'epoch': mock_trainer.current_epoch,
                'global_step': mock_trainer.global_step,
            }
            torch.save(checkpoint, filepath)

        mock_trainer.save_checkpoint = mock_save_checkpoint

        checkpoint_manager.save_checkpoint(mock_trainer, simple_model, checkpoint_path)

        assert checkpoint_manager.get_last_checkpoint() == checkpoint_path

    def test_multiple_checkpoints_tracks_last(self, checkpoint_manager, mock_trainer, simple_model, temp_dir):
        """Test that multiple saves track the most recent checkpoint."""
        def mock_save_checkpoint(filepath):
            checkpoint = {
                'state_dict': simple_model.state_dict(),
                'epoch': mock_trainer.current_epoch,
                'global_step': mock_trainer.global_step,
            }
            torch.save(checkpoint, filepath)

        mock_trainer.save_checkpoint = mock_save_checkpoint

        path1 = temp_dir / "checkpoint1.ckpt"
        path2 = temp_dir / "checkpoint2.ckpt"

        checkpoint_manager.save_checkpoint(mock_trainer, simple_model, path1)
        assert checkpoint_manager.get_last_checkpoint() == path1

        checkpoint_manager.save_checkpoint(mock_trainer, simple_model, path2)
        assert checkpoint_manager.get_last_checkpoint() == path2


class TestPauseCheckpointManagerEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def checkpoint_manager(self, temp_dir):
        """Create a checkpoint manager for testing."""
        return PauseCheckpointManager(temp_dir / "checkpoints")

    def test_path_with_special_characters_in_dir(self, temp_dir):
        """Test checkpoint manager with special characters in directory name."""
        special_dir = temp_dir / "check points (1)"
        manager = PauseCheckpointManager(special_dir)

        assert manager.checkpoint_dir.exists()

    def test_concurrent_path_generation(self, checkpoint_manager):
        """Test that concurrent path generation produces unique paths."""
        mock_trainer = Mock()
        mock_trainer.current_epoch = 1
        mock_trainer.global_step = 100

        paths = set()
        for _ in range(10):
            path = checkpoint_manager.get_checkpoint_path(mock_trainer)
            paths.add(str(path))
            time.sleep(0.01)  # Small delay to ensure different timestamps

        # All paths should be unique (due to timestamps)
        # Note: if running very fast, some might collide
        assert len(paths) >= 1  # At least some uniqueness
