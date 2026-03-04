"""Tests for time-based periodic checkpoint upload to W&B.

Verifies the upload_every_n_hours feature that uploads checkpoints based on
wall-clock time elapsed, ensuring long-running jobs always have a resumable
checkpoint even if killed before teardown.
"""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from lightning_reflow.callbacks.wandb.wandb_artifact_checkpoint import (
    UploadReason,
    UploadState,
    WandbArtifactCheckpoint,
    WandbCheckpointConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_callback(**kwargs) -> WandbArtifactCheckpoint:
    """Create a WandbArtifactCheckpoint with given config overrides."""
    with patch(
        "lightning_reflow.callbacks.wandb.wandb_artifact_checkpoint"
        ".WandbArtifactCheckpoint._register_for_state_persistence"
    ):
        cb = WandbArtifactCheckpoint(**kwargs)
    return cb


def make_trainer(pausing: bool = False) -> MagicMock:
    """Create a mock trainer with optional pause context."""
    trainer = MagicMock()
    trainer.is_global_zero = True
    trainer.current_epoch = 10
    trainer.global_step = 5000

    # PauseCallback mock
    pause_cb = MagicMock()
    pause_cb.__class__.__name__ = "PauseCallback"
    pause_cb.is_pausing.return_value = pausing
    pause_cb.is_pause_scheduled.return_value = False
    pause_cb.is_pausing = MagicMock(return_value=pausing)
    pause_cb.is_pause_scheduled = MagicMock(return_value=False)

    trainer.callbacks = [pause_cb]
    return trainer


def setup_refs(cb: WandbArtifactCheckpoint, last_model_path: str = "/tmp/last.ckpt"):
    """Set up wandb run and model checkpoint references."""
    cb._wandb_run_ref = MagicMock()
    cb._wandb_run_ref.id = "test-run-123"
    cb._wandb_run_ref.entity = "test-entity"
    cb._wandb_run_ref.project = "test-project"

    mc = MagicMock()
    mc.last_model_path = last_model_path
    mc.best_model_path = ""
    mc.best_model_score = None
    cb._model_checkpoint_ref = mc


# ---------------------------------------------------------------------------
# Tests: _should_upload_periodic_hours
# ---------------------------------------------------------------------------


class TestShouldUploadPeriodicHours:
    def test_disabled_by_default(self):
        """upload_every_n_hours=None returns False."""
        cb = make_callback()
        assert cb.config.upload_every_n_hours is None
        assert cb._should_upload_periodic_hours() is False

    def test_triggers_after_hours_elapsed(self):
        """Returns True when elapsed time exceeds threshold."""
        cb = make_callback(upload_every_n_hours=3.0)
        cb.state.training_start_time = time.time() - (4 * 3600)  # 4h ago
        assert cb._should_upload_periodic_hours() is True

    def test_not_triggered_before_threshold(self):
        """Returns False when elapsed time is below threshold."""
        cb = make_callback(upload_every_n_hours=3.0)
        cb.state.training_start_time = time.time() - (1 * 3600)  # 1h ago
        assert cb._should_upload_periodic_hours() is False

    def test_uses_last_upload_time_as_reference(self):
        """After an upload, uses last_periodic_upload_time instead of start."""
        cb = make_callback(upload_every_n_hours=1.0)
        cb.state.training_start_time = time.time() - (5 * 3600)  # 5h ago
        cb.state.last_periodic_upload_time = time.time() - (30 * 60)  # 30min ago
        assert cb._should_upload_periodic_hours() is False

    def test_no_start_time_returns_false(self):
        """Returns False if training hasn't started yet."""
        cb = make_callback(upload_every_n_hours=1.0)
        cb.state.training_start_time = None
        assert cb._should_upload_periodic_hours() is False


# ---------------------------------------------------------------------------
# Tests: _upload_time_based_checkpoint
# ---------------------------------------------------------------------------


class TestUploadTimeBasedCheckpoint:
    def test_uploads_with_latest_ckpt_type(self, tmp_path):
        """Verify ckpt_type='latest' is passed to _upload_checkpoint."""
        ckpt_path = str(tmp_path / "last.ckpt")
        Path(ckpt_path).write_bytes(b"fake checkpoint")

        cb = make_callback(upload_every_n_hours=1.0)
        setup_refs(cb, last_model_path=ckpt_path)

        trainer = make_trainer()
        pl_module = MagicMock()

        with patch.object(cb, "_upload_checkpoint", return_value={"artifact": "test-art"}) as mock_upload, \
             patch.object(cb, "_get_current_score", return_value=0.5):
            cb._upload_time_based_checkpoint(trainer, pl_module)

        mock_upload.assert_called_once()
        _, kwargs = mock_upload.call_args
        # positional args: trainer, pl_module, path, ckpt_type, score, reason
        args = mock_upload.call_args[0]
        assert args[3] == "latest"
        assert args[5] == UploadReason.PERIODIC_HOURS

    def test_updates_last_periodic_upload_time(self, tmp_path):
        """State.last_periodic_upload_time is updated after successful upload."""
        ckpt_path = str(tmp_path / "last.ckpt")
        Path(ckpt_path).write_bytes(b"fake checkpoint")

        cb = make_callback(upload_every_n_hours=1.0)
        setup_refs(cb, last_model_path=ckpt_path)
        assert cb.state.last_periodic_upload_time is None

        trainer = make_trainer()
        pl_module = MagicMock()
        before = time.time()

        with patch.object(cb, "_upload_checkpoint", return_value={"artifact": "ok"}), \
             patch.object(cb, "_get_current_score", return_value=None):
            cb._upload_time_based_checkpoint(trainer, pl_module)

        assert cb.state.last_periodic_upload_time is not None
        assert cb.state.last_periodic_upload_time >= before

    def test_no_checkpoint_available_skips(self):
        """Empty last_model_path does not crash."""
        cb = make_callback(upload_every_n_hours=1.0)
        setup_refs(cb, last_model_path="")

        trainer = make_trainer()
        pl_module = MagicMock()

        with patch.object(cb, "_upload_checkpoint") as mock_upload:
            cb._upload_time_based_checkpoint(trainer, pl_module)

        mock_upload.assert_not_called()

    def test_missing_file_skips(self, tmp_path):
        """Non-existent checkpoint path does not crash."""
        cb = make_callback(upload_every_n_hours=1.0)
        setup_refs(cb, last_model_path=str(tmp_path / "nonexistent.ckpt"))

        trainer = make_trainer()
        pl_module = MagicMock()

        with patch.object(cb, "_upload_checkpoint") as mock_upload:
            cb._upload_time_based_checkpoint(trainer, pl_module)

        mock_upload.assert_not_called()

    def test_no_wandb_run_skips(self, tmp_path):
        """No W&B run reference does not crash."""
        ckpt_path = str(tmp_path / "last.ckpt")
        Path(ckpt_path).write_bytes(b"fake")

        cb = make_callback(upload_every_n_hours=1.0)
        setup_refs(cb, last_model_path=ckpt_path)
        cb._wandb_run_ref = None

        trainer = make_trainer()
        pl_module = MagicMock()

        with patch.object(cb, "_upload_checkpoint") as mock_upload:
            cb._upload_time_based_checkpoint(trainer, pl_module)

        mock_upload.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: on_train_batch_end integration
# ---------------------------------------------------------------------------


class TestOnTrainBatchEnd:
    def test_skipped_during_pause(self, tmp_path):
        """Pause context prevents time-based upload."""
        ckpt_path = str(tmp_path / "last.ckpt")
        Path(ckpt_path).write_bytes(b"fake")

        cb = make_callback(upload_every_n_hours=1.0)
        setup_refs(cb, last_model_path=ckpt_path)
        cb.state.training_start_time = time.time() - (2 * 3600)

        trainer = make_trainer(pausing=True)
        pl_module = MagicMock()

        with patch.object(cb, "_upload_time_based_checkpoint") as mock_upload:
            # Call unwrapped (rank_zero_only wraps it)
            cb.on_train_batch_end.__wrapped__(cb, trainer, pl_module, None, None, 0)

        mock_upload.assert_not_called()

    def test_end_of_training_upload_not_blocked(self, tmp_path):
        """Periodic upload does NOT set has_uploaded, so on_fit_end still fires."""
        ckpt_path = str(tmp_path / "last.ckpt")
        Path(ckpt_path).write_bytes(b"fake checkpoint")

        cb = make_callback(upload_every_n_hours=1.0, upload_last_model=True)
        setup_refs(cb, last_model_path=ckpt_path)
        cb.state.training_start_time = time.time() - (2 * 3600)

        trainer = make_trainer()
        pl_module = MagicMock()

        # Simulate a periodic upload
        with patch.object(cb, "_upload_checkpoint", return_value={"artifact": "ok"}), \
             patch.object(cb, "_get_current_score", return_value=None):
            cb._upload_time_based_checkpoint(trainer, pl_module)

        # has_uploaded should still be False (periodic doesn't set it)
        assert cb.state.has_uploaded is False


# ---------------------------------------------------------------------------
# Tests: State persistence round-trip
# ---------------------------------------------------------------------------


class TestStatePersistence:
    def test_state_persistence_round_trip(self):
        """last_periodic_upload_time survives save/restore via WandbArtifactState."""
        from lightning_reflow.utils.checkpoint.wandb_artifact_state import WandbArtifactState

        cb = make_callback(upload_every_n_hours=2.0)
        cb.state.last_periodic_upload_time = 1709500000.0
        cb.state.training_start_time = 1709490000.0

        state_mgr = WandbArtifactState(cb)
        captured = state_mgr.capture_state()

        # Create a fresh callback and restore
        cb2 = make_callback(upload_every_n_hours=2.0)
        state_mgr2 = WandbArtifactState(cb2)
        assert state_mgr2.restore_state(captured)

        assert cb2.state.last_periodic_upload_time == 1709500000.0
        assert cb2.state.training_start_time == 1709490000.0


# ---------------------------------------------------------------------------
# Tests: Alias creation
# ---------------------------------------------------------------------------


class TestAliasCreation:
    def test_periodic_hours_gets_periodic_alias(self):
        """PERIODIC_HOURS reason produces 'periodic' alias."""
        from lightning_reflow.callbacks.wandb.checkpoint_upload_helper import (
            CheckpointUploadHelper,
        )

        helper = CheckpointUploadHelper(WandbCheckpointConfig())
        aliases = helper.create_aliases("latest", UploadReason.PERIODIC_HOURS)
        assert "periodic" in aliases
        assert "latest" in aliases
