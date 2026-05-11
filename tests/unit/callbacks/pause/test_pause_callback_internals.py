"""Internal-method tests for PauseCallback.

These cover the helpers that used to live in PauseCheckpointManager,
PauseUploadHandler, and ResumeCommandPrinter before they were folded back
into PauseCallback. End-to-end pause→resume coverage lives in
tests/integration/test_pause_resume_cycle.py.
"""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from lightning.pytorch import Trainer

from lightning_reflow.callbacks.pause.pause_callback import PauseCallback


def _make_callback(tmp_path: Path) -> PauseCallback:
    return PauseCallback(
        checkpoint_dir=str(tmp_path),
        enable_pause=False,
        save_rng_states=False,
    )


# ---------------------------------------------------------------------------
# Checkpoint path + save
# ---------------------------------------------------------------------------


class TestCheckpointPath:

    def test_path_encodes_epoch_step_and_timestamp(self, tmp_path):
        cb = _make_callback(tmp_path)
        trainer = MagicMock(current_epoch=3, global_step=42)
        path = cb._get_checkpoint_path(trainer, upload=False)
        assert path.parent == tmp_path
        assert path.suffix == ".ckpt"
        assert "epoch=3" in path.name
        assert "step=42" in path.name
        assert path.stem.startswith("pause_")

    def test_path_uses_upload_prefix_when_requested(self, tmp_path):
        cb = _make_callback(tmp_path)
        trainer = MagicMock(current_epoch=0, global_step=1)
        path = cb._get_checkpoint_path(trainer, upload=True)
        assert path.stem.startswith("upload_")


class TestSaveCheckpoint:

    def test_save_checkpoint_tracks_last_path(self, tmp_path):
        cb = _make_callback(tmp_path)
        trainer = MagicMock()
        target = tmp_path / "foo.ckpt"
        cb._save_checkpoint(trainer, MagicMock(), target)
        trainer.save_checkpoint.assert_called_once_with(target)
        assert cb.last_checkpoint_path == target


class TestSaveCheckpointWithValidation:

    def _real_checkpoint(self) -> dict:
        # Lightning-shaped checkpoint with the required keys; ~5KB of payload.
        return {
            "state_dict": {"w": torch.randn(50, 10)},
            "epoch": 5,
            "global_step": 100,
            "optimizer_states": [{}],
        }

    def test_atomic_save_renames_temp_to_final(self, tmp_path):
        cb = _make_callback(tmp_path)
        target = tmp_path / "ckpt.ckpt"
        payload = self._real_checkpoint()

        def fake_save(path):
            torch.save(payload, path)

        trainer = MagicMock()
        trainer.save_checkpoint.side_effect = fake_save
        pl_module = MagicMock()

        with patch.object(cb, "add_config_metadata"):
            cb._save_checkpoint_with_validation(trainer, pl_module, target)

        assert target.exists()
        assert not target.with_suffix(".tmp").exists()
        assert cb.last_checkpoint_path == target

    def test_raises_when_checkpoint_missing_required_keys(self, tmp_path):
        cb = _make_callback(tmp_path)
        target = tmp_path / "bad.ckpt"

        def fake_save(path):
            torch.save({"only_state_dict": torch.randn(50, 10)}, path)

        trainer = MagicMock()
        trainer.save_checkpoint.side_effect = fake_save

        with pytest.raises(RuntimeError, match="missing required keys"):
            cb._save_checkpoint_with_validation(trainer, MagicMock(), target)
        # Temp file should be cleaned up
        assert not target.with_suffix(".tmp").exists()

    def test_raises_when_checkpoint_too_small(self, tmp_path):
        cb = _make_callback(tmp_path)
        target = tmp_path / "tiny.ckpt"

        def fake_save(path):
            Path(path).write_bytes(b"x")

        trainer = MagicMock()
        trainer.save_checkpoint.side_effect = fake_save

        with pytest.raises(RuntimeError, match="too small"):
            cb._save_checkpoint_with_validation(trainer, MagicMock(), target)
        assert not target.with_suffix(".tmp").exists()


class TestValidateTrainerStateForPause:

    def test_rejects_missing_pl_module(self, tmp_path):
        cb = _make_callback(tmp_path)
        trainer = MagicMock(global_step=1, current_epoch=0, logger=None)
        assert cb._validate_trainer_state_for_pause(trainer, None) is False

    def test_rejects_missing_trainer_attrs(self, tmp_path):
        cb = _make_callback(tmp_path)

        class BareTrainer:
            pass

        assert cb._validate_trainer_state_for_pause(BareTrainer(), MagicMock()) is False

    def test_accepts_healthy_state(self, tmp_path):
        cb = _make_callback(tmp_path)
        trainer = MagicMock(global_step=1, current_epoch=0, logger=None)
        assert cb._validate_trainer_state_for_pause(trainer, MagicMock()) is True


# ---------------------------------------------------------------------------
# W&B upload
# ---------------------------------------------------------------------------


class TestHandleWandbUpload:

    def test_no_wandb_callback_returns_none(self, tmp_path):
        cb = _make_callback(tmp_path)
        # spec=[] makes the mock NOT auto-provide upload_pause_checkpoint /
        # _upload_pause_checkpoint_artifact, simulating a non-wandb callback.
        non_wandb_cb = MagicMock(spec=[])
        trainer = MagicMock(callbacks=[non_wandb_cb])
        result = cb._handle_wandb_upload(trainer, MagicMock(), "/tmp/foo.ckpt")
        assert result is None

    def test_uses_modern_upload_pause_checkpoint_when_available(self, tmp_path):
        cb = _make_callback(tmp_path)
        wandb_cb = MagicMock()
        wandb_cb.upload_pause_checkpoint = MagicMock(return_value="entity/proj/art:latest")
        trainer = MagicMock(callbacks=[wandb_cb])
        trainer.lightning_module = MagicMock()

        result = cb._handle_wandb_upload(trainer, MagicMock(), "/tmp/foo.ckpt")
        assert result == "entity/proj/art:latest"
        wandb_cb.upload_pause_checkpoint.assert_called_once()

    def test_raises_on_none_trainer(self, tmp_path):
        cb = _make_callback(tmp_path)
        with pytest.raises(ValueError):
            cb._handle_wandb_upload(None, MagicMock(), "/tmp/foo.ckpt")


class TestUploadWithFallback:

    def test_returns_none_when_upload_raises(self, tmp_path):
        cb = _make_callback(tmp_path)
        with patch.object(
            cb, "_handle_wandb_upload", side_effect=RuntimeError("network")
        ):
            result = cb._handle_upload_with_fallback(
                MagicMock(), MagicMock(), "/tmp/foo.ckpt",
            )
        assert result is None

    def test_returns_artifact_path_on_success(self, tmp_path):
        cb = _make_callback(tmp_path)
        with patch.object(cb, "_handle_wandb_upload", return_value="entity/proj/a:v1"):
            result = cb._handle_upload_with_fallback(
                MagicMock(), MagicMock(), "/tmp/foo.ckpt",
            )
        assert result == "entity/proj/a:v1"


# ---------------------------------------------------------------------------
# Resume command output
# ---------------------------------------------------------------------------


class TestResumeCommandOutput:

    def test_prints_local_and_legacy_commands(self, tmp_path, capsys):
        cb = _make_callback(tmp_path)
        cb._original_argv = ["python", "train.py", "fit", "--config", "c.yaml"]
        cb._print_resume_commands(MagicMock(), "/tmp/foo.ckpt")
        out = capsys.readouterr().out
        assert "Local resume:" in out
        assert "/tmp/foo.ckpt" in out
        assert "Legacy method:" in out

    def test_prints_wandb_artifact_when_provided(self, tmp_path, capsys):
        cb = _make_callback(tmp_path)
        cb._original_argv = ["python", "train.py", "fit"]
        cb._print_resume_commands(MagicMock(), "/tmp/foo.ckpt", "entity/proj/a:v1")
        out = capsys.readouterr().out
        assert "W&B resume:" in out
        assert "entity/proj/a:v1" in out
        assert "Legacy W&B:" in out

    def test_raises_when_checkpoint_path_empty(self, tmp_path):
        cb = _make_callback(tmp_path)
        with pytest.raises(ValueError):
            cb._print_resume_commands(MagicMock(), "")

    def test_legacy_command_strips_existing_ckpt_path(self, tmp_path):
        cb = _make_callback(tmp_path)
        cb._original_argv = ["python", "train.py", "fit", "--ckpt_path", "/old.ckpt"]
        assert "--ckpt_path" not in cb._build_legacy_command()

    def test_legacy_command_strips_equals_form(self, tmp_path):
        cb = _make_callback(tmp_path)
        cb._original_argv = ["python", "train.py", "fit", "--ckpt_path=/old.ckpt"]
        assert "--ckpt_path" not in cb._build_legacy_command()

    def test_print_with_fallback_prints_something_when_main_path_raises(
        self, tmp_path, capsys,
    ):
        cb = _make_callback(tmp_path)
        cb._original_argv = []  # forces ValueError in _print_resume_commands
        cb._print_resume_commands_with_fallback(MagicMock(), "/tmp/foo.ckpt")
        out = capsys.readouterr().out
        assert "/tmp/foo.ckpt" in out
