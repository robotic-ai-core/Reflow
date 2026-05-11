"""Real subprocess resume coverage.

The unit tests in tests/unit/cli/ and tests/unit/core/ mock subprocess.run
and inspect the constructed command list. These tests spawn an actual
subprocess so we exercise the full ``python -m lightning_reflow.cli``
entrypoint end-to-end.
"""

import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

import lightning.pytorch as pl

from lightning_reflow.core import LightningReflow
from lightning_reflow.data import SimpleDataModule
from lightning_reflow.models import SimpleReflowModel


def _run_cli(args: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
    """Invoke `python -m lightning_reflow.cli` and capture output."""
    return subprocess.run(
        [sys.executable, "-m", "lightning_reflow.cli", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _make_resumable_config(tmp_path: Path) -> Path:
    """Write a minimal config that the generic CLI can use to fit + resume."""
    config = {
        "model": {
            "class_path": "lightning_reflow.models.SimpleReflowModel",
            "init_args": {
                "input_dim": 10,
                "hidden_dim": 16,
                "output_dim": 2,
                "learning_rate": 0.01,
                "loss_type": "cross_entropy",
            },
        },
        "data": {
            "class_path": "lightning_reflow.data.SimpleDataModule",
            "init_args": {
                "batch_size": 4,
                "train_samples": 16,
                "val_samples": 4,
                "input_dim": 10,
                "output_dim": 2,
                "task_type": "classification",
            },
        },
        "trainer": {
            "max_epochs": 2,
            "max_steps": 4,
            "enable_progress_bar": False,
            "enable_checkpointing": True,
            "logger": False,
            "default_root_dir": str(tmp_path / "run"),
            "accelerator": "cpu",
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config))
    return config_path


class TestCliSubprocessSmoke:

    def test_help_text_lists_resume_subcommand(self):
        result = _run_cli(["resume", "--help"])
        assert result.returncode == 0
        assert "--checkpoint-path" in result.stdout
        assert "--checkpoint-artifact" in result.stdout

    def test_resume_with_missing_checkpoint_exits_nonzero(self, tmp_path):
        result = _run_cli([
            "resume",
            "--checkpoint-path", str(tmp_path / "does_not_exist.ckpt"),
        ])
        assert result.returncode != 0


class TestCliSubprocessRealResume:
    """Fit in-process to produce a checkpoint, then resume in a real subprocess."""

    @pytest.mark.slow
    def test_resume_subprocess_continues_training(self, tmp_path):
        config_path = _make_resumable_config(tmp_path)

        # Produce a checkpoint in-process (faster than spawning two subprocesses).
        model = SimpleReflowModel(
            input_dim=10, hidden_dim=16, output_dim=2, learning_rate=0.01,
            loss_type="cross_entropy",
        )
        datamodule = SimpleDataModule(
            batch_size=4, train_samples=16, val_samples=4,
            input_dim=10, output_dim=2, task_type="classification",
        )
        ckpt_dir = tmp_path / "ckpts"
        ckpt_dir.mkdir()
        ckpt_path = ckpt_dir / "first.ckpt"

        trainer = pl.Trainer(
            max_epochs=1,
            max_steps=2,
            enable_progress_bar=False,
            enable_checkpointing=False,
            logger=False,
            accelerator="cpu",
            default_root_dir=str(tmp_path / "first_run"),
        )
        trainer.fit(model, datamodule)
        trainer.save_checkpoint(str(ckpt_path))
        first_step = trainer.global_step
        assert ckpt_path.exists()

        # Resume in a real subprocess via the generic CLI entrypoint.
        # LightningReflow.resume_cli internally spawns python -m lightning_reflow.cli fit ...
        # We invoke resume_cli directly (no test-side mocks) so that subprocess
        # actually runs end-to-end.
        reflow = LightningReflow()
        # resume_cli calls sys.exit at the end; catch it so the test can assert.
        with pytest.raises(SystemExit) as exc_info:
            reflow.resume_cli(
                resume_source=str(ckpt_path),
                config_overrides=[str(config_path)],
            )
        assert exc_info.value.code == 0, (
            f"Resume subprocess failed with code {exc_info.value.code}"
        )

        # Verify the resume actually advanced training by checking that
        # Lightning produced a new checkpoint on disk under the configured root.
        run_dir = tmp_path / "run"
        new_ckpts = list(run_dir.rglob("*.ckpt"))
        assert new_ckpts, (
            f"Resume subprocess did not produce a continuation checkpoint under {run_dir}"
        )
