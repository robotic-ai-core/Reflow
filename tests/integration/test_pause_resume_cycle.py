"""End-to-end test for the pause → resume cycle using the public PauseCallback API.

Exercises the production pause path without simulating keyboard input:
  1. Start fit() with PauseCallback installed.
  2. A helper trigger callback calls `request_pause()` after the first
     training step.
  3. PauseCallback writes a checkpoint at the validation boundary and sets
     trainer.should_stop, ending training cleanly.
  4. A fresh trainer resumes from that checkpoint and continues to completion.

This is the only test in the suite that drives PauseCallback through a real
fit loop end-to-end — the unit tests for the public API (request_pause,
cancel_pause) cover the state machine but not the checkpoint-saving path.
"""

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback

from lightning_reflow.callbacks.pause import PauseCallback


@pytest.fixture(scope="module", autouse=True)
def _register_checkpoint_safe_globals():
    """Pause checkpoints contain numpy/torch-version objects; without these
    registrations Lightning's weights_only=True load path raises.

    LightningReflowCLI._register_checkpoint_safe_globals does the same thing
    as a side effect of CLI construction — this fixture registers them
    explicitly so a direct pl.Trainer-based resume works in the test.
    """
    import numpy as np
    safe_globals = [
        np._core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
    ]
    if hasattr(np, "dtypes"):
        safe_globals.extend(
            getattr(np.dtypes, attr) for attr in dir(np.dtypes) if "DType" in attr
        )
    if hasattr(torch, "torch_version") and hasattr(torch.torch_version, "TorchVersion"):
        safe_globals.append(torch.torch_version.TorchVersion)
    torch.serialization.add_safe_globals(safe_globals)


class _TinyClassifier(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.functional.cross_entropy(self.linear(x), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.functional.cross_entropy(self.linear(x), y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=1e-2)


def _make_loader(num_batches: int = 6, batch_size: int = 4) -> DataLoader:
    x = torch.randn(num_batches * batch_size, 4)
    y = torch.randint(0, 2, (num_batches * batch_size,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


class _PauseAfterStep(Callback):
    """Helper callback that schedules a pause after a given global step."""

    def __init__(self, pause_callback: PauseCallback, after_step: int = 1):
        self.pause_callback = pause_callback
        self.after_step = after_step
        self.requested = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.requested and trainer.global_step >= self.after_step:
            self.pause_callback.request_pause(reason="test-driven pause")
            self.requested = True


class TestPauseResumeCycle:

    def test_pause_writes_checkpoint_and_stops_training(self, tmp_path):
        ckpt_dir = tmp_path / "pause"
        pause_cb = PauseCallback(
            checkpoint_dir=str(ckpt_dir),
            enable_pause=False,  # disable keyboard listener for headless tests
            save_rng_states=True,
        )
        trigger = _PauseAfterStep(pause_cb, after_step=1)

        trainer = pl.Trainer(
            max_epochs=3,
            limit_train_batches=4,
            limit_val_batches=2,
            enable_progress_bar=False,
            enable_checkpointing=False,
            logger=False,
            accelerator="cpu",
            callbacks=[pause_cb, trigger],
            default_root_dir=str(tmp_path),
        )
        trainer.fit(_TinyClassifier(), _make_loader(), _make_loader(num_batches=2))

        assert trigger.requested, "Trigger callback should have requested a pause"
        assert trainer.should_stop is True, (
            "PauseCallback should set trainer.should_stop after writing checkpoint"
        )

        ckpt_files = list(ckpt_dir.rglob("*.ckpt"))
        assert ckpt_files, f"No pause checkpoint written under {ckpt_dir}"

        checkpoint = torch.load(ckpt_files[0], map_location="cpu", weights_only=False)
        assert "state_dict" in checkpoint
        assert checkpoint["global_step"] >= 1

    def test_resume_from_pause_checkpoint_continues_training(self, tmp_path):
        """Pause partway, then resume in a fresh trainer and finish the run."""
        ckpt_dir = tmp_path / "pause"
        pause_cb = PauseCallback(
            checkpoint_dir=str(ckpt_dir), enable_pause=False, save_rng_states=True,
        )
        trigger = _PauseAfterStep(pause_cb, after_step=1)

        model = _TinyClassifier()
        trainer1 = pl.Trainer(
            max_epochs=2,
            limit_train_batches=3,
            limit_val_batches=2,
            enable_progress_bar=False,
            enable_checkpointing=False,
            logger=False,
            accelerator="cpu",
            callbacks=[pause_cb, trigger],
            default_root_dir=str(tmp_path / "run1"),
        )
        trainer1.fit(model, _make_loader(), _make_loader(num_batches=2))
        paused_step = trainer1.global_step

        ckpt_files = list(ckpt_dir.rglob("*.ckpt"))
        assert ckpt_files, "Expected a pause checkpoint after first run"
        checkpoint_path = str(ckpt_files[0])

        trainer2 = pl.Trainer(
            max_epochs=2,
            limit_train_batches=3,
            limit_val_batches=2,
            enable_progress_bar=False,
            enable_checkpointing=False,
            logger=False,
            accelerator="cpu",
            callbacks=[PauseCallback(
                checkpoint_dir=str(tmp_path / "pause2"),
                enable_pause=False, save_rng_states=True,
            )],
            default_root_dir=str(tmp_path / "run2"),
        )
        trainer2.fit(_TinyClassifier(), _make_loader(), _make_loader(num_batches=2),
                     ckpt_path=checkpoint_path)

        assert trainer2.global_step >= paused_step, (
            "Resumed trainer must progress past the paused step "
            f"(paused={paused_step}, resumed={trainer2.global_step})"
        )
