"""Unit tests for TrainingProfilerCallback.

Covers the profiler lifecycle (start on first batch, stop after the
scheduled window, trace export, cleanup on fit_end / on_exception) without
requiring a GPU.
"""

from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import lightning.pytorch as pl

from lightning_reflow.callbacks.profiler import TrainingProfilerCallback


class _TinyModel(pl.LightningModule):
    """Minimal LightningModule used to drive the profiler callback."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.linear(x)
        loss = nn.functional.mse_loss(out, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=1e-2)


def _make_dataloader(num_batches: int = 8, batch_size: int = 4) -> DataLoader:
    x = torch.randn(num_batches * batch_size, 4)
    y = torch.randn(num_batches * batch_size, 2)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


class TestProfilerCallbackPhases:

    def test_step_phase_classifies_wait_warmup_active(self):
        cb = TrainingProfilerCallback(active_steps=3, warmup_steps=2, wait_steps=1)
        assert cb._step_phase(0) == "wait"
        assert cb._step_phase(1) == "warmup"
        assert cb._step_phase(2) == "warmup"
        assert cb._step_phase(3) == "active"
        assert cb._step_phase(5) == "active"

    def test_total_steps_property(self):
        cb = TrainingProfilerCallback(active_steps=10, warmup_steps=3, wait_steps=2)
        assert cb._total_steps == 15


class TestProfilerLifecycle:

    def test_profiler_runs_and_exports_trace_within_window(self, tmp_path):
        cb = TrainingProfilerCallback(
            active_steps=2,
            warmup_steps=1,
            wait_steps=0,
            trace_dir=str(tmp_path),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_flops=False,
        )

        trainer = pl.Trainer(
            max_steps=cb._total_steps,
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False,
            accelerator="cpu",
            callbacks=[cb],
        )
        trainer.fit(_TinyModel(), _make_dataloader())

        trace_path = tmp_path / "profile_trace.json"
        assert trace_path.exists(), "Profiler should export a Chrome trace"
        assert trace_path.stat().st_size > 0
        assert cb._finished is True
        assert len(cb._step_times) == cb._total_steps

    def test_cleanup_on_fit_end_clears_active_profiler(self, tmp_path):
        """If training ends before the schedule completes, cleanup tears down the profiler."""
        cb = TrainingProfilerCallback(
            active_steps=100,
            warmup_steps=0,
            wait_steps=0,
            trace_dir=str(tmp_path),
            record_shapes=False,
            profile_memory=False,
        )
        trainer = pl.Trainer(
            max_steps=2,
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False,
            accelerator="cpu",
            callbacks=[cb],
        )
        trainer.fit(_TinyModel(), _make_dataloader())

        assert cb._prof is None
        assert cb._finished is True

    def test_cleanup_on_exception_tears_down_profiler(self, tmp_path):
        """on_exception ensures the profiler doesn't leak when training fails."""
        cb = TrainingProfilerCallback(
            active_steps=10, warmup_steps=0, wait_steps=0,
            trace_dir=str(tmp_path),
            record_shapes=False, profile_memory=False,
        )

        cb._start_profiler()
        assert cb._prof is not None

        cb.on_exception(Mock(), Mock(), RuntimeError("boom"))

        assert cb._prof is None
        assert cb._finished is True

    def test_no_op_after_finished(self, tmp_path):
        """Subsequent batch hooks after profiler finishes are silent no-ops."""
        cb = TrainingProfilerCallback(
            active_steps=1, warmup_steps=0, wait_steps=0,
            trace_dir=str(tmp_path),
            record_shapes=False, profile_memory=False,
        )

        trainer = pl.Trainer(
            max_steps=3,  # more than profiler schedule
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False,
            accelerator="cpu",
            callbacks=[cb],
        )
        trainer.fit(_TinyModel(), _make_dataloader())

        # Profiler should have stopped at step 1 even though training continued
        assert cb._finished is True
        assert len(cb._step_times) == cb._total_steps  # 1 step recorded
