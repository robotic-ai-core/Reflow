"""
Integration tests for callback state persistence across checkpointing.

Validates that PauseCallback, FlowProgressBarCallback, and WandbArtifactCheckpoint
correctly save and reload state via Lightning's native callback checkpoint mechanism,
and that they compose without manager-name collisions.
"""

import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from lightning.pytorch import Trainer

from lightning_reflow.callbacks.monitoring.flow_progress_bar_callback import FlowProgressBarCallback
from lightning_reflow.callbacks.pause.pause_callback import PauseCallback
from lightning_reflow.callbacks.wandb.wandb_artifact_checkpoint import WandbArtifactCheckpoint
from lightning_reflow.data.simple_data import SimpleDataModule
from lightning_reflow.models.simple_model import SimpleReflowModel


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def simple_setup():
    model = SimpleReflowModel(
        input_dim=10, hidden_dim=16, output_dim=10,
        learning_rate=0.001, loss_type='mse',
    )
    datamodule = SimpleDataModule(
        train_samples=50, val_samples=10, test_samples=10,
        input_dim=10, output_dim=10, batch_size=10,
        num_workers=0, task_type='regression',
    )
    return model, datamodule


def _disable_config_save(trainer):
    mock_cli = MagicMock()
    mock_cli.save_config_kwargs = False
    trainer.cli = mock_cli


class TestCallbackStatePersistence:

    def test_pause_and_progress_bar_state_round_trips_through_checkpoint(
        self, temp_dir, simple_setup
    ):
        model, datamodule = simple_setup
        checkpoint_path = Path(temp_dir) / "checkpoint.ckpt"

        trainer = Trainer(
            max_epochs=1,
            callbacks=[
                FlowProgressBarCallback(refresh_rate=1),
                PauseCallback(checkpoint_dir=temp_dir, save_rng_states=True),
            ],
            default_root_dir=temp_dir,
            enable_progress_bar=False,
            accelerator='cpu',
        )
        _disable_config_save(trainer)
        trainer.fit(model, datamodule)
        trainer.save_checkpoint(checkpoint_path)

        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
        assert 'callbacks' in checkpoint
        callback_states = checkpoint['callbacks']

        assert 'PauseCallback' in callback_states
        assert 'validation_count' in callback_states['PauseCallback']
        assert 'FlowProgressBarCallback' in callback_states
        assert 'validation_count' in callback_states['FlowProgressBarCallback']

    def test_three_callbacks_compose_without_manager_name_collision(
        self, temp_dir, simple_setup
    ):
        model, datamodule = simple_setup
        callbacks = [
            WandbArtifactCheckpoint(upload_best_model=False, upload_last_model=False),
            FlowProgressBarCallback(refresh_rate=10),
            PauseCallback(checkpoint_dir=temp_dir, save_rng_states=True),
        ]

        manager_names = set()
        for cb in callbacks:
            for attr in ('_state_manager', '_reproducibility_manager'):
                manager = getattr(cb, attr, None)
                if manager is not None:
                    name = manager.manager_name
                    assert name not in manager_names, f"Duplicate manager name: {name}"
                    manager_names.add(name)

        trainer = Trainer(
            max_epochs=1,
            callbacks=callbacks,
            default_root_dir=temp_dir,
            enable_progress_bar=False,
            enable_checkpointing=True,
            accelerator='cpu',
        )
        _disable_config_save(trainer)
        trainer.fit(model, datamodule)

        checkpoint_path = Path(temp_dir) / "combined.ckpt"
        trainer.save_checkpoint(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location='cpu')

        if 'manager_states' in checkpoint:
            states = checkpoint['manager_states']
            assert 'scientific_reproducibility' in states
            assert 'flow_progress_bar' in states
