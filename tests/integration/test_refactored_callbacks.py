"""
Integration test for refactored callbacks.

This test validates that the refactored WandbArtifactCheckpoint and
FlowProgressBarCallback work correctly in a real training scenario.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

from lightning_reflow.callbacks.wandb.wandb_artifact_checkpoint import WandbArtifactCheckpoint
from lightning_reflow.callbacks.monitoring.flow_progress_bar_callback import FlowProgressBarCallback
from lightning_reflow.callbacks.pause.pause_callback import PauseCallback
from lightning_reflow.models.simple_model import SimpleReflowModel
from lightning_reflow.data.simple_data import SimpleDataModule


class TestRefactoredCallbacksIntegration:
    """Test that refactored callbacks work in real training scenarios."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def simple_setup(self, temp_dir):
        """Create a simple model and datamodule for testing."""
        model = SimpleReflowModel(
            input_dim=10,
            hidden_dim=16,
            output_dim=10,  # Match data output_dim
            learning_rate=0.001,
            loss_type='mse'  # Use MSE for regression
        )

        datamodule = SimpleDataModule(
            train_samples=50,
            val_samples=10,
            test_samples=10,
            input_dim=10,
            output_dim=10,  # Match model output_dim
            batch_size=10,
            num_workers=0,
            task_type='regression'  # Use regression to match MSE loss
        )

        return model, datamodule

    def test_refactored_callbacks_initialization(self, temp_dir):
        """Test that all refactored callbacks can be initialized."""
        # Initialize callbacks
        wandb_callback = WandbArtifactCheckpoint(
            upload_best_model=True,
            upload_last_model=True,
            use_compression=False
        )

        progress_callback = FlowProgressBarCallback(
            refresh_rate=1,
            global_bar_metrics=['loss', 'val_loss']
        )

        pause_callback = PauseCallback(
            checkpoint_dir=temp_dir,
            enable_pause=True,
            save_rng_states=True
        )

        # Verify they have the correct manager states
        assert hasattr(wandb_callback, '_state_manager')
        assert hasattr(pause_callback, '_reproducibility_manager')
        # FlowProgressBarCallback registers its manager in _register_for_state_persistence

    @patch('wandb.init')
    @patch('wandb.run')
    def test_training_with_refactored_callbacks(self, mock_run, mock_init, temp_dir, simple_setup):
        """Test that training works with all refactored callbacks."""
        model, datamodule = simple_setup

        # Setup mock wandb
        mock_run_instance = MagicMock()
        mock_run_instance.id = 'test_run_123'
        mock_run_instance.name = 'test_run'
        mock_run.return_value = mock_run_instance
        mock_init.return_value = mock_run_instance

        # Create callbacks
        callbacks = [
            WandbArtifactCheckpoint(
                upload_best_model=False,
                upload_last_model=False  # Disable uploads for testing
            ),
            FlowProgressBarCallback(
                refresh_rate=10
            ),
            PauseCallback(
                checkpoint_dir=temp_dir,
                save_rng_states=True  # Test the new RNG state saving
            )
        ]

        # Create trainer
        trainer = Trainer(
            max_epochs=2,
            logger=TensorBoardLogger(save_dir=temp_dir),
            callbacks=callbacks,
            default_root_dir=temp_dir,
            enable_progress_bar=False,  # Disabled by FlowProgressBarCallback
            enable_checkpointing=True,
            accelerator='cpu'
        )

        # Mock CLI context to indicate config saving is disabled (no CLI used in this test)
        mock_cli = MagicMock()
        mock_cli.save_config_kwargs = False  # Config saving disabled
        trainer.cli = mock_cli

        # Train
        trainer.fit(model, datamodule)

        # Verify training completed
        assert trainer.current_epoch == 2  # Actually runs 2 full epochs, ending at epoch 2
        assert trainer.global_step > 0

    def test_checkpoint_state_persistence(self, temp_dir, simple_setup):
        """Test that manager states are properly saved and restored."""
        model, datamodule = simple_setup

        # Create checkpoint path
        checkpoint_path = Path(temp_dir) / "test_checkpoint.ckpt"

        # Create trainer with callbacks
        trainer = Trainer(
            max_epochs=1,
            callbacks=[
                FlowProgressBarCallback(refresh_rate=1),
                PauseCallback(checkpoint_dir=temp_dir, save_rng_states=True)
            ],
            default_root_dir=temp_dir,
            enable_progress_bar=False,
            accelerator='cpu'
        )

        # Mock CLI context to indicate config saving is disabled
        mock_cli = MagicMock()
        mock_cli.save_config_kwargs = False
        trainer.cli = mock_cli

        # Train for one epoch
        trainer.fit(model, datamodule)

        # Save checkpoint
        trainer.save_checkpoint(checkpoint_path)

        # Verify checkpoint contains callback states
        import torch
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location='cpu')

        # Check that callback states are saved (Lightning's native checkpoint mechanism)
        assert 'callbacks' in checkpoint
        callbacks_state = checkpoint['callbacks']

        # PauseCallback state should be saved (includes validation_count from parent FlowProgressBar)
        assert 'PauseCallback' in callbacks_state
        assert 'validation_count' in callbacks_state['PauseCallback']

        # FlowProgressBarCallback state should be saved
        assert 'FlowProgressBarCallback' in callbacks_state
        assert 'validation_count' in callbacks_state['FlowProgressBarCallback']

    @patch('wandb.init')
    @patch('wandb.Api')
    def test_wandb_artifact_state_manager(self, mock_api, mock_init, temp_dir, simple_setup):
        """Test that WandbArtifactCheckpoint uses the extracted state manager."""
        model, datamodule = simple_setup

        # Setup mock wandb
        mock_run = MagicMock()
        mock_run.id = 'test_run_456'
        mock_init.return_value = mock_run

        # Create callback
        wandb_callback = WandbArtifactCheckpoint(
            upload_best_model=False,
            upload_last_model=False
        )

        # Verify state manager is created
        assert hasattr(wandb_callback, '_state_manager')

        # Test state capture
        from lightning_reflow.utils.checkpoint.wandb_artifact_state import WandbArtifactState
        assert isinstance(wandb_callback._state_manager, WandbArtifactState)

        # Capture state
        state = wandb_callback._state_manager.capture_state()
        assert 'version' in state
        assert 'config' in state
        assert 'state' in state

        # Test state restoration
        wandb_callback.state.epoch_count = 10
        wandb_callback.state.validation_count = 20

        captured = wandb_callback._state_manager.capture_state()
        assert captured['state']['epoch_count'] == 10
        assert captured['state']['validation_count'] == 20

        # Create new callback and restore state
        new_callback = WandbArtifactCheckpoint()
        new_callback._state_manager.restore_state(captured)

        assert new_callback.state.epoch_count == 10
        assert new_callback.state.validation_count == 20

    def test_all_callbacks_compatible(self, temp_dir, simple_setup):
        """Test that all refactored callbacks work together without conflicts."""
        model, datamodule = simple_setup

        # Create all callbacks
        callbacks = [
            WandbArtifactCheckpoint(upload_best_model=False, upload_last_model=False),
            FlowProgressBarCallback(refresh_rate=10),
            PauseCallback(checkpoint_dir=temp_dir, save_rng_states=True)
        ]

        # Verify no conflicts in manager names
        manager_names = set()
        for callback in callbacks:
            if hasattr(callback, '_state_manager'):
                name = callback._state_manager.manager_name
                assert name not in manager_names, f"Duplicate manager name: {name}"
                manager_names.add(name)
            if hasattr(callback, '_reproducibility_manager'):
                name = callback._reproducibility_manager.manager_name
                assert name not in manager_names, f"Duplicate manager name: {name}"
                manager_names.add(name)

        # Create trainer with all callbacks
        trainer = Trainer(
            max_epochs=1,
            callbacks=callbacks,
            default_root_dir=temp_dir,
            enable_progress_bar=False,
            enable_checkpointing=True,
            accelerator='cpu'
        )

        # Mock CLI context to indicate config saving is disabled
        mock_cli = MagicMock()
        mock_cli.save_config_kwargs = False
        trainer.cli = mock_cli

        # Train
        trainer.fit(model, datamodule)

        # Save checkpoint
        checkpoint_path = Path(temp_dir) / "combined_test.ckpt"
        trainer.save_checkpoint(checkpoint_path)

        # Load and verify all states are saved
        import torch
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location='cpu')

        if 'manager_states' in checkpoint:
            states = checkpoint['manager_states']
            # All expected states should be present
            assert 'scientific_reproducibility' in states
            assert 'flow_progress_bar' in states
            # Note: wandb_artifact_checkpoint only saves if it's been registered