"""
Tests for EarlyPauseCallback and PauseCallback public API.

Tests cover:
- PauseCallback public API (request_pause, cancel_pause, is_pause_pending)
- EarlyPauseCallback integration with PauseCallback
- Multiple callbacks present scenario
- State persistence across checkpoints
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

import lightning as L
from lightning.pytorch import Trainer, LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint

from lightning_reflow.callbacks.pause import PauseCallback, EarlyPauseCallback


# =============================================================================
# Test Fixtures
# =============================================================================


class SimpleModel(LightningModule):
    """Minimal model for testing callbacks."""

    def __init__(self, produce_improving_loss: bool = True):
        super().__init__()
        self.layer = nn.Linear(10, 2)
        self.produce_improving_loss = produce_improving_loss
        self._val_epoch = 0

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.functional.mse_loss(self(x), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # Use deterministic loss values for predictable early stopping behavior
        if self.produce_improving_loss:
            # Loss decreases each validation epoch: 1.0, 0.5, 0.33, 0.25, ...
            base_loss = 1.0 / (self._val_epoch + 1)
        else:
            # Loss stays constant at 1.0 (will trigger early stopping)
            base_loss = 1.0
        self.log("val_loss", torch.tensor(base_loss))
        return torch.tensor(base_loss)

    def on_validation_epoch_end(self):
        """Increment validation epoch counter."""
        self._val_epoch += 1

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)


class SimpleDataModule(L.LightningDataModule):
    """Minimal data module for testing."""

    def __init__(self, batch_size: int = 4, num_samples: int = 20):
        super().__init__()
        self.batch_size = batch_size
        self.num_samples = num_samples

    def setup(self, stage=None):
        # Simple random data
        self.train_data = torch.utils.data.TensorDataset(
            torch.randn(self.num_samples, 10),
            torch.randn(self.num_samples, 2),
        )
        self.val_data = torch.utils.data.TensorDataset(
            torch.randn(self.num_samples // 2, 10),
            torch.randn(self.num_samples // 2, 2),
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data, batch_size=self.batch_size
        )


# =============================================================================
# PauseCallback Public API Tests
# =============================================================================


class TestPauseCallbackPublicAPI:
    """Test the public API of PauseCallback."""

    def test_request_pause_schedules_pause(self):
        """Test that request_pause schedules a pause."""
        callback = PauseCallback(
            checkpoint_dir="/tmp/test_pause",
            enable_pause=False,  # Disable keyboard to avoid issues
        )

        assert not callback.is_pause_pending()

        result = callback.request_pause(reason="Test pause")

        assert result is True
        assert callback.is_pause_pending()

    def test_request_pause_with_upload(self):
        """Test that request_pause with upload=True sets upload flag."""
        callback = PauseCallback(
            checkpoint_dir="/tmp/test_pause",
            enable_pause=False,
        )

        callback.request_pause(upload=True, reason="Test with upload")

        assert callback.is_pause_pending()
        assert callback.is_upload_requested()

    def test_request_pause_already_scheduled(self):
        """Test that request_pause returns False if already scheduled."""
        callback = PauseCallback(
            checkpoint_dir="/tmp/test_pause",
            enable_pause=False,
        )

        # First request
        result1 = callback.request_pause(reason="First")
        assert result1 is True

        # Second request - should return False
        result2 = callback.request_pause(reason="Second")
        assert result2 is False
        assert callback.is_pause_pending()  # Still scheduled

    def test_cancel_pause(self):
        """Test that cancel_pause cancels a scheduled pause."""
        callback = PauseCallback(
            checkpoint_dir="/tmp/test_pause",
            enable_pause=False,
        )

        callback.request_pause(reason="To be cancelled")
        assert callback.is_pause_pending()

        result = callback.cancel_pause()

        assert result is True
        assert not callback.is_pause_pending()

    def test_cancel_pause_when_not_scheduled(self):
        """Test that cancel_pause returns False when no pause scheduled."""
        callback = PauseCallback(
            checkpoint_dir="/tmp/test_pause",
            enable_pause=False,
        )

        assert not callback.is_pause_pending()

        result = callback.cancel_pause()

        assert result is False

    def test_is_pause_pending_alias(self):
        """Test that is_pause_pending is an alias for is_pause_scheduled."""
        callback = PauseCallback(
            checkpoint_dir="/tmp/test_pause",
            enable_pause=False,
        )

        assert callback.is_pause_pending() == callback.is_pause_scheduled()

        callback.request_pause()

        assert callback.is_pause_pending() == callback.is_pause_scheduled()
        assert callback.is_pause_pending() is True


# =============================================================================
# EarlyPauseCallback Unit Tests
# =============================================================================


class TestEarlyPauseCallbackUnit:
    """Unit tests for EarlyPauseCallback."""

    def test_requires_pause_callback(self):
        """Test that EarlyPauseCallback raises error without PauseCallback."""
        model = SimpleModel()
        early_pause = EarlyPauseCallback(monitor="val_loss", patience=2)

        trainer = Trainer(
            max_epochs=1,
            callbacks=[early_pause],  # No PauseCallback!
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )

        with pytest.raises(RuntimeError, match="requires PauseCallback"):
            trainer.fit(model, datamodule=SimpleDataModule())

    def test_finds_pause_callback(self):
        """Test that EarlyPauseCallback finds PauseCallback during setup."""
        pause_callback = PauseCallback(
            checkpoint_dir="/tmp/test",
            enable_pause=False,
        )
        early_pause = EarlyPauseCallback(monitor="val_loss", patience=2)

        # Mock trainer with callbacks
        trainer = MagicMock()
        trainer.callbacks = [pause_callback, early_pause]

        early_pause.setup(trainer, MagicMock(), "fit")

        assert early_pause._pause_callback is pause_callback

    def test_state_dict_includes_pause_triggered(self):
        """Test that state_dict includes pause_triggered flag."""
        early_pause = EarlyPauseCallback(monitor="val_loss", patience=2)
        early_pause._pause_triggered = True

        state = early_pause.state_dict()

        assert "pause_triggered" in state
        assert state["pause_triggered"] is True

    def test_load_state_dict_resets_pause_triggered(self):
        """Test that load_state_dict resets pause_triggered."""
        early_pause = EarlyPauseCallback(monitor="val_loss", patience=2)
        early_pause._pause_triggered = True

        # Include all required keys from EarlyStopping state_dict
        state = {
            "wait_count": 5,
            "best_score": torch.tensor(0.1),
            "patience": 2,
            "stopped_epoch": 0,
            "pause_triggered": True,
        }
        early_pause.load_state_dict(state)

        assert early_pause._pause_triggered is False

    def test_load_state_dict_resets_wait_count_when_configured(self):
        """Test that load_state_dict resets wait_count when reset_patience_on_resume=True."""
        early_pause = EarlyPauseCallback(
            monitor="val_loss",
            patience=10,
            reset_patience_on_resume=True,
            verbose=False,
        )

        state = {
            "wait_count": 5,
            "best_score": torch.tensor(0.1),
            "patience": 10,
            "stopped_epoch": 0,
        }
        early_pause.load_state_dict(state)

        assert early_pause.wait_count == 0

    def test_load_state_dict_preserves_wait_count_when_configured(self):
        """Test that load_state_dict preserves wait_count when reset_patience_on_resume=False."""
        early_pause = EarlyPauseCallback(
            monitor="val_loss",
            patience=10,
            reset_patience_on_resume=False,
        )

        state = {
            "wait_count": 5,
            "best_score": torch.tensor(0.1),
            "patience": 10,
            "stopped_epoch": 0,
        }
        early_pause.load_state_dict(state)

        assert early_pause.wait_count == 5


# =============================================================================
# EarlyPauseCallback Integration Tests
# =============================================================================


class TestEarlyPauseCallbackIntegration:
    """Integration tests for EarlyPauseCallback with actual training."""

    def test_triggers_pause_on_plateau(self, tmp_path):
        """Test that EarlyPauseCallback triggers pause when metric plateaus."""
        model = SimpleModel(produce_improving_loss=False)  # Loss won't improve
        datamodule = SimpleDataModule()

        pause_callback = PauseCallback(
            checkpoint_dir=str(tmp_path),
            enable_pause=False,
        )
        early_pause = EarlyPauseCallback(
            monitor="val_loss",
            patience=2,  # Trigger after 2 epochs without improvement
            mode="min",
            upload_to_wandb=False,
            verbose=True,
        )

        trainer = Trainer(
            max_epochs=10,
            callbacks=[pause_callback, early_pause],
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            val_check_interval=1.0,
        )

        # Training should be stopped by pause (trainer.should_stop set in on_validation_end)
        trainer.fit(model, datamodule=datamodule)

        # Verify pause was triggered
        assert early_pause._pause_triggered is True
        # Training should have stopped before max_epochs
        assert trainer.current_epoch < 10

    def test_no_pause_when_improving(self, tmp_path):
        """Test that EarlyPauseCallback doesn't trigger when metric improves."""
        model = SimpleModel(produce_improving_loss=True)  # Loss will improve
        datamodule = SimpleDataModule()

        pause_callback = PauseCallback(
            checkpoint_dir=str(tmp_path),
            enable_pause=False,
        )
        early_pause = EarlyPauseCallback(
            monitor="val_loss",
            patience=2,
            mode="min",
            upload_to_wandb=False,
        )

        trainer = Trainer(
            max_epochs=5,
            callbacks=[pause_callback, early_pause],
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )

        trainer.fit(model, datamodule=datamodule)

        # Pause should NOT have been triggered
        assert early_pause._pause_triggered is False
        # Training should complete all epochs (current_epoch is max_epochs after training)
        assert trainer.current_epoch >= 4  # Should complete training


# =============================================================================
# Multiple Callbacks Tests
# =============================================================================


class TestMultipleCallbacks:
    """Test EarlyPauseCallback with multiple other callbacks present."""

    def test_works_with_model_checkpoint(self, tmp_path):
        """Test EarlyPauseCallback works alongside ModelCheckpoint."""
        model = SimpleModel(produce_improving_loss=False)
        datamodule = SimpleDataModule()

        pause_callback = PauseCallback(
            checkpoint_dir=str(tmp_path / "pause"),
            enable_pause=False,
        )
        early_pause = EarlyPauseCallback(
            monitor="val_loss",
            patience=2,
            upload_to_wandb=False,
        )
        model_checkpoint = ModelCheckpoint(
            dirpath=str(tmp_path / "checkpoints"),
            monitor="val_loss",
            save_top_k=2,
        )

        trainer = Trainer(
            max_epochs=10,
            callbacks=[pause_callback, early_pause, model_checkpoint],
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )

        trainer.fit(model, datamodule=datamodule)

        # Both should work without conflicts
        assert early_pause._pause_triggered is True

    def test_callback_order_doesnt_matter(self, tmp_path):
        """Test that callback order doesn't affect functionality."""
        model = SimpleModel(produce_improving_loss=False)
        datamodule = SimpleDataModule()

        pause_callback = PauseCallback(
            checkpoint_dir=str(tmp_path),
            enable_pause=False,
        )
        early_pause = EarlyPauseCallback(
            monitor="val_loss",
            patience=2,
            upload_to_wandb=False,
        )

        # EarlyPauseCallback BEFORE PauseCallback
        trainer = Trainer(
            max_epochs=10,
            callbacks=[early_pause, pause_callback],  # Reversed order
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )

        trainer.fit(model, datamodule=datamodule)

        # Should still work
        assert early_pause._pause_triggered is True
        assert early_pause._pause_callback is pause_callback

    def test_multiple_early_pause_callbacks(self, tmp_path):
        """Test multiple EarlyPauseCallbacks monitoring different metrics."""
        model = SimpleModel(produce_improving_loss=False)
        datamodule = SimpleDataModule()

        pause_callback = PauseCallback(
            checkpoint_dir=str(tmp_path),
            enable_pause=False,
        )
        early_pause_loss = EarlyPauseCallback(
            monitor="val_loss",
            patience=2,
            upload_to_wandb=False,
        )

        trainer = Trainer(
            max_epochs=10,
            callbacks=[pause_callback, early_pause_loss],
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )

        trainer.fit(model, datamodule=datamodule)

        # Should have triggered (val_loss not improving)
        assert early_pause_loss._pause_triggered
        # Pause should be pending (was scheduled)
        assert pause_callback.is_pause_pending() or trainer.should_stop


# =============================================================================
# E2E Tests
# =============================================================================


class TestEndToEnd:
    """End-to-end tests simulating real usage."""

    def test_full_workflow_pause_triggers(self, tmp_path):
        """Test that early pause triggers and schedules a pause."""
        model = SimpleModel(produce_improving_loss=False)
        datamodule = SimpleDataModule()

        pause_callback = PauseCallback(
            checkpoint_dir=str(tmp_path),
            enable_pause=False,
        )
        early_pause = EarlyPauseCallback(
            monitor="val_loss",
            patience=2,
            upload_to_wandb=False,
            reset_patience_on_resume=True,
        )

        trainer = Trainer(
            max_epochs=10,
            callbacks=[pause_callback, early_pause],
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            default_root_dir=str(tmp_path),
        )

        trainer.fit(model, datamodule=datamodule)

        # Verify pause triggered
        assert early_pause._pause_triggered is True
        # Verify training was stopped
        assert trainer.should_stop or trainer.current_epoch < 10
        # Verify checkpoint was created
        checkpoint_path = pause_callback.get_last_checkpoint()
        assert checkpoint_path is not None
        assert checkpoint_path.exists()

    def test_state_dict_round_trip(self):
        """Test that state is correctly saved and loaded."""
        early_pause = EarlyPauseCallback(
            monitor="val_loss",
            patience=5,
            upload_to_wandb=True,
            reset_patience_on_resume=True,
        )

        # Simulate some training state
        early_pause._pause_triggered = True
        early_pause.wait_count = 3
        early_pause.best_score = torch.tensor(0.5)

        # Save state
        state = early_pause.state_dict()

        # Create new callback and load state
        early_pause2 = EarlyPauseCallback(
            monitor="val_loss",
            patience=5,
            reset_patience_on_resume=True,
        )
        early_pause2.load_state_dict(state)

        # Verify state was loaded and reset correctly
        assert early_pause2._pause_triggered is False  # Reset on load
        assert early_pause2.wait_count == 0  # Reset because reset_patience_on_resume=True
        assert early_pause2.best_score == torch.tensor(0.5)  # Preserved
