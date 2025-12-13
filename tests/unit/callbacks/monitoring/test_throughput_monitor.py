"""
Tests for ThroughputMonitorCallback.

Test coverage:
- Callback creation and configuration
- Batch size extraction from various formats
- EMA throughput calculation
- Training metrics logging
- Validation metrics logging
- State persistence (save/load)
- Integration with Lightning trainer
"""

import pytest
import time
import torch
import torch.nn as nn
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from unittest.mock import Mock, patch, MagicMock

from lightning_reflow.callbacks.monitoring.throughput_monitor import (
    ThroughputMonitorCallback,
    DEFAULT_EMA_SMOOTHING,
)


class SimpleModel(pl.LightningModule):
    """Simple LightningModule for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


class SimpleDummyDataset(torch.utils.data.Dataset):
    """Simple dataset for testing."""

    def __init__(self, size=100):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = torch.randn(10)
        y = torch.randn(2)
        return x, y


class TestThroughputMonitorCallback:
    """Test suite for ThroughputMonitorCallback."""

    @pytest.fixture
    def callback(self):
        """Create a callback with default settings."""
        return ThroughputMonitorCallback()

    @pytest.fixture
    def model(self):
        """Create a test model."""
        return SimpleModel()

    @pytest.fixture
    def dataloader(self):
        """Create a test dataloader."""
        dataset = SimpleDummyDataset(size=20)
        return torch.utils.data.DataLoader(dataset, batch_size=4)

    # -------------------------------------------------------------------------
    # Initialization tests
    # -------------------------------------------------------------------------

    def test_callback_creation_default(self):
        """Test callback creation with default parameters."""
        callback = ThroughputMonitorCallback()

        assert callback.ema_smoothing == DEFAULT_EMA_SMOOTHING
        assert callback.log_on_step is True
        assert callback.log_on_epoch is True
        assert callback._train_total_samples == 0
        assert callback._val_total_samples == 0
        assert callback._val_run_count == 0
        assert callback._train_throughput_ema is None

    def test_callback_creation_custom(self):
        """Test callback creation with custom parameters."""
        callback = ThroughputMonitorCallback(
            ema_smoothing=0.2,
            log_on_step=False,
            log_on_epoch=False,
        )

        assert callback.ema_smoothing == 0.2
        assert callback.log_on_step is False
        assert callback.log_on_epoch is False

    # -------------------------------------------------------------------------
    # Batch size extraction tests
    # -------------------------------------------------------------------------

    def test_get_batch_size_dict_observation_images(self, callback):
        """Test batch size extraction from dict with observation.images key."""
        batch = {"observation.images": torch.randn(8, 3, 64, 64)}
        assert callback._get_batch_size(batch) == 8

    def test_get_batch_size_dict_input(self, callback):
        """Test batch size extraction from dict with input key."""
        batch = {"input": torch.randn(16, 10)}
        assert callback._get_batch_size(batch) == 16

    def test_get_batch_size_dict_x(self, callback):
        """Test batch size extraction from dict with x key."""
        batch = {"x": torch.randn(4, 20), "y": torch.randn(4)}
        assert callback._get_batch_size(batch) == 4

    def test_get_batch_size_dict_fallback(self, callback):
        """Test batch size extraction from dict with unknown keys."""
        batch = {"custom_key": torch.randn(12, 5)}
        assert callback._get_batch_size(batch) == 12

    def test_get_batch_size_tuple(self, callback):
        """Test batch size extraction from tuple."""
        batch = (torch.randn(6, 10), torch.randn(6, 2))
        assert callback._get_batch_size(batch) == 6

    def test_get_batch_size_list(self, callback):
        """Test batch size extraction from list."""
        batch = [torch.randn(5, 10), torch.randn(5)]
        assert callback._get_batch_size(batch) == 5

    def test_get_batch_size_tensor(self, callback):
        """Test batch size extraction from direct tensor."""
        batch = torch.randn(7, 10)
        assert callback._get_batch_size(batch) == 7

    def test_get_batch_size_unknown_format(self, callback):
        """Test batch size extraction from unknown format defaults to 1."""
        batch = "unknown"
        with patch.object(callback, '_get_batch_size', wraps=callback._get_batch_size):
            size = callback._get_batch_size(batch)
            assert size == 1

    # -------------------------------------------------------------------------
    # EMA calculation tests
    # -------------------------------------------------------------------------

    def test_update_ema_initial(self, callback):
        """Test EMA update when no previous value exists."""
        result = callback._update_ema(100.0, None)
        assert result == 100.0

    def test_update_ema_subsequent(self, callback):
        """Test EMA update with existing value."""
        # With smoothing=0.1: new_ema = 0.1 * 200 + 0.9 * 100 = 110
        callback.ema_smoothing = 0.1
        result = callback._update_ema(200.0, 100.0)
        assert result == pytest.approx(110.0)

    def test_update_ema_convergence(self, callback):
        """Test EMA converges to constant value."""
        callback.ema_smoothing = 0.5
        ema = None
        constant_value = 50.0

        for _ in range(20):
            ema = callback._update_ema(constant_value, ema)

        assert ema == pytest.approx(constant_value, rel=0.01)

    # -------------------------------------------------------------------------
    # Training hooks tests
    # -------------------------------------------------------------------------

    def test_on_train_epoch_start(self, callback):
        """Test epoch start resets counters."""
        callback._train_epoch_samples = 100
        callback._train_epoch_start_time = 123.0

        trainer = Mock()
        pl_module = Mock()

        callback.on_train_epoch_start(trainer, pl_module)

        assert callback._train_epoch_samples == 0
        assert callback._train_epoch_start_time is not None

    def test_on_train_batch_start(self, callback):
        """Test batch start records time."""
        trainer = Mock()
        pl_module = Mock()
        batch = torch.randn(4, 10)

        callback.on_train_batch_start(trainer, pl_module, batch, 0)

        assert callback._train_batch_start_time is not None

    def test_on_train_batch_end_updates_counters(self, callback):
        """Test batch end updates sample counters."""
        trainer = Mock()
        trainer.global_step = 10
        trainer.log_every_n_steps = 10

        pl_module = Mock()
        batch = (torch.randn(8, 10), torch.randn(8, 2))

        callback._train_batch_start_time = time.perf_counter() - 0.1
        callback.on_train_batch_end(trainer, pl_module, None, batch, 0)

        assert callback._train_total_samples == 8
        assert callback._train_epoch_samples == 8
        assert callback._train_throughput_ema is not None

    def test_on_train_batch_end_logs_at_interval(self, callback):
        """Test batch end logs at trainer's interval."""
        trainer = Mock()
        trainer.global_step = 10
        trainer.log_every_n_steps = 10

        pl_module = Mock()
        batch = (torch.randn(4, 10), torch.randn(4, 2))

        callback._train_batch_start_time = time.perf_counter() - 0.1
        callback._train_throughput_ema = 100.0
        callback.on_train_batch_end(trainer, pl_module, None, batch, 0)

        # Should have logged train/throughput and train/total_samples
        assert pl_module.log.call_count >= 2

    def test_on_train_batch_end_skips_log_not_at_interval(self, callback):
        """Test batch end skips logging when not at interval."""
        trainer = Mock()
        trainer.global_step = 5
        trainer.log_every_n_steps = 10

        pl_module = Mock()
        batch = (torch.randn(4, 10), torch.randn(4, 2))

        callback._train_batch_start_time = time.perf_counter() - 0.1
        callback.on_train_batch_end(trainer, pl_module, None, batch, 0)

        # Should NOT have logged
        pl_module.log.assert_not_called()

    def test_on_train_epoch_end_logs_throughput(self, callback):
        """Test epoch end logs epoch throughput."""
        trainer = Mock()
        pl_module = Mock()

        callback._train_epoch_start_time = time.perf_counter() - 1.0
        callback._train_epoch_samples = 100

        callback.on_train_epoch_end(trainer, pl_module)

        # Should have logged train/epoch_throughput
        pl_module.log.assert_called_once()
        call_args = pl_module.log.call_args
        assert call_args[0][0] == "train/epoch_throughput"

    # -------------------------------------------------------------------------
    # Validation hooks tests
    # -------------------------------------------------------------------------

    def test_on_validation_epoch_start(self, callback):
        """Test validation epoch start resets counters."""
        callback._val_epoch_samples = 50

        trainer = Mock()
        pl_module = Mock()

        callback.on_validation_epoch_start(trainer, pl_module)

        assert callback._val_epoch_samples == 0
        assert callback._val_epoch_start_time is not None

    def test_on_validation_batch_end_updates_counters(self, callback):
        """Test validation batch end updates counters."""
        trainer = Mock()
        pl_module = Mock()
        batch = (torch.randn(8, 10), torch.randn(8, 2))

        callback._val_batch_start_time = time.perf_counter()
        callback.on_validation_batch_end(trainer, pl_module, None, batch, 0)

        assert callback._val_epoch_samples == 8
        assert callback._val_total_samples == 8

    def test_on_validation_epoch_end_logs_metrics(self, callback):
        """Test validation epoch end logs all metrics."""
        trainer = Mock()
        pl_module = Mock()

        callback._val_epoch_start_time = time.perf_counter() - 0.5
        callback._val_epoch_samples = 50
        callback._val_total_samples = 100

        callback.on_validation_epoch_end(trainer, pl_module)

        # Should log val/throughput, val/total_samples, val/epoch_samples, val/run_count
        assert pl_module.log.call_count == 4

        logged_metrics = [call[0][0] for call in pl_module.log.call_args_list]
        assert "val/throughput" in logged_metrics
        assert "val/total_samples" in logged_metrics
        assert "val/epoch_samples" in logged_metrics
        assert "val/run_count" in logged_metrics

    def test_on_validation_epoch_end_increments_run_count(self, callback):
        """Test validation epoch end increments run count."""
        trainer = Mock()
        pl_module = Mock()

        callback._val_epoch_start_time = time.perf_counter() - 0.1
        callback._val_epoch_samples = 10

        assert callback._val_run_count == 0

        callback.on_validation_epoch_end(trainer, pl_module)
        assert callback._val_run_count == 1

        # Reset for another validation run
        callback._val_epoch_start_time = time.perf_counter() - 0.1
        callback._val_epoch_samples = 10

        callback.on_validation_epoch_end(trainer, pl_module)
        assert callback._val_run_count == 2

    # -------------------------------------------------------------------------
    # State persistence tests
    # -------------------------------------------------------------------------

    def test_state_dict(self, callback):
        """Test state dict contains expected keys."""
        callback._train_total_samples = 1000
        callback._val_total_samples = 200
        callback._val_run_count = 5
        callback._train_throughput_ema = 50.5

        state = callback.state_dict()

        assert state["train_total_samples"] == 1000
        assert state["val_total_samples"] == 200
        assert state["val_run_count"] == 5
        assert state["train_throughput_ema"] == 50.5

    def test_load_state_dict(self, callback):
        """Test load state dict restores values."""
        state = {
            "train_total_samples": 5000,
            "val_total_samples": 1000,
            "val_run_count": 10,
            "train_throughput_ema": 75.0,
        }

        callback.load_state_dict(state)

        assert callback._train_total_samples == 5000
        assert callback._val_total_samples == 1000
        assert callback._val_run_count == 10
        assert callback._train_throughput_ema == 75.0

    def test_load_state_dict_missing_keys(self, callback):
        """Test load state dict handles missing keys gracefully."""
        state = {"train_total_samples": 100}

        callback.load_state_dict(state)

        assert callback._train_total_samples == 100
        assert callback._val_total_samples == 0
        assert callback._val_run_count == 0
        assert callback._train_throughput_ema is None

    # -------------------------------------------------------------------------
    # Integration tests
    # -------------------------------------------------------------------------

    def test_integration_with_trainer(self, model, dataloader):
        """Test callback integrates correctly with Lightning trainer."""
        callback = ThroughputMonitorCallback(ema_smoothing=0.5)

        trainer = Trainer(
            max_epochs=1,
            max_steps=5,
            callbacks=[callback],
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=False,
        )

        trainer.fit(model, dataloader, val_dataloaders=dataloader)

        # Check that samples were counted
        assert callback._train_total_samples > 0
        assert callback._val_total_samples > 0
        assert callback._train_throughput_ema is not None

    def test_integration_throughput_reasonable(self, model, dataloader):
        """Test that throughput values are reasonable."""
        callback = ThroughputMonitorCallback()

        trainer = Trainer(
            max_epochs=1,
            max_steps=10,
            callbacks=[callback],
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=False,
        )

        trainer.fit(model, dataloader)

        # Throughput should be positive and reasonable (not 0 or inf)
        assert callback._train_throughput_ema is not None
        assert callback._train_throughput_ema > 0
        assert callback._train_throughput_ema < 1e9  # Less than 1 billion samples/sec

    def test_log_on_step_false_skips_step_logging(self, callback):
        """Test that log_on_step=False skips per-step logging."""
        callback = ThroughputMonitorCallback(log_on_step=False)

        trainer = Mock()
        trainer.global_step = 10
        trainer.log_every_n_steps = 10

        pl_module = Mock()
        batch = (torch.randn(4, 10), torch.randn(4, 2))

        callback._train_batch_start_time = time.perf_counter() - 0.1
        callback.on_train_batch_end(trainer, pl_module, None, batch, 0)

        # Should NOT have logged due to log_on_step=False
        pl_module.log.assert_not_called()

    def test_log_on_epoch_false_skips_epoch_logging(self, callback):
        """Test that log_on_epoch=False skips epoch logging."""
        callback = ThroughputMonitorCallback(log_on_epoch=False)

        trainer = Mock()
        pl_module = Mock()

        callback._train_epoch_start_time = time.perf_counter() - 1.0
        callback._train_epoch_samples = 100

        callback.on_train_epoch_end(trainer, pl_module)

        # Should NOT have logged due to log_on_epoch=False
        pl_module.log.assert_not_called()


class TestThroughputMonitorCallbackImport:
    """Test that callback can be imported from expected locations."""

    def test_import_from_monitoring(self):
        """Test import from monitoring subpackage."""
        from lightning_reflow.callbacks.monitoring import ThroughputMonitorCallback
        assert ThroughputMonitorCallback is not None

    def test_import_from_callbacks(self):
        """Test import from callbacks package."""
        from lightning_reflow.callbacks import ThroughputMonitorCallback
        assert ThroughputMonitorCallback is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
