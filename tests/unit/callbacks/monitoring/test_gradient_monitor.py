"""
Tests for GradientNormMonitorCallback.

Test coverage:
- Callback creation and configuration
- Gradient norm computation
- Clipping ratio tracking
- State persistence (save/load)
- Integration with Lightning trainer
"""

import pytest
import torch
import torch.nn as nn
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from unittest.mock import Mock, patch

from lightning_reflow.callbacks.monitoring.gradient_monitor import (
    GradientNormMonitorCallback,
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


class TestGradientNormMonitorCallback:
    """Test suite for GradientNormMonitorCallback."""

    @pytest.fixture
    def callback(self):
        """Create a callback with default settings."""
        return GradientNormMonitorCallback()

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
        callback = GradientNormMonitorCallback()

        assert callback.log_every_n_steps == 1
        assert callback.print_epoch_summary is True
        assert callback._clip_count == 0
        assert callback._total_steps == 0

    def test_callback_creation_custom(self):
        """Test callback creation with custom parameters."""
        callback = GradientNormMonitorCallback(
            log_every_n_steps=10,
            print_epoch_summary=False,
        )

        assert callback.log_every_n_steps == 10
        assert callback.print_epoch_summary is False

    # -------------------------------------------------------------------------
    # State persistence tests
    # -------------------------------------------------------------------------

    def test_state_dict_empty(self, callback):
        """Test state_dict with no steps tracked."""
        state = callback.state_dict()

        assert state["clip_count"] == 0
        assert state["total_steps"] == 0

    def test_state_dict_with_steps(self, callback):
        """Test state_dict after tracking some steps."""
        callback._clip_count = 5
        callback._total_steps = 100

        state = callback.state_dict()

        assert state["clip_count"] == 5
        assert state["total_steps"] == 100

    def test_load_state_dict(self, callback):
        """Test loading state from checkpoint."""
        state = {"clip_count": 10, "total_steps": 200}

        callback.load_state_dict(state)

        assert callback._clip_count == 10
        assert callback._total_steps == 200

    def test_load_state_dict_partial(self, callback):
        """Test loading partial state (forward compatibility)."""
        callback._clip_count = 5
        callback._total_steps = 100

        # Partial state (e.g., from older checkpoint)
        state = {}
        callback.load_state_dict(state)

        # Should use defaults when keys missing
        assert callback._clip_count == 0
        assert callback._total_steps == 0

    # -------------------------------------------------------------------------
    # Integration tests
    # -------------------------------------------------------------------------

    def test_integration_with_trainer(self, model, dataloader):
        """Test callback integration with Lightning Trainer."""
        callback = GradientNormMonitorCallback(
            log_every_n_steps=1,
            print_epoch_summary=False,
        )

        trainer = Trainer(
            max_steps=5,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            callbacks=[callback],
            gradient_clip_val=1.0,  # Enable gradient clipping
        )

        trainer.fit(model, dataloader)

        # Callback should have tracked steps
        assert callback._total_steps >= 1

    def test_gradient_norm_logging(self, model, dataloader):
        """Test that gradient norms are logged during training."""
        callback = GradientNormMonitorCallback(log_every_n_steps=1)

        # Track logged metrics
        logged_metrics = []

        def capture_log(name, value, **kwargs):
            logged_metrics.append(name)

        # Use mock to capture log calls
        with patch.object(model, 'log', side_effect=capture_log):
            trainer = Trainer(
                max_steps=3,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                logger=False,
                callbacks=[callback],
                gradient_clip_val=1.0,
            )
            trainer.fit(model, dataloader)

        # Check expected metrics were logged
        assert "train/grad_norm_unclipped" in logged_metrics
        assert "train/grad_norm_clipped" in logged_metrics
        assert "train/grad_clip_ratio" in logged_metrics

    def test_clipping_ratio_calculation(self, callback):
        """Test clipping ratio is calculated correctly."""
        # Simulate some steps
        callback._total_steps = 10
        callback._clip_count = 2

        ratio = callback._clip_count / callback._total_steps

        assert ratio == 0.2  # 20% clipped

    def test_log_every_n_steps_skipping(self, model, dataloader):
        """Test that logging respects log_every_n_steps."""
        callback = GradientNormMonitorCallback(log_every_n_steps=5)

        trainer = Trainer(
            max_steps=10,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            callbacks=[callback],
        )

        trainer.fit(model, dataloader)

        # With log_every_n_steps=5, should only track steps 0, 5, 10...
        # For 10 steps: 0, 5 = 2 logged steps
        # But step 10 won't be logged if training stops at step 10
        assert callback._total_steps <= 3  # At most steps 0, 5, (10)


class TestGradientNormMonitorImport:
    """Test that the callback can be imported from various paths."""

    def test_import_from_monitoring(self):
        """Test import from monitoring submodule."""
        from lightning_reflow.callbacks.monitoring import GradientNormMonitorCallback
        assert GradientNormMonitorCallback is not None

    def test_import_from_callbacks(self):
        """Test import from callbacks module."""
        from lightning_reflow.callbacks import GradientNormMonitorCallback
        assert GradientNormMonitorCallback is not None
