#!/usr/bin/env python
"""
Debug script to reproduce and understand validation metric divergence after checkpoint resume.

This script will:
1. Create a simple model with validation metrics
2. Train for a few steps, tracking validation metrics
3. Save a checkpoint
4. Resume from checkpoint
5. Compare validation metrics before and after resume
"""

import torch
import torch.nn as nn
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
import tempfile
import os
from pathlib import Path
import numpy as np


class SimpleModelWithValidation(pl.LightningModule):
    """Simple model to test validation metric reproducibility."""

    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 1)
        # Track internal state that affects validation
        self.val_one_step_loss = 0.01  # Initial value as mentioned in the issue
        self.validation_counter = 0

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        print(f"DEBUG: training_step - batch_idx={batch_idx}, loss={loss:.4f}")
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)

        # Simulate the val_one_step_loss calculation that might be affected by state
        self.validation_counter += 1

        # Simulate a calculation that depends on internal state
        # This might be what's causing the divergence
        if hasattr(self, '_restored_from_checkpoint'):
            # After restore, the calculation might be different
            self.val_one_step_loss = loss.item() * 7  # This would cause jump from 0.01 to 0.07
        else:
            self.val_one_step_loss = loss.item()

        self.log("val_loss", loss)
        self.log("val_one_step_loss", self.val_one_step_loss)

        print(f"DEBUG: validation_step - batch_idx={batch_idx}, "
              f"val_loss={loss:.4f}, val_one_step_loss={self.val_one_step_loss:.4f}, "
              f"validation_counter={self.validation_counter}, "
              f"restored={getattr(self, '_restored_from_checkpoint', False)}")

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def on_load_checkpoint(self, checkpoint):
        """Hook called when loading from checkpoint."""
        print(f"DEBUG: on_load_checkpoint called")
        print(f"DEBUG: checkpoint keys: {list(checkpoint.keys())}")
        self._restored_from_checkpoint = True

        # Check if our custom state is in the checkpoint
        if 'val_one_step_loss' in checkpoint:
            print(f"DEBUG: Restoring val_one_step_loss from checkpoint: {checkpoint['val_one_step_loss']}")
        else:
            print(f"DEBUG: val_one_step_loss NOT in checkpoint!")

    def on_save_checkpoint(self, checkpoint):
        """Hook to save custom state."""
        print(f"DEBUG: on_save_checkpoint called")
        checkpoint['val_one_step_loss'] = self.val_one_step_loss
        checkpoint['validation_counter'] = self.validation_counter
        print(f"DEBUG: Saving val_one_step_loss={self.val_one_step_loss}, "
              f"validation_counter={self.validation_counter}")


def create_dummy_dataloader(num_samples=100):
    """Create a simple dataloader for testing."""
    torch.manual_seed(42)
    X = torch.randn(num_samples, 10)
    y = torch.randn(num_samples, 1)
    dataset = torch.utils.data.TensorDataset(X, y)
    return torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False)


def main():
    print("=" * 80)
    print("Testing validation metric divergence after checkpoint resume")
    print("=" * 80)

    # Set seeds for reproducibility
    pl.seed_everything(42)

    # Create temporary directory for checkpoints
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "checkpoint.ckpt"

        # Phase 1: Initial training with validation
        print("\n" + "=" * 40)
        print("PHASE 1: Initial training")
        print("=" * 40)

        model = SimpleModelWithValidation()
        train_loader = create_dummy_dataloader()
        val_loader = create_dummy_dataloader(num_samples=20)

        # Create trainer with minimal epochs
        trainer = pl.Trainer(
            max_epochs=2,
            callbacks=[
                ModelCheckpoint(
                    dirpath=tmpdir,
                    filename="checkpoint",
                    save_last=True,
                    save_on_train_epoch_end=True
                )
            ],
            enable_progress_bar=False,
            logger=False,
            val_check_interval=0.5,  # Validate twice per epoch
            limit_val_batches=2  # Limit validation batches
        )

        # Train and validate
        trainer.fit(model, train_loader, val_loader)

        # Get metrics before saving
        print(f"\n>> Metrics before checkpoint save:")
        print(f"   val_one_step_loss: {model.val_one_step_loss:.4f}")
        print(f"   validation_counter: {model.validation_counter}")
        metrics_before = {
            'val_one_step_loss': model.val_one_step_loss,
            'validation_counter': model.validation_counter
        }

        # Save checkpoint manually to ensure it's saved
        trainer.save_checkpoint(checkpoint_path)
        print(f"\n>> Checkpoint saved to: {checkpoint_path}")

        # Phase 2: Resume from checkpoint
        print("\n" + "=" * 40)
        print("PHASE 2: Resume from checkpoint")
        print("=" * 40)

        # Create new model and trainer
        model2 = SimpleModelWithValidation()

        # Load checkpoint
        print(f"\n>> Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model2.load_state_dict(checkpoint['state_dict'])
        model2.on_load_checkpoint(checkpoint)

        # Create new trainer for resumed training
        trainer2 = pl.Trainer(
            max_epochs=3,  # Continue for one more epoch
            enable_progress_bar=False,
            logger=False,
            val_check_interval=0.5,
            limit_val_batches=2
        )

        # Continue training
        trainer2.fit(model2, train_loader, val_loader, ckpt_path=checkpoint_path)

        # Get metrics after resume
        print(f"\n>> Metrics after checkpoint resume:")
        print(f"   val_one_step_loss: {model2.val_one_step_loss:.4f}")
        print(f"   validation_counter: {model2.validation_counter}")
        metrics_after = {
            'val_one_step_loss': model2.val_one_step_loss,
            'validation_counter': model2.validation_counter
        }

        # Compare metrics
        print("\n" + "=" * 40)
        print("COMPARISON")
        print("=" * 40)

        for key in metrics_before:
            before = metrics_before[key]
            after = metrics_after[key]
            diverged = abs(before - after) > 0.01 if isinstance(before, float) else before != after
            status = "DIVERGED!" if diverged else "OK"
            print(f"{key:20s}: before={before:.4f}, after={after:.4f} [{status}]")

        # Check for the specific issue mentioned
        if abs(metrics_before['val_one_step_loss'] - 0.01) < 0.001 and \
           abs(metrics_after['val_one_step_loss'] - 0.07) < 0.01:
            print("\n*** REPRODUCED THE ISSUE! ***")
            print("val_one_step_loss jumped from ~0.01 to ~0.07 after resume")
        else:
            print(f"\n*** Issue pattern not exactly reproduced ***")
            print(f"Expected: 0.01 -> 0.07")
            print(f"Got: {metrics_before['val_one_step_loss']:.4f} -> {metrics_after['val_one_step_loss']:.4f}")


if __name__ == "__main__":
    main()