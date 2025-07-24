#!/usr/bin/env python3
"""
Simple training example using Lightning Reflow minimal pipeline.

This demonstrates basic usage of the Lightning Reflow framework with
the minimal test components.
"""

import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger

from models import SimpleReflowModel
from data import SimpleDataModule
from callbacks.pause.pause_callback import PauseCallback


def main():
    """Run simple training example."""
    
    # Set seed for reproducibility
    pl.seed_everything(42)
    
    # Create model
    model = SimpleReflowModel(
        input_dim=20,
        hidden_dim=64,
        output_dim=3,
        learning_rate=0.01,
        loss_type="cross_entropy"  # Classification task
    )
    
    # Create data module
    data_module = SimpleDataModule(
        batch_size=16,
        train_samples=200,
        val_samples=40,
        test_samples=20,
        input_dim=20,
        output_dim=3,
        task_type="classification",
        seed=42
    )
    
    # Create callbacks
    callbacks = []
    
    # Add pause callback for pause/resume functionality
    try:
        pause_callback = PauseCallback(
            checkpoint_dir="./checkpoints",
            wandb_project="lightning-reflow-example"
        )
        callbacks.append(pause_callback)
        print("✓ Pause callback enabled (press 'p' to pause, 'w' to upload)")
    except Exception as e:
        print(f"⚠ Pause callback disabled: {e}")
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=5,
        max_steps=20,  # Limit steps for quick example
        logger=CSVLogger("logs", name="simple_example"),
        callbacks=callbacks,
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        log_every_n_steps=5
    )
    
    # Print model info
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training samples: {data_module.train_samples}")
    print(f"Validation samples: {data_module.val_samples}")
    
    # Train
    print("\nStarting training...")
    trainer.fit(model, data_module)
    
    # Test
    print("\nRunning test...")
    trainer.test(model, data_module)
    
    print("\n✓ Training completed!")
    print(f"Logs saved to: {trainer.logger.log_dir}")
    
    if trainer.checkpoint_callback and trainer.checkpoint_callback.best_model_path:
        print(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()