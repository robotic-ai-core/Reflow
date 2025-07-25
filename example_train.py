#!/usr/bin/env python3
"""
Lightning Reflow CLI training example.

This demonstrates how to use the LightningReflowCLI with the enhanced
features like pause/resume, W&B integration, and config management.

Usage examples:
    # Basic training with default settings
    python example_train.py fit
    
    # Training with custom parameters
    python example_train.py fit --model.learning_rate=0.01 --trainer.max_epochs=20
    
    # Training with config file
    python example_train.py fit --config example_config.yaml
    
    # Show help for all options
    python example_train.py fit --help
    
    # Resume from checkpoint (Lightning Reflow feature)
    python example_train.py resume --checkpoint-path ./checkpoints/last.ckpt
    
    # Resume from W&B artifact (Lightning Reflow feature)  
    python example_train.py resume --checkpoint-artifact entity/project/run-id:latest

Key Lightning Reflow features demonstrated:
    - Enhanced CLI with resume subcommand
    - Pause/resume functionality (press 'p' during training)
    - W&B integration with checkpoint artifacts
    - Config embedding in checkpoints
    - Memory cleanup and environment management
"""

from lightning_reflow import LightningReflowCLI
from lightning_reflow.models import SimpleReflowModel
from lightning_reflow.data import SimpleDataModule
from lightning_reflow.callbacks import (
    PauseCallback,
    WandbWatchCallback,
    ConfigSummaryLogger,
    MemoryCleanupCallback,
    EnvironmentCallback
)


def main():
    """Run Lightning Reflow CLI training example."""
    
    # Define trainer defaults with Lightning Reflow callbacks
    trainer_defaults = {
        "max_epochs": 5,
        "max_steps": 50,  # Limit steps for quick example
        "enable_checkpointing": True,
        "check_val_every_n_epoch": 1,
        "log_every_n_steps": 10,
                 "callbacks": [
             PauseCallback(
                 checkpoint_dir="./checkpoints",
                 enable_pause=True,
                 pause_key='p',
                 upload_key='w'
             ),
             WandbWatchCallback(log_level="gradients", log_freq=100),
             ConfigSummaryLogger(),
             MemoryCleanupCallback(),
             EnvironmentCallback()
         ]
    }
    
    # Initialize Lightning Reflow CLI with enhanced features
    cli = LightningReflowCLI(
        model_class=SimpleReflowModel,
        datamodule_class=SimpleDataModule,
        trainer_defaults=trainer_defaults,
        seed_everything_default=42,
        save_config_kwargs={"overwrite": True},
        parser_kwargs={"default_env": False},
        subclass_mode_model=True,
        subclass_mode_data=True
    )


if __name__ == "__main__":
    main()