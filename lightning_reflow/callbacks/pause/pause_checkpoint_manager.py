"""
Checkpoint manager for pause functionality.

This module handles all checkpoint-related operations for the PauseCallback:
- Checkpoint path generation
- Checkpoint saving with validation
- Atomic save operations
- Trainer state validation for pause
"""

import time
import shutil
import torch
from pathlib import Path
from typing import Optional, Callable, Dict, Any

from lightning.pytorch import Trainer, LightningModule


class PauseCheckpointManager:
    """
    Manages checkpoint operations for pause functionality.

    Handles checkpoint path generation, saving, validation, and atomic operations.
    Designed to be used as a composition component by PauseCallback.

    Args:
        checkpoint_dir: Directory to save pause checkpoints

    Example:
        manager = PauseCheckpointManager(Path("pause_checkpoints"))
        checkpoint_path = manager.get_checkpoint_path(trainer, upload=True)
        manager.save_checkpoint_with_validation(trainer, pl_module, checkpoint_path)
    """

    def __init__(self, checkpoint_dir: Path):
        """
        Initialize the checkpoint manager.

        Args:
            checkpoint_dir: Directory for saving pause checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Track last checkpoint path for HPO integration
        self.last_checkpoint_path: Optional[Path] = None

    def get_checkpoint_path(self, trainer: Trainer, upload: bool = False) -> Path:
        """
        Generate a checkpoint path with timestamp and metadata.

        Args:
            trainer: PyTorch Lightning trainer
            upload: If True, tag checkpoint for upload

        Returns:
            Path to the checkpoint file
        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time())
        tag = "upload" if upload else "pause"
        filename = f"{tag}_epoch={trainer.current_epoch}_step={trainer.global_step}_{timestamp}.ckpt"
        return self.checkpoint_dir / filename

    def save_checkpoint(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        checkpoint_path: Path
    ) -> None:
        """
        Save checkpoint using trainer's built-in method.

        This is a simple save without validation. Use save_checkpoint_with_validation
        for production-grade saves with atomic operations.

        Args:
            trainer: PyTorch Lightning trainer
            pl_module: Lightning module being trained
            checkpoint_path: Path to save checkpoint
        """
        trainer.save_checkpoint(checkpoint_path)
        self.last_checkpoint_path = checkpoint_path

    def save_checkpoint_with_validation(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        checkpoint_path: Path,
        config_metadata_fn: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> None:
        """
        Save checkpoint with validation and atomic operation.

        This method:
        1. Saves to a temporary file first
        2. Validates the checkpoint structure and size
        3. Optionally adds config metadata via callback
        4. Atomically moves to final location

        Args:
            trainer: PyTorch Lightning trainer
            pl_module: Lightning module being trained
            checkpoint_path: Target checkpoint path
            config_metadata_fn: Optional callback to add config metadata.
                               Called with checkpoint dict to modify in-place.

        Raises:
            RuntimeError: If checkpoint save or validation fails
        """
        # Create temporary checkpoint path for atomic operation
        temp_checkpoint_path = checkpoint_path.with_suffix('.tmp')

        try:
            # Save to temporary file first
            trainer.save_checkpoint(temp_checkpoint_path)

            # Validate the checkpoint was created and is readable
            if not temp_checkpoint_path.exists():
                raise RuntimeError(f"Checkpoint was not created at {temp_checkpoint_path}")

            checkpoint_size = temp_checkpoint_path.stat().st_size
            if checkpoint_size < 1024:  # Less than 1KB is suspicious
                raise RuntimeError(f"Checkpoint file too small ({checkpoint_size} bytes) - likely corrupted")

            # Try to load and validate the checkpoint structure
            try:
                checkpoint_dict = torch.load(temp_checkpoint_path, map_location='cpu', weights_only=False)
                required_keys = ['state_dict', 'epoch', 'global_step']
                missing_keys = [key for key in required_keys if key not in checkpoint_dict]
                if missing_keys:
                    raise RuntimeError(f"Checkpoint missing required keys: {missing_keys}")
            except Exception as e:
                raise RuntimeError(f"Checkpoint validation failed: {e}")

            # Add config metadata if callback provided
            if config_metadata_fn is not None:
                try:
                    # Load the saved checkpoint to add metadata
                    checkpoint = torch.load(temp_checkpoint_path, map_location='cpu', weights_only=False)

                    # Call the metadata function to add config
                    config_metadata_fn(checkpoint)

                    # Save back to temporary file
                    torch.save(checkpoint, temp_checkpoint_path)
                    print(f"Added config metadata to pause checkpoint")

                except Exception as e:
                    print(f"Could not add config metadata to checkpoint: {e}")
                    # Continue without metadata - not critical for pause functionality

            # Atomic move from temporary to final location
            temp_checkpoint_path.rename(checkpoint_path)
            print(f"Checkpoint atomically saved to {checkpoint_path} ({checkpoint_size:,} bytes)")

            # Track the checkpoint path
            self.last_checkpoint_path = checkpoint_path

        except Exception as e:
            # Clean up temporary file on any failure
            if temp_checkpoint_path.exists():
                try:
                    temp_checkpoint_path.unlink()
                except Exception:
                    pass
            raise RuntimeError(f"Failed to save pause checkpoint: {e}")

    def validate_trainer_state_for_pause(
        self,
        trainer: Trainer,
        pl_module: LightningModule
    ) -> bool:
        """
        Validate that trainer and module state is safe for pause checkpoint creation.

        Performs several safety checks:
        - Trainer has required attributes
        - pl_module is valid
        - Checkpoint directory is accessible
        - Sufficient disk space available

        Args:
            trainer: PyTorch Lightning trainer
            pl_module: Lightning module being trained

        Returns:
            True if state is valid for pause, False otherwise
        """
        # Check trainer has required attributes for checkpointing
        required_trainer_attrs = ['global_step', 'current_epoch', 'logger']
        for attr in required_trainer_attrs:
            if not hasattr(trainer, attr):
                print(f"Trainer missing required attribute for pause: {attr}")
                return False

        # Check pl_module is valid
        if pl_module is None:
            print(f"LightningModule is None - cannot create pause checkpoint")
            return False

        # Check checkpoint directory is accessible
        try:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Cannot access checkpoint directory {self.checkpoint_dir}: {e}")
            return False

        # Check we have disk space (basic check)
        try:
            free_space = shutil.disk_usage(self.checkpoint_dir).free
            if free_space < 100 * 1024 * 1024:  # Less than 100MB
                print(f"Low disk space for pause checkpoint: {free_space / (1024*1024):.1f} MB")
                return False
        except Exception as e:
            print(f"Could not check disk space: {e}")
            # Continue anyway - disk space check is optional

        return True

    def get_last_checkpoint(self) -> Optional[Path]:
        """
        Get the last saved checkpoint path (for HPO integration).

        Returns:
            Path to last checkpoint, or None if no checkpoint saved yet
        """
        return self.last_checkpoint_path
