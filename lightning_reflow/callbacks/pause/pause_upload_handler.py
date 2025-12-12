"""
W&B upload handler for pause functionality.

This module handles all W&B artifact upload operations for the PauseCallback:
- Artifact upload with metadata
- Fallback handling when upload fails
- Integration with WandbArtifactManager
"""

from pathlib import Path
from typing import Optional, Any

from lightning.pytorch import Trainer, LightningModule

from ...utils.wandb.wandb_artifact_manager import WandbArtifactManager


class PauseUploadHandler:
    """
    Handles W&B upload operations for pause checkpoints.

    Manages artifact upload, fallback handling, and metadata generation.
    Designed to be used as a composition component by PauseCallback.

    Args:
        wandb_manager: WandbArtifactManager instance for artifact operations

    Example:
        handler = PauseUploadHandler(wandb_manager)
        artifact_path = handler.handle_wandb_upload(trainer, pl_module, checkpoint_path)
    """

    def __init__(self, wandb_manager: WandbArtifactManager):
        """
        Initialize the upload handler.

        Args:
            wandb_manager: WandbArtifactManager instance for artifact operations
        """
        self._wandb_manager = wandb_manager

    def handle_wandb_upload(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        checkpoint_path: str
    ) -> Optional[str]:
        """
        Handle W&B upload and return artifact path if successful.

        Args:
            trainer: PyTorch Lightning trainer
            pl_module: Lightning module
            checkpoint_path: Path to checkpoint file

        Returns:
            Full artifact path if upload successful, None if W&B not available

        Raises:
            ValueError: If required inputs are invalid
        """
        # Fail early: Validate critical inputs
        if trainer is None:
            raise ValueError("Trainer cannot be None for W&B upload")
        if not hasattr(trainer, 'callbacks') or trainer.callbacks is None:
            raise ValueError("Trainer must have callbacks list for W&B upload")

        # Find WandbArtifactCheckpoint callback
        wandb_callback = None
        for callback in trainer.callbacks:
            # Check for WandbArtifactCheckpoint callback with upload_pause_checkpoint method
            if hasattr(callback, 'upload_pause_checkpoint'):
                wandb_callback = callback
                break
            # Fallback to old method name for backward compatibility
            elif hasattr(callback, '_upload_pause_checkpoint_artifact'):
                wandb_callback = callback
                break

        # Graceful degradation: No W&B callback is not an error, just no upload
        if not wandb_callback:
            print(f"No W&B callback found - checkpoint saved locally only")
            return None

        # Attempt upload with proper error handling
        try:
            # Use new method if available, otherwise use fallback
            if hasattr(wandb_callback, 'upload_pause_checkpoint'):
                artifact_path = wandb_callback.upload_pause_checkpoint(
                    trainer, trainer.lightning_module, checkpoint_path
                )
            else:
                artifact_path = self._upload_pause_checkpoint_artifact(
                    wandb_callback, trainer, checkpoint_path
                )
            print(f"Pause checkpoint uploaded to W&B successfully")
            return artifact_path

        except (ValueError, RuntimeError) as e:
            # Expected errors from upload method - log and gracefully degrade
            print(f"Failed to upload pause checkpoint to W&B: {e}")
            return None
        except Exception as e:
            # Unexpected errors - re-raise with context
            raise RuntimeError(f"Unexpected error during W&B upload: {e}") from e

    def _upload_pause_checkpoint_artifact(
        self,
        wandb_callback: Any,
        trainer: Trainer,
        checkpoint_path: str
    ) -> Optional[str]:
        """
        Upload pause checkpoint artifact using shared WandB artifact manager.

        Args:
            wandb_callback: WandbArtifactCheckpoint instance for compatibility
            trainer: PyTorch Lightning trainer
            checkpoint_path: Path to checkpoint file

        Returns:
            Full artifact path if upload successful (e.g. "entity/project/artifact:version")

        Raises:
            ValueError: If required inputs are invalid
            RuntimeError: If W&B manager is not available or configured incorrectly
        """
        # Fail early: Validate critical inputs
        if trainer is None:
            raise ValueError("Trainer cannot be None for artifact upload")
        if not checkpoint_path or not Path(checkpoint_path).exists():
            raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
        if self._wandb_manager is None:
            raise RuntimeError("W&B artifact manager is not initialized")

        # Fail early: Check W&B run availability
        wandb_run = self._wandb_manager.get_wandb_run(trainer)
        if not wandb_run:
            raise RuntimeError("No active W&B run found - cannot upload artifacts")

        # Fail early: Validate trainer state
        if not hasattr(trainer, 'lightning_module') or trainer.lightning_module is None:
            raise ValueError("Trainer must have a valid lightning_module for upload")

        try:
            # Create pause-specific metadata
            extra_metadata = {
                "pause_type": "manual_pause",
                "checkpoint_type": "pause_checkpoint",
                "pause_callback_version": "2.1"
            }

            # Upload using shared artifact manager - returns full artifact path
            artifact_path = self._wandb_manager.upload_checkpoint_artifact(
                trainer=trainer,
                pl_module=trainer.lightning_module,
                filepath=checkpoint_path,
                ckpt_type="pause",
                aliases=["pause", "latest"],
                score=None,  # Pause checkpoints don't have scores
                epoch=trainer.current_epoch,
                step=trainer.global_step,
                wandb_run=wandb_run,
                extra_metadata=extra_metadata
            )

            if not artifact_path:
                raise RuntimeError("Artifact upload returned None - upload failed")

            return artifact_path  # Returns full path like "entity/project/artifact:version"

        except (AttributeError, KeyError) as e:
            # Specific exceptions for missing attributes/keys
            raise RuntimeError(f"Missing required attribute for artifact upload: {e}") from e
        except Exception as e:
            # Re-raise with more context
            raise RuntimeError(f"Artifact upload failed: {e}") from e

    def handle_upload_with_fallback(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        checkpoint_path: str
    ) -> Optional[str]:
        """
        Handle W&B upload with comprehensive fallback.

        This method wraps handle_wandb_upload with additional error handling
        to ensure the pause operation continues even if upload fails.

        Args:
            trainer: PyTorch Lightning trainer
            pl_module: Lightning module
            checkpoint_path: Path to checkpoint file

        Returns:
            Full artifact path if upload successful, None otherwise
        """
        try:
            artifact_path = self.handle_wandb_upload(trainer, pl_module, checkpoint_path)
            if artifact_path:
                print(f"Pause checkpoint uploaded to W&B: {artifact_path}")
                return artifact_path
            else:
                print(f"W&B upload returned None - checkpoint saved locally only")
                return None

        except (ValueError, RuntimeError) as e:
            print(f"W&B upload failed but pause will continue: {e}")
            print(f"Checkpoint available locally at: {checkpoint_path}")
            return None
        except Exception as e:
            print(f"Unexpected error during W&B upload: {e}")
            print(f"Checkpoint available locally at: {checkpoint_path}")
            return None
