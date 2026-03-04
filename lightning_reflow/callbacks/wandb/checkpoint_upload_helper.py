"""
Helper class for checkpoint upload operations.

Extracts upload-related logic from WandbArtifactCheckpoint to reduce complexity.
"""

import gzip
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import torch
import lightning.pytorch as pl

from ...utils.logging.logging_config import get_logger

if TYPE_CHECKING:
    from .wandb_artifact_checkpoint import UploadReason, WandbCheckpointConfig, UploadState


class CheckpointUploadHelper:
    """
    Helper class that handles checkpoint upload operations.

    Extracts upload-related logic from WandbArtifactCheckpoint to improve
    maintainability and testability.
    """

    def __init__(self, config: "WandbCheckpointConfig"):
        """
        Initialize the upload helper.

        Args:
            config: Configuration from WandbArtifactCheckpoint
        """
        self.config = config
        self.logger = get_logger(__name__)

    def create_aliases(self, ckpt_type: str, reason: "UploadReason") -> List[str]:
        """
        Create aliases for the artifact.

        Args:
            ckpt_type: Type of checkpoint (best, latest, pause, etc.)
            reason: Reason for upload

        Returns:
            List of aliases for the artifact
        """
        from .wandb_artifact_checkpoint import UploadReason

        aliases = [ckpt_type]

        if reason == UploadReason.EXCEPTION:
            aliases.append("crash_recovery")
        elif reason == UploadReason.PAUSE_REQUESTED:
            aliases.append("pause")
        elif reason in [UploadReason.PERIODIC_VALIDATION, UploadReason.PERIODIC_EPOCH, UploadReason.PERIODIC_HOURS]:
            aliases.append("periodic")

        aliases.append("latest")  # Always mark as latest

        return aliases

    def get_extra_metadata(
        self,
        reason: "UploadReason",
        training_start_time: Optional[float]
    ) -> Dict[str, Any]:
        """
        Get extra metadata for the upload.

        Args:
            reason: Reason for upload
            training_start_time: When training started

        Returns:
            Dictionary of extra metadata
        """
        return {
            "upload_reason": reason.value,
            "artifact_type": self.config.artifact_type,
            "compressed": self.config.use_compression,
            "monitored_metric": self.config.model_checkpoint_monitor_metric,
            "training_duration_minutes": (
                (time.time() - training_start_time) / 60.0
                if training_start_time else None
            )
        }

    def resolve_epoch_step(
        self,
        filepath: str,
        trainer: "pl.Trainer"
    ) -> Tuple[int, int]:
        """
        Extract or infer epoch and step from checkpoint path.

        Args:
            filepath: Path to checkpoint file
            trainer: Lightning trainer

        Returns:
            Tuple of (epoch, step)
        """
        path = Path(filepath)
        filename = path.stem

        epoch, step = trainer.current_epoch, trainer.global_step

        # Try to extract from filename patterns
        if "epoch" in filename and "step" in filename:
            epoch_match = re.search(r'epoch[=_]?(\d+)', filename)
            step_match = re.search(r'step[=_]?(\d+)', filename)

            if epoch_match:
                epoch = int(epoch_match.group(1))
            if step_match:
                step = int(step_match.group(1))

        return epoch, step

    def prepare_upload_path(self, filepath: str) -> str:
        """
        Prepare file for upload, potentially compressing it.

        Args:
            filepath: Original checkpoint path

        Returns:
            Path to upload (may be compressed copy)
        """
        if not self.config.use_compression:
            return filepath

        try:
            compressed_path = tempfile.mktemp(suffix='.ckpt.gz')
            with open(filepath, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    f_out.write(f_in.read())

            original_size = Path(filepath).stat().st_size
            compressed_size = Path(compressed_path).stat().st_size
            ratio = (1 - compressed_size / original_size) * 100
            self.logger.info(f"Compressed checkpoint: {ratio:.1f}% size reduction")

            return compressed_path
        except Exception as e:
            self.logger.warning(f"Compression failed, using original: {e}")
            return filepath

    def get_best_score(self, model_checkpoint) -> Optional[float]:
        """
        Get the best model score from ModelCheckpoint.

        Args:
            model_checkpoint: ModelCheckpoint callback reference

        Returns:
            Best score if available
        """
        if model_checkpoint and model_checkpoint.best_model_score:
            score = model_checkpoint.best_model_score
            return score.item() if isinstance(score, torch.Tensor) else float(score)
        return None

    def is_duplicate_upload(self, path: str, uploaded: List[Dict]) -> bool:
        """
        Check if this would be a duplicate upload.

        Args:
            path: Checkpoint path
            uploaded: List of already uploaded artifacts

        Returns:
            True if this is a duplicate
        """
        for artifact in uploaded:
            if artifact.get('filepath') == path:
                return True
        return False

    def create_emergency_checkpoint(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        reason: str,
        model_checkpoint_dirpath: Optional[str],
        wandb_run_id: Optional[str]
    ) -> Optional[str]:
        """
        Create an emergency checkpoint with current state.

        Args:
            trainer: Lightning trainer
            pl_module: Lightning module
            reason: Reason for emergency checkpoint
            model_checkpoint_dirpath: ModelCheckpoint directory path
            wandb_run_id: W&B run ID for metadata

        Returns:
            Path to created checkpoint, or None if failed
        """
        try:
            # Generate filename
            filename = f"emergency-{reason}-epoch={trainer.current_epoch}-step={trainer.global_step}.ckpt"

            # Determine path
            if model_checkpoint_dirpath:
                checkpoint_path = Path(model_checkpoint_dirpath) / filename
            else:
                checkpoint_dir = Path(trainer.default_root_dir) / "checkpoints"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = checkpoint_dir / filename

            # Save comprehensive checkpoint
            from ...utils.checkpoint.checkpoint_utils import save_comprehensive_checkpoint
            save_comprehensive_checkpoint(
                trainer, pl_module,
                str(checkpoint_path),
                reason=f"emergency_{reason}",
                extra_metadata={'wandb_run_id': wandb_run_id}
            )

            if checkpoint_path.exists() and checkpoint_path.stat().st_size > 0:
                self.logger.info(f"Created emergency checkpoint: {checkpoint_path}")
                return str(checkpoint_path)

        except Exception as e:
            self.logger.error(f"Failed to create emergency checkpoint: {e}")

        return None

    def get_periodic_checkpoints_to_upload(
        self,
        model_checkpoint,
        epoch_count: int,
        validation_count: int
    ) -> List[Tuple[str, str]]:
        """
        Determine which checkpoints to upload for periodic uploads.

        Args:
            model_checkpoint: ModelCheckpoint callback reference
            epoch_count: Current epoch count
            validation_count: Current validation count

        Returns:
            List of (checkpoint_path, checkpoint_type) tuples
        """
        checkpoints = []

        # For periodic uploads, use timestamped names
        if model_checkpoint and model_checkpoint.last_model_path:
            checkpoints.append((
                model_checkpoint.last_model_path,
                f"epoch_{epoch_count}_step_{validation_count}"
            ))

        return checkpoints
