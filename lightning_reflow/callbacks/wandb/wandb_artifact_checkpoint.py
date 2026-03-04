"""
WandB Artifact Checkpoint Callback for Lightning Reflow (Refactored).

This is a simplified and more maintainable version of WandbArtifactCheckpoint.
Key improvements:
1. Configuration extracted to dataclass
2. Upload logic consolidated
3. State management simplified
4. Cleaner separation of concerns
"""

import lightning.pytorch as pl
import torch
import wandb
import time
import gzip
import tempfile
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from ...utils.logging.logging_config import get_logger
from ...utils.wandb.wandb_artifact_manager import WandbArtifactManager
from ...utils.checkpoint.wandb_artifact_state import WandbArtifactState
from .checkpoint_upload_helper import CheckpointUploadHelper


class UploadReason(Enum):
    """Reasons for uploading checkpoints."""
    NORMAL_COMPLETION = "normal_completion"
    EXCEPTION = "exception"
    TEARDOWN = "teardown"
    PAUSE_REQUESTED = "pause_requested"
    PERIODIC_VALIDATION = "periodic_validation"
    PERIODIC_EPOCH = "periodic_epoch"
    PERIODIC_HOURS = "periodic_hours"


@dataclass
class WandbCheckpointConfig:
    """Configuration for WandbArtifactCheckpoint."""
    # Basic upload settings
    upload_best_model: bool = True
    upload_last_model: bool = True
    upload_all_checkpoints: bool = False
    
    # Monitoring settings
    model_checkpoint_monitor_metric: Optional[str] = None
    monitor_pause_checkpoints: bool = True
    
    # Timing and frequency
    min_training_minutes: float = 5.0
    min_training_minutes_for_exceptions: float = 0.0
    upload_every_n_validation: Optional[int] = None
    upload_every_n_epoch: Optional[int] = None
    upload_periodic_checkpoints: bool = False
    upload_every_n_hours: Optional[float] = None
    
    # Storage optimization
    use_compression: bool = True
    upload_best_last_only_at_end: bool = False
    periodic_upload_pattern: str = "timestamped"  # "timestamped", "best", "last", "both"
    keep_n_versions: Optional[int] = None  # Delete old artifact versions, keep N most recent
    
    # Emergency handling
    create_emergency_checkpoints: bool = True
    cleanup_emergency_checkpoints: bool = True
    upload_on_exception: bool = True
    upload_on_teardown: bool = True
    
    # Logging
    wandb_verbose: bool = True
    artifact_type: str = "model"


@dataclass
class UploadState:
    """Tracks upload state for resume and deduplication."""
    has_uploaded: bool = False
    training_start_time: Optional[float] = None
    uploaded_pause_checkpoints: Dict[str, str] = field(default_factory=dict)
    validation_count: int = 0
    epoch_count: int = 0
    last_uploaded_epoch: int = -1
    last_uploaded_validation_step: int = -1
    last_checkpoint_epoch: int = -1
    last_checkpoint_step: int = -1
    checkpoint_was_loaded: bool = False
    next_checkpoint_epoch: Optional[int] = None
    next_checkpoint_step: Optional[int] = None
    last_periodic_upload_time: Optional[float] = None


class WandbArtifactCheckpoint(pl.Callback):
    """
    Simplified PyTorch Lightning callback for uploading checkpoints as W&B artifacts.
    
    This refactored version maintains all functionality while being more maintainable:
    - Configuration is centralized in a dataclass
    - Upload logic is consolidated into fewer methods
    - State management is cleaner
    - Helper methods are better organized
    """
    
    def __init__(self, **kwargs):
        """Initialize with configuration."""
        super().__init__()
        self.config = WandbCheckpointConfig(**kwargs)
        self.state = UploadState()
        self.logger = get_logger(__name__)

        # References to other components
        self._model_checkpoint_ref: Optional[ModelCheckpoint] = None
        self._wandb_run_ref: Optional[wandb.sdk.wandb_run.Run] = None
        self._wandb_manager = WandbArtifactManager(
            verbose=self.config.wandb_verbose,
            keep_n_versions=self.config.keep_n_versions,
        )
        self._upload_helper = CheckpointUploadHelper(self.config)

        # Register for state persistence
        self._register_for_state_persistence()
    
    
    # ============= Core Lightning Hooks =============
    
    @rank_zero_only
    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Initialize at training start."""
        self.state.training_start_time = time.time()
        self._wandb_run_ref = self._get_wandb_run(trainer)
        self._model_checkpoint_ref = self._find_model_checkpoint(trainer)
        
        if self._model_checkpoint_ref and self.config.wandb_verbose:
            self._log_configuration()
    
    @rank_zero_only
    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Upload checkpoints at training end."""
        if self._should_skip_upload(trainer, UploadReason.NORMAL_COMPLETION):
            return
        
        self._upload_checkpoints(trainer, pl_module, UploadReason.NORMAL_COMPLETION)
    
    @rank_zero_only
    def on_exception(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", exception: BaseException) -> None:
        """Handle exceptions by uploading emergency checkpoints."""
        if self.config.upload_on_exception:
            # TrialPruned is normal HPO behavior, not a crash
            exception_name = type(exception).__name__
            if exception_name == "TrialPruned":
                self.logger.info(f"Trial pruned (normal HPO behavior)")
            else:
                self.logger.warning(f"Training crashed with {exception_name}: {exception}")
            self._upload_checkpoints(trainer, pl_module, UploadReason.EXCEPTION)
    
    @rank_zero_only
    def teardown(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        """Clean teardown with optional checkpoint upload."""
        if self.config.upload_on_teardown and stage in ["fit", "train"]:
            if not self._should_skip_upload(trainer, UploadReason.TEARDOWN):
                self._upload_checkpoints(trainer, pl_module, UploadReason.TEARDOWN)
    
    @rank_zero_only
    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Handle periodic uploads on validation."""
        if not self._should_upload_periodic_validation(trainer):
            return
        
        # Skip uploads if we're pausing - best/latest should only upload at actual training end
        if self._is_pause_context(trainer):
            self._log_verbose(trainer, "Skipping periodic upload during pause - will upload pause checkpoint instead")
            return
        
        self._upload_periodic_checkpoints(trainer, pl_module, UploadReason.PERIODIC_VALIDATION)
    
    @rank_zero_only
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Handle periodic uploads on epoch end."""
        if not self._should_upload_periodic_epoch(trainer):
            return
        
        # Skip uploads if we're pausing - best/latest should only upload at actual training end
        if self._is_pause_context(trainer):
            self._log_verbose(trainer, "Skipping periodic upload during pause - will upload pause checkpoint instead")
            return
        
        self._upload_periodic_checkpoints(trainer, pl_module, UploadReason.PERIODIC_EPOCH)

    @rank_zero_only
    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule",
                           outputs, batch, batch_idx) -> None:
        """Handle time-based periodic checkpoint uploads."""
        if self.config.upload_every_n_hours is None:
            return

        if not self._should_upload_periodic_hours():
            return

        if self._is_pause_context(trainer):
            return

        self._upload_time_based_checkpoint(trainer, pl_module)

    # ============= Main Upload Logic (Consolidated) =============
    
    def _upload_checkpoints(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule",
                          reason: UploadReason) -> None:
        """
        Main checkpoint upload method - consolidates all upload logic.
        """
        # Check prerequisites
        if not self._can_upload(trainer, reason):
            return

        uploaded_artifacts = []
        emergency_checkpoint_path = None

        # Handle different upload patterns
        if self.config.upload_all_checkpoints:
            uploaded_artifacts = self._upload_all_checkpoints(trainer, pl_module, reason)
        else:
            # Upload best model if configured
            if self.config.upload_best_model and self._model_checkpoint_ref.best_model_path:
                artifact = self._upload_checkpoint(
                    trainer, pl_module,
                    self._model_checkpoint_ref.best_model_path,
                    "best",
                    self._get_best_score(),
                    reason
                )
                if artifact:
                    uploaded_artifacts.append(artifact)

            # Upload last/latest model
            if self.config.upload_last_model:
                last_path = self._get_last_checkpoint_path(trainer, pl_module, reason)
                # Track if this is an emergency checkpoint for later cleanup
                if last_path and "emergency-" in str(last_path):
                    emergency_checkpoint_path = last_path
                if last_path and not self._is_duplicate_upload(last_path, uploaded_artifacts):
                    artifact = self._upload_checkpoint(
                        trainer, pl_module,
                        last_path,
                        "latest",
                        self._get_current_score(trainer),
                        reason
                    )
                    if artifact:
                        uploaded_artifacts.append(artifact)

        # Log results
        if uploaded_artifacts:
            self._log_upload_summary(uploaded_artifacts, reason)
            self.state.has_uploaded = True

            # Cleanup emergency checkpoint after successful upload
            if emergency_checkpoint_path and self.config.cleanup_emergency_checkpoints:
                self._cleanup_emergency_checkpoint(emergency_checkpoint_path)
    
    def _upload_periodic_checkpoints(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule",
                                    reason: UploadReason) -> None:
        """Handle periodic checkpoint uploads with storage optimization."""
        if not self._model_checkpoint_ref:
            return
        
        checkpoints_to_upload = self._get_periodic_checkpoints_to_upload()
        
        for checkpoint_path, checkpoint_type in checkpoints_to_upload:
            if Path(checkpoint_path).exists():
                self._upload_checkpoint(
                    trainer, pl_module,
                    checkpoint_path,
                    checkpoint_type,
                    self._get_current_score(trainer),
                    reason
                )
        
        # Update state
        if reason == UploadReason.PERIODIC_VALIDATION:
            self.state.last_uploaded_validation_step = trainer.global_step
            self._update_next_validation_checkpoint(trainer)
        elif reason == UploadReason.PERIODIC_EPOCH:
            self.state.last_uploaded_epoch = trainer.current_epoch
            self._update_next_epoch_checkpoint(trainer)
    
    def upload_pause_checkpoint(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule",
                               checkpoint_path: str) -> Optional[str]:
        """Upload a pause checkpoint - called by PauseCallback."""
        if not self._can_upload_pause_checkpoint(checkpoint_path):
            return None
        
        # Check for cached upload
        if checkpoint_path in self.state.uploaded_pause_checkpoints:
            self._log_verbose(trainer, f"Pause checkpoint already uploaded: {checkpoint_path}")
            return self.state.uploaded_pause_checkpoints[checkpoint_path]
        
        # Upload the checkpoint
        artifact = self._upload_checkpoint(
            trainer, pl_module,
            checkpoint_path,
            "pause",
            None,  # Pause checkpoints don't have scores
            UploadReason.PAUSE_REQUESTED
        )
        
        if artifact:
            self.state.uploaded_pause_checkpoints[checkpoint_path] = artifact['artifact']
            return artifact['artifact']
        
        return None
    
    # ============= Helper Methods (Simplified) =============
    
    def _can_upload(self, trainer: "pl.Trainer", reason: UploadReason) -> bool:
        """Check if uploads are possible and needed."""
        # Already uploaded check
        if self.state.has_uploaded and reason != UploadReason.PAUSE_REQUESTED:
            self._log_verbose(trainer, f"Checkpoints already uploaded, skipping {reason.value}")
            return False
        
        # Check basic requirements
        if not self._wandb_run_ref or not self._model_checkpoint_ref:
            self._log_verbose(trainer, f"Missing requirements for upload: {reason.value}")
            return False
        
        # Check training duration
        if not self._has_sufficient_training_time(reason):
            return False
        
        return True
    
    def _can_upload_pause_checkpoint(self, checkpoint_path: str) -> bool:
        """Check if pause checkpoint can be uploaded."""
        if not self.config.monitor_pause_checkpoints:
            self._log_verbose(None, "Pause checkpoint monitoring disabled")
            return False
        
        if not self._wandb_run_ref:
            self._log_verbose(None, "No W&B run available for pause checkpoint upload")
            return False
        
        if not Path(checkpoint_path).exists():
            self._log_verbose(None, f"Pause checkpoint not found: {checkpoint_path}")
            return False
        
        return True
    
    def _should_skip_upload(self, trainer: "pl.Trainer", reason: UploadReason) -> bool:
        """Determine if upload should be skipped."""
        # Check if in pause context
        if self._is_pause_context(trainer):
            self._log_verbose(trainer, "Skipping upload - training paused")
            return True
        
        # Check if training is complete
        if reason == UploadReason.NORMAL_COMPLETION and not self._is_training_complete(trainer):
            self._log_verbose(trainer, "Skipping upload - training not complete")
            return True
        
        return False
    
    def _has_sufficient_training_time(self, reason: UploadReason) -> bool:
        """Check if training has run long enough for uploads."""
        if self.state.training_start_time is None:
            return True
        
        duration_minutes = (time.time() - self.state.training_start_time) / 60.0
        
        if reason == UploadReason.EXCEPTION:
            threshold = self.config.min_training_minutes_for_exceptions
        else:
            threshold = self.config.min_training_minutes
        
        return duration_minutes >= threshold
    
    def _get_last_checkpoint_path(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule",
                                 reason: UploadReason) -> Optional[str]:
        """Get the path to the last/latest checkpoint."""
        # Only create emergency checkpoints for actual emergencies (exceptions),
        # not for normal teardown - this prevents unnecessary disk usage
        if reason == UploadReason.EXCEPTION and self.config.create_emergency_checkpoints:
            emergency_path = self._create_emergency_checkpoint(trainer, pl_module, reason.value)
            if emergency_path:
                return emergency_path

        # Fall back to last model path
        return self._model_checkpoint_ref.last_model_path if self._model_checkpoint_ref else None
    
    def _upload_checkpoint(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule",
                         filepath: str, ckpt_type: str, score: Optional[float],
                         reason: UploadReason) -> Optional[Dict[str, Any]]:
        """Upload a single checkpoint and return artifact info."""
        if not Path(filepath).exists():
            return None

        # Use helper for preparation
        aliases = self._upload_helper.create_aliases(ckpt_type, reason)
        epoch, step = self._upload_helper.resolve_epoch_step(filepath, trainer)
        upload_path = self._upload_helper.prepare_upload_path(filepath)
        extra_metadata = self._upload_helper.get_extra_metadata(reason, self.state.training_start_time)

        try:
            # Upload via manager
            artifact_path = self._wandb_manager.upload_checkpoint_artifact(
                trainer=trainer,
                pl_module=pl_module,
                filepath=upload_path,
                ckpt_type=ckpt_type,
                aliases=aliases,
                score=score,
                epoch=epoch,
                step=step,
                wandb_run=self._wandb_run_ref,
                extra_metadata=extra_metadata
            )

            if artifact_path:
                return {
                    "type": ckpt_type,
                    "artifact": artifact_path,
                    "score": score,
                    "aliases": aliases,
                    "reason": reason.value
                }
        finally:
            # Cleanup temporary files
            if upload_path != filepath and Path(upload_path).exists():
                Path(upload_path).unlink()

        return None
    
    def _get_periodic_checkpoints_to_upload(self) -> List[Tuple[str, str]]:
        """Determine which checkpoints to upload for periodic uploads."""
        return self._upload_helper.get_periodic_checkpoints_to_upload(
            self._model_checkpoint_ref,
            self.state.epoch_count,
            self.state.validation_count
        )
    
    # ============= Utility Methods =============
    
    def _find_model_checkpoint(self, trainer: "pl.Trainer") -> Optional[ModelCheckpoint]:
        """Find ModelCheckpoint callback in trainer."""
        for callback in trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                return callback
        return None
    
    def _get_wandb_run(self, trainer: "pl.Trainer") -> Optional[wandb.sdk.wandb_run.Run]:
        """Get W&B run from trainer."""
        return self._wandb_manager.get_wandb_run(trainer)
    
    def _is_pause_context(self, trainer: "pl.Trainer") -> bool:
        """Check if training is paused."""
        for callback in trainer.callbacks:
            if (hasattr(callback, 'is_pausing') and 
                hasattr(callback, 'is_pause_scheduled') and
                callback.__class__.__name__ == 'PauseCallback'):
                return callback.is_pausing() or callback.is_pause_scheduled()
        return False
    
    def _is_training_complete(self, trainer: "pl.Trainer") -> bool:
        """Check if training is complete."""
        # Check max steps
        if trainer.max_steps and trainer.max_steps > 0:
            if trainer.global_step >= trainer.max_steps - 1:
                return True
        
        # Check max epochs
        if trainer.max_epochs and trainer.max_epochs > 0:
            if trainer.current_epoch >= trainer.max_epochs - 1:
                return True
        
        return False
    
    def _should_upload_periodic_validation(self, trainer: "pl.Trainer") -> bool:
        """Check if periodic validation upload is due."""
        if not self.config.upload_every_n_validation and not self.config.upload_periodic_checkpoints:
            return False
        
        self.state.validation_count += 1
        
        if self.config.upload_every_n_validation:
            if self.state.next_checkpoint_step and trainer.global_step >= self.state.next_checkpoint_step:
                return True
            elif self.state.validation_count % self.config.upload_every_n_validation == 0:
                return True
        
        return self.config.upload_periodic_checkpoints
    
    def _should_upload_periodic_epoch(self, trainer: "pl.Trainer") -> bool:
        """Check if periodic epoch upload is due."""
        if not self.config.upload_every_n_epoch:
            return False
        
        self.state.epoch_count += 1
        
        if self.state.next_checkpoint_epoch and trainer.current_epoch >= self.state.next_checkpoint_epoch:
            return True
        elif self.state.epoch_count % self.config.upload_every_n_epoch == 0:
            return True
        
        return False

    def _should_upload_periodic_hours(self) -> bool:
        """Check if enough wall-clock time has elapsed for a periodic upload."""
        if self.config.upload_every_n_hours is None:
            return False

        reference = self.state.last_periodic_upload_time or self.state.training_start_time
        if reference is None:
            return False

        elapsed_hours = (time.time() - reference) / 3600.0
        return elapsed_hours >= self.config.upload_every_n_hours

    def _upload_time_based_checkpoint(self, trainer: "pl.Trainer",
                                      pl_module: "pl.LightningModule") -> None:
        """Upload a time-based periodic checkpoint as ckpt_type='latest'."""
        if not self._wandb_run_ref or not self._model_checkpoint_ref:
            return

        last_path = self._model_checkpoint_ref.last_model_path
        if not last_path or not Path(last_path).exists():
            return

        score = self._get_current_score(trainer)
        artifact = self._upload_checkpoint(
            trainer, pl_module, last_path, "latest", score, UploadReason.PERIODIC_HOURS
        )

        if artifact:
            self.state.last_periodic_upload_time = time.time()
            self._log_verbose(
                trainer,
                f"Time-based periodic upload complete "
                f"(every {self.config.upload_every_n_hours}h): {artifact['artifact']}"
            )

    def _create_emergency_checkpoint(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule",
                                    reason: str) -> Optional[str]:
        """Create an emergency checkpoint with current state."""
        dirpath = self._model_checkpoint_ref.dirpath if self._model_checkpoint_ref else None
        wandb_run_id = self._wandb_run_ref.id if self._wandb_run_ref else None

        result = self._upload_helper.create_emergency_checkpoint(
            trainer, pl_module, reason, dirpath, wandb_run_id
        )

        if result:
            self._log_verbose(trainer, f"Created emergency checkpoint: {result}")

        return result

    def _cleanup_emergency_checkpoint(self, checkpoint_path: str) -> None:
        """Clean up emergency checkpoint after successful upload."""
        try:
            path = Path(checkpoint_path)
            if path.exists() and "emergency-" in path.name:
                path.unlink()
                self.logger.info(f"Cleaned up emergency checkpoint: {checkpoint_path}")
        except Exception as e:
            self.logger.warning(f"Failed to cleanup emergency checkpoint: {e}")

    # ============= Helper Methods =============
    
    def _log_verbose(self, trainer: Optional["pl.Trainer"], message: str) -> None:
        """Log message if verbose mode is enabled."""
        if self.config.wandb_verbose:
            if trainer and hasattr(trainer, 'is_global_zero') and not trainer.is_global_zero:
                return
            self.logger.info(message)
    
    def _log_configuration(self) -> None:
        """Log callback configuration."""
        upload_types = []
        if self.config.upload_best_model:
            upload_types.append("best")
        if self.config.upload_last_model:
            upload_types.append("latest")
        if self.config.upload_all_checkpoints:
            upload_types.append("all")
        
        self.logger.info(f"WandbArtifactCheckpoint configured to upload: {', '.join(upload_types)}")
        
        if self.config.upload_every_n_epoch:
            self.logger.info(f"  - Periodic uploads every {self.config.upload_every_n_epoch} epochs")
        if self.config.upload_every_n_validation:
            self.logger.info(f"  - Periodic uploads every {self.config.upload_every_n_validation} validations")
        if self.config.upload_every_n_hours:
            self.logger.info(f"  - Time-based uploads every {self.config.upload_every_n_hours} hours")
        if self.config.use_compression:
            self.logger.info(f"  - Compression enabled")
    
    def _log_upload_summary(self, artifacts: List[Dict[str, Any]], reason: UploadReason) -> None:
        """Log summary of uploaded artifacts."""
        self.logger.info(f"Successfully uploaded {len(artifacts)} artifacts ({reason.value})")
        for artifact in artifacts:
            self.logger.info(f"  - {artifact['type']}: {artifact['artifact']}")

    def _get_best_score(self) -> Optional[float]:
        """Get the best model score."""
        return self._upload_helper.get_best_score(self._model_checkpoint_ref)
    
    def _get_current_score(self, trainer: "pl.Trainer") -> Optional[float]:
        """Get current score from trainer."""
        return self._wandb_manager.extract_score_from_trainer(
            trainer, 
            self.config.model_checkpoint_monitor_metric
        )
    
    def _is_duplicate_upload(self, path: str, uploaded: List[Dict]) -> bool:
        """Check if this would be a duplicate upload."""
        return self._upload_helper.is_duplicate_upload(path, uploaded)
    
    def _upload_all_checkpoints(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule",
                               reason: UploadReason) -> List[Dict[str, Any]]:
        """Upload all available checkpoints."""
        uploaded = []
        
        if not self._model_checkpoint_ref or not self._model_checkpoint_ref.dirpath:
            return uploaded
        
        checkpoint_dir = Path(self._model_checkpoint_ref.dirpath)
        if not checkpoint_dir.exists():
            return uploaded
        
        for ckpt_file in checkpoint_dir.glob("*.ckpt"):
            ckpt_path = str(ckpt_file)
            
            # Determine checkpoint type
            if ckpt_path == self._model_checkpoint_ref.best_model_path:
                ckpt_type = "best"
            elif ckpt_path == self._model_checkpoint_ref.last_model_path:
                ckpt_type = "latest"
            else:
                ckpt_type = f"checkpoint-{ckpt_file.stem}"
            
            # Upload
            artifact = self._upload_checkpoint(
                trainer, pl_module,
                ckpt_path,
                ckpt_type,
                self._get_current_score(trainer),
                reason
            )
            
            if artifact:
                uploaded.append(artifact)
        
        return uploaded
    
    def _update_next_validation_checkpoint(self, trainer: "pl.Trainer") -> None:
        """Update next validation checkpoint step."""
        if self.config.upload_every_n_validation:
            # Calculate next checkpoint step
            val_interval = self._get_validation_interval_steps(trainer) or 1
            self.state.next_checkpoint_step = trainer.global_step + (
                self.config.upload_every_n_validation * val_interval
            )
    
    def _update_next_epoch_checkpoint(self, trainer: "pl.Trainer") -> None:
        """Update next epoch checkpoint."""
        if self.config.upload_every_n_epoch:
            self.state.next_checkpoint_epoch = trainer.current_epoch + self.config.upload_every_n_epoch
    
    def _get_validation_interval_steps(self, trainer: "pl.Trainer") -> Optional[int]:
        """Get validation interval in steps."""
        if hasattr(trainer, 'val_check_interval'):
            interval = trainer.val_check_interval
            if isinstance(interval, int):
                return interval
            elif isinstance(interval, float) and interval > 1.0:
                return int(interval)
        return None
    
    def _register_for_state_persistence(self) -> None:
        """Register for checkpoint state persistence."""
        try:
            from ...utils.checkpoint.manager_state import register_manager

            # Use the extracted WandbArtifactState class
            self._state_manager = WandbArtifactState(self)
            register_manager(self._state_manager)
            self.logger.info("🔗 WandbArtifactCheckpoint registered for checkpoint persistence")

        except ImportError:
            self.logger.debug("Manager state system not available")