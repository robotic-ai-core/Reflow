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


class UploadReason(Enum):
    """Reasons for uploading checkpoints."""
    NORMAL_COMPLETION = "normal_completion"
    EXCEPTION = "exception"
    TEARDOWN = "teardown"
    PAUSE_REQUESTED = "pause_requested"
    PERIODIC_VALIDATION = "periodic_validation"
    PERIODIC_EPOCH = "periodic_epoch"


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
    
    # Storage optimization
    use_compression: bool = True
    upload_best_last_only_at_end: bool = False
    periodic_upload_pattern: str = "timestamped"  # "timestamped", "best", "last", "both"
    
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
        self._wandb_manager = WandbArtifactManager(verbose=self.config.wandb_verbose)
        
        # Register for state persistence
        self._register_for_state_persistence()
    
    # ============= Backward Compatibility Properties =============
    # These properties provide compatibility with the old attribute-based interface
    
    @property
    def upload_best_model(self) -> bool:
        return self.config.upload_best_model
    
    @property
    def upload_last_model(self) -> bool:
        return self.config.upload_last_model
    
    @property
    def upload_all_checkpoints(self) -> bool:
        return self.config.upload_all_checkpoints
    
    @property
    def upload_every_n_epoch(self) -> Optional[int]:
        return self.config.upload_every_n_epoch
    
    @property
    def upload_every_n_validation(self) -> Optional[int]:
        return self.config.upload_every_n_validation
    
    @property
    def min_training_minutes(self) -> float:
        return self.config.min_training_minutes
    
    @property
    def use_compression(self) -> bool:
        return self.config.use_compression
    
    @property
    def upload_best_last_only_at_end(self) -> bool:
        return self.config.upload_best_last_only_at_end
    
    @property
    def periodic_upload_pattern(self) -> str:
        return self.config.periodic_upload_pattern
    
    @property
    def monitor_pause_checkpoints(self) -> bool:
        return self.config.monitor_pause_checkpoints
    
    @property
    def create_emergency_checkpoints(self) -> bool:
        return self.config.create_emergency_checkpoints
    
    @property
    def wandb_verbose(self) -> bool:
        return self.config.wandb_verbose
    
    @property
    def _training_start_time(self) -> Optional[float]:
        """Backward compatibility for old attribute name."""
        return self.state.training_start_time
    
    @_training_start_time.setter
    def _training_start_time(self, value: Optional[float]) -> None:
        """Backward compatibility setter."""
        self.state.training_start_time = value
    
    @property
    def _validation_count(self) -> int:
        """Backward compatibility for old attribute name."""
        return self.state.validation_count
    
    @_validation_count.setter
    def _validation_count(self, value: int) -> None:
        """Backward compatibility setter."""
        self.state.validation_count = value
    
    @property
    def _epoch_count(self) -> int:
        """Backward compatibility for old attribute name."""
        return self.state.epoch_count
    
    @_epoch_count.setter
    def _epoch_count(self, value: int) -> None:
        """Backward compatibility setter."""
        self.state.epoch_count = value
    
    # Backward compatibility methods
    def _should_upload_epoch(self, trainer: "pl.Trainer") -> bool:
        """Backward compatibility wrapper."""
        return self._should_upload_periodic_epoch(trainer)
    
    def _should_upload_validation(self, trainer: "pl.Trainer") -> bool:
        """Backward compatibility wrapper."""
        return self._should_upload_periodic_validation(trainer)
    
    def _save_comprehensive_checkpoint(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule",
                                      filepath: str, reason: str) -> bool:
        """Backward compatibility wrapper for emergency checkpoint creation."""
        try:
            from ...utils.checkpoint.checkpoint_utils import save_comprehensive_checkpoint
            save_comprehensive_checkpoint(
                trainer, pl_module,
                filepath,
                reason=reason,
                extra_metadata={'wandb_run_id': self._wandb_run_ref.id if self._wandb_run_ref else None}
            )
            return Path(filepath).exists() and Path(filepath).stat().st_size > 0
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            return False
    
    def _upload_single_checkpoint(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule",
                                 filepath: str, ckpt_type: str, score: Optional[float] = None,
                                 reason: str = "manual") -> Optional[str]:
        """Backward compatibility wrapper for single checkpoint upload."""
        upload_reason = UploadReason.NORMAL_COMPLETION
        if reason == "exception":
            upload_reason = UploadReason.EXCEPTION
        elif reason == "teardown":
            upload_reason = UploadReason.TEARDOWN
        
        result = self._upload_checkpoint(trainer, pl_module, filepath, ckpt_type, score, upload_reason)
        return result['artifact'] if result else None
    
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
            self.logger.warning(f"Training crashed with {type(exception).__name__}: {exception}")
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
        # For exceptions/teardown, potentially create emergency checkpoint
        if reason in [UploadReason.EXCEPTION, UploadReason.TEARDOWN] and self.config.create_emergency_checkpoints:
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
        
        # Prepare aliases
        aliases = self._create_aliases(ckpt_type, reason)
        
        # Get epoch and step
        epoch, step = self._resolve_epoch_step(filepath, trainer)
        
        # Handle compression if enabled
        upload_path = self._prepare_upload_path(filepath)
        
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
                extra_metadata=self._get_extra_metadata(reason)
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
        checkpoints = []
        
        # For periodic uploads, we should upload timestamped checkpoints
        # Best/latest should only be uploaded at the actual end of training
        if self._model_checkpoint_ref.last_model_path:
            # Always use timestamped names for periodic uploads
            checkpoints.append((
                self._model_checkpoint_ref.last_model_path,
                f"epoch_{self.state.epoch_count}_step_{self.state.validation_count}"
            ))
        
        return checkpoints
    
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
    
    def _create_emergency_checkpoint(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule",
                                    reason: str) -> Optional[str]:
        """Create an emergency checkpoint with current state."""
        try:
            # Generate filename
            filename = f"emergency-{reason}-epoch={trainer.current_epoch}-step={trainer.global_step}.ckpt"
            
            # Determine path
            if self._model_checkpoint_ref and self._model_checkpoint_ref.dirpath:
                checkpoint_path = Path(self._model_checkpoint_ref.dirpath) / filename
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
                extra_metadata={'wandb_run_id': self._wandb_run_ref.id if self._wandb_run_ref else None}
            )
            
            if checkpoint_path.exists() and checkpoint_path.stat().st_size > 0:
                self._log_verbose(trainer, f"Created emergency checkpoint: {checkpoint_path}")
                return str(checkpoint_path)
                
        except Exception as e:
            self.logger.error(f"Failed to create emergency checkpoint: {e}")
        
        return None
    
    def _prepare_upload_path(self, filepath: str) -> str:
        """Prepare file for upload, potentially compressing it."""
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
    
    # ============= State Persistence =============
    
    @rank_zero_only
    def on_save_checkpoint(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", 
                          checkpoint: Dict[str, Any]) -> None:
        """Save callback state to checkpoint."""
        # Save in new format
        checkpoint['wandb_artifact_checkpoint_state'] = {
            'config': self.config.__dict__,
            'state': self.state.__dict__,
            'version': '2.0'
        }
        
        # Also save in old format for backward compatibility
        checkpoint['wandb_artifact_checkpoint'] = {
            'epoch_count': self.state.epoch_count,
            'validation_count': self.state.validation_count,
            'has_uploaded': self.state.has_uploaded,
            'uploaded_pause_checkpoints': self.state.uploaded_pause_checkpoints,
            'last_uploaded_epoch': self.state.last_uploaded_epoch,
            'last_uploaded_validation_step': self.state.last_uploaded_validation_step,
            'checkpoint_was_loaded': self.state.checkpoint_was_loaded,
            'next_checkpoint_epoch': self.state.next_checkpoint_epoch,
            'next_checkpoint_step': self.state.next_checkpoint_step
        }
    
    @rank_zero_only
    def on_load_checkpoint(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule",
                          checkpoint: Dict[str, Any]) -> None:
        """Restore callback state from checkpoint."""
        # Try new format first
        if 'wandb_artifact_checkpoint_state' in checkpoint:
            saved_state = checkpoint['wandb_artifact_checkpoint_state']
            
            # Restore state
            if 'state' in saved_state:
                for key, value in saved_state['state'].items():
                    if hasattr(self.state, key):
                        setattr(self.state, key, value)
            
            self.logger.info(f"Restored WandbArtifactCheckpoint state from checkpoint (v2.0)")
        # Fall back to old format for backward compatibility
        elif 'wandb_artifact_checkpoint' in checkpoint:
            old_state = checkpoint['wandb_artifact_checkpoint']
            
            # Map old state keys to new state object
            if 'epoch_count' in old_state:
                self.state.epoch_count = old_state['epoch_count']
            if 'validation_count' in old_state:
                self.state.validation_count = old_state['validation_count']
            if 'has_uploaded' in old_state:
                self.state.has_uploaded = old_state['has_uploaded']
            if 'uploaded_pause_checkpoints' in old_state:
                self.state.uploaded_pause_checkpoints = old_state['uploaded_pause_checkpoints']
            if 'last_uploaded_epoch' in old_state:
                self.state.last_uploaded_epoch = old_state['last_uploaded_epoch']
            if 'last_uploaded_validation_step' in old_state:
                self.state.last_uploaded_validation_step = old_state['last_uploaded_validation_step']
            if 'checkpoint_was_loaded' in old_state:
                self.state.checkpoint_was_loaded = old_state['checkpoint_was_loaded']
            if 'next_checkpoint_epoch' in old_state:
                self.state.next_checkpoint_epoch = old_state['next_checkpoint_epoch']
            if 'next_checkpoint_step' in old_state:
                self.state.next_checkpoint_step = old_state['next_checkpoint_step']
            
            self.logger.info(f"Restored WandbArtifactCheckpoint state from checkpoint (legacy format)")
        
        # Note: We don't restore config as it should come from current run
    
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
        if self.config.use_compression:
            self.logger.info(f"  - Compression enabled")
    
    def _log_upload_summary(self, artifacts: List[Dict[str, Any]], reason: UploadReason) -> None:
        """Log summary of uploaded artifacts."""
        self.logger.info(f"Successfully uploaded {len(artifacts)} artifacts ({reason.value})")
        for artifact in artifacts:
            self.logger.info(f"  - {artifact['type']}: {artifact['artifact']}")
    
    def _create_aliases(self, ckpt_type: str, reason: UploadReason) -> List[str]:
        """Create aliases for the artifact."""
        aliases = [ckpt_type]
        
        if reason == UploadReason.EXCEPTION:
            aliases.append("crash_recovery")
        elif reason == UploadReason.PAUSE_REQUESTED:
            aliases.append("pause")
        elif reason in [UploadReason.PERIODIC_VALIDATION, UploadReason.PERIODIC_EPOCH]:
            aliases.append("periodic")
        
        aliases.append("latest")  # Always mark as latest
        
        return aliases
    
    def _get_extra_metadata(self, reason: UploadReason) -> Dict[str, Any]:
        """Get extra metadata for the upload."""
        return {
            "upload_reason": reason.value,
            "artifact_type": self.config.artifact_type,
            "compressed": self.config.use_compression,
            "monitored_metric": self.config.model_checkpoint_monitor_metric,
            "training_duration_minutes": (
                (time.time() - self.state.training_start_time) / 60.0 
                if self.state.training_start_time else None
            )
        }
    
    def _resolve_epoch_step(self, filepath: str, trainer: "pl.Trainer") -> Tuple[int, int]:
        """Extract or infer epoch and step from checkpoint path."""
        # Try to parse from filename
        path = Path(filepath)
        filename = path.stem
        
        epoch, step = trainer.current_epoch, trainer.global_step
        
        # Try to extract from filename patterns
        if "epoch" in filename and "step" in filename:
            import re
            epoch_match = re.search(r'epoch[=_]?(\d+)', filename)
            step_match = re.search(r'step[=_]?(\d+)', filename)
            
            if epoch_match:
                epoch = int(epoch_match.group(1))
            if step_match:
                step = int(step_match.group(1))
        
        return epoch, step
    
    def _get_best_score(self) -> Optional[float]:
        """Get the best model score."""
        if self._model_checkpoint_ref and self._model_checkpoint_ref.best_model_score:
            score = self._model_checkpoint_ref.best_model_score
            return score.item() if isinstance(score, torch.Tensor) else float(score)
        return None
    
    def _get_current_score(self, trainer: "pl.Trainer") -> Optional[float]:
        """Get current score from trainer."""
        return self._wandb_manager.extract_score_from_trainer(
            trainer, 
            self.config.model_checkpoint_monitor_metric
        )
    
    def _is_duplicate_upload(self, path: str, uploaded: List[Dict]) -> bool:
        """Check if this would be a duplicate upload."""
        for artifact in uploaded:
            if artifact.get('filepath') == path:
                return True
        return False
    
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
            from ...utils.checkpoint.manager_state import register_manager, ManagerState
            
            class WandbArtifactState(ManagerState):
                def __init__(self, callback):
                    self.callback = callback
                
                @property
                def manager_name(self) -> str:
                    return "wandb_artifact_checkpoint"
                
                def capture_state(self) -> Dict[str, Any]:
                    return {
                        'config': self.callback.config.__dict__,
                        'state': self.callback.state.__dict__,
                        'version': '2.0'
                    }
                
                def restore_state(self, state: Dict[str, Any]) -> bool:
                    if 'state' in state:
                        for key, value in state['state'].items():
                            if hasattr(self.callback.state, key):
                                setattr(self.callback.state, key, value)
                    return True
                
                def validate_state(self, state: Dict[str, Any]) -> bool:
                    return isinstance(state, dict) and 'version' in state
            
            register_manager(WandbArtifactState(self))
            self.logger.info("🔗 WandbArtifactCheckpoint registered for checkpoint persistence")
            
        except ImportError:
            self.logger.debug("Manager state system not available")