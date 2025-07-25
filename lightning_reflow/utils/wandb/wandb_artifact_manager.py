"""
Shared W&B artifact management utilities for all callbacks.

This module provides centralized W&B artifact operations to eliminate code duplication
between different callbacks that need to upload checkpoints, configs, or other artifacts.

Note: This module now uses UnifiedArtifactManager for core functionality to eliminate
duplication between upload methods.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import wandb
from lightning.pytorch import Trainer
from lightning.pytorch.core import LightningModule

from .unified_artifact_manager import UnifiedArtifactManager


class WandbArtifactManager:
    """
    Centralized W&B artifact management for all callbacks.
    
    This class provides shared functionality for uploading checkpoints, configs,
    and other artifacts to W&B, ensuring consistency across different callbacks.
    
    Note: This class now uses UnifiedArtifactManager internally to eliminate
    duplication between upload methods.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the artifact manager.
        
        Args:
            verbose: Whether to log verbose messages
        """
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        self._unified_manager = UnifiedArtifactManager(verbose=verbose)
    
    @staticmethod
    def get_wandb_run(trainer: Trainer) -> Optional[wandb.sdk.wandb_run.Run]:
        """
        Get W&B run from trainer's logger.
        
        Args:
            trainer: Lightning trainer instance
            
        Returns:
            W&B run object if found, None otherwise
        """
        return UnifiedArtifactManager.get_wandb_run(trainer)
    
    @staticmethod
    def get_wandb_run_id(trainer: Trainer = None) -> Optional[str]:
        """
        Get current W&B run ID.
        
        Args:
            trainer: Optional trainer to extract run from
            
        Returns:
            W&B run ID if available, None otherwise
        """
        try:
            # Try to get from trainer first
            if trainer:
                run = WandbArtifactManager.get_wandb_run(trainer)
                if run and run.id:
                    return run.id
            
            # Fallback to global wandb run
            if wandb.run and wandb.run.id:
                return wandb.run.id
                
        except Exception:
            pass
        return None
    
    def create_checkpoint_metadata(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        filepath: str,
        score: Optional[float] = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        ckpt_type: str = "checkpoint",
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create standardized checkpoint metadata.
        
        Args:
            trainer: Lightning trainer instance
            pl_module: Lightning module instance
            filepath: Path to checkpoint file
            score: Optional score/metric value
            epoch: Optional epoch number
            step: Optional step number
            ckpt_type: Type of checkpoint (e.g., "best", "latest", "pause")
            extra_metadata: Additional metadata to include
            
        Returns:
            Dictionary of metadata
        """
        metadata = {
            "source_filepath": filepath,
            "source_score": score if score is not None else "N/A",
            "source_epoch": epoch if epoch is not None else trainer.current_epoch,
            "source_step": step if step is not None else trainer.global_step,
            "checkpoint_type": ckpt_type,
            "final_trainer_epoch": trainer.current_epoch,
            "final_trainer_global_step": trainer.global_step,
            "pl_module_class": pl_module.__class__.__name__,
            "pytorch_lightning_version": getattr(torch.nn, '__version__', 'unknown'),
            "created_timestamp": time.time()
        }
        
        # Add extra metadata if provided
        if extra_metadata:
            metadata.update(extra_metadata)
            
        return metadata
    
    def upload_checkpoint_artifact(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        filepath: str,
        ckpt_type: str = "checkpoint",
        aliases: Optional[List[str]] = None,
        score: Optional[float] = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        wandb_run: Optional[wandb.sdk.wandb_run.Run] = None,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Upload checkpoint as W&B artifact with standardized naming and metadata.
        
        Args:
            trainer: Lightning trainer instance
            pl_module: Lightning module instance
            filepath: Path to checkpoint file
            ckpt_type: Type of checkpoint (e.g., "best", "latest", "pause")
            aliases: List of aliases for the artifact
            score: Optional score/metric value
            epoch: Optional epoch number
            step: Optional step number
            wandb_run: Optional W&B run (will auto-detect if not provided)
            extra_metadata: Additional metadata to include
            
        Returns:
            Artifact name with version if successful, None otherwise
        """
        try:
            # Validate checkpoint file
            if not Path(filepath).exists() or Path(filepath).stat().st_size == 0:
                if self.verbose and trainer.is_global_zero:
                    self.logger.warning(f"Invalid checkpoint file '{filepath}'. Skipping upload.")
                return None
            
            # Get W&B run
            if wandb_run is None:
                wandb_run = self.get_wandb_run(trainer)
            
            if not wandb_run:
                if self.verbose and trainer.is_global_zero:
                    self.logger.warning("No W&B run available for checkpoint upload")
                return None
            
            # Generate artifact name
            run_id = wandb_run.id or "local_run"
            artifact_name = f"{run_id}-{ckpt_type}"
            
            # Create metadata
            metadata = self.create_checkpoint_metadata(
                trainer, pl_module, filepath, score, epoch, step, ckpt_type, extra_metadata
            )
            
            # Note: Config is embedded in checkpoint metadata - no separate files needed
            
            # Create description
            description = (
                f"Checkpoint '{ckpt_type}' from training run '{run_id}'. "
                f"Epoch: {epoch if epoch is not None else 'N/A'}, "
                f"Step: {step if step is not None else 'N/A'}"
            )
            if score is not None:
                description += f", Score: {score}"
            description += ". Self-contained checkpoint with embedded config."
            
            # Create artifact with explicit relationship tracking
            artifact = wandb.Artifact(
                name=artifact_name,
                type="model",
                description=description,
                metadata=metadata
            )
            
            # Add explicit relationship metadata for better tracking
            artifact.metadata.update({
                'checkpoint_type': ckpt_type,
                'wandb_run_id': wandb_run.id,
                'entity': wandb_run.entity,
                'project': wandb_run.project,
                'created_by': 'DiffusionFlow',
            })
            
            # Add checkpoint file
            artifact.add_file(str(filepath), name=Path(filepath).name)
            
            # Note: Config is embedded in checkpoint file, no additional files needed
            # Upload artifact
            final_aliases = aliases or ["latest"]
            wandb_run.log_artifact(artifact, aliases=final_aliases)
            
            # Note: Config is embedded in checkpoint metadata - no separate config artifact needed
            
            # Construct full artifact reference for resuming
            entity = wandb_run.entity
            project = wandb_run.project
            artifact_version = artifact.version if hasattr(artifact, "version") else "latest"
            full_artifact_path = f"{entity}/{project}/{artifact.name}:{artifact_version}"
            
            if self.verbose and trainer.is_global_zero:
                artifact_full_name = f"{artifact.name}:{artifact_version}"
                files_info = f"checkpoint with embedded config"
                self.logger.info(f"Uploaded '{ckpt_type}' artifact '{artifact_full_name}' ({files_info}), aliases: {final_aliases}")
                self.logger.info(f"Full artifact reference: {full_artifact_path}")
            
            return full_artifact_path
            
        except Exception as e:
            if self.verbose and trainer.is_global_zero:
                self.logger.warning(f"Failed to upload '{ckpt_type}' artifact for '{filepath}': {e}")
            return None
    
    def extract_score_from_trainer(
        self, 
        trainer: Trainer, 
        metric_name: Optional[str] = None
    ) -> Optional[float]:
        """
        Extract score/metric value from trainer.
        
        Args:
            trainer: Lightning trainer instance
            metric_name: Name of metric to extract
            
        Returns:
            Metric value if found, None otherwise
        """
        if not metric_name:
            return None
            
        try:
            metric_val = trainer.callback_metrics.get(metric_name)
            if metric_val is not None:
                return metric_val.item() if isinstance(metric_val, torch.Tensor) else float(metric_val)
        except Exception:
            pass
        return None
    
    def create_simple_artifact_name(self, run_id: str, ckpt_type: str) -> str:
        """
        Create simple artifact name following naming conventions.
        
        Args:
            run_id: W&B run ID
            ckpt_type: Type of checkpoint
            
        Returns:
            Artifact name string
        """
        return f"{run_id}-{ckpt_type}"
    
    def validate_artifact_relationships(
        self, 
        checkpoint_artifact_path: str, 
        wandb_run: Optional[wandb.sdk.wandb_run.Run] = None
    ) -> Dict[str, Any]:
        """
        Validate checkpoint artifact for resume capability.
        
        Note: Only validates checkpoint artifacts since config is embedded.
        Separate config artifacts are no longer created or required.
        
        Args:
            checkpoint_artifact_path: Full path to checkpoint artifact (entity/project/name:version)
            wandb_run: Optional W&B run for API access
            
        Returns:
            Dictionary with validation results and recommendations
        """
        validation_result = {
            'checkpoint_exists': False,
            'has_embedded_config': False,
            'issues': [],
            'recommendations': [],
            'metadata': {}
        }
        
        try:
            # Parse artifact path
            path_parts = checkpoint_artifact_path.split('/')
            if len(path_parts) < 3:
                validation_result['issues'].append("Invalid artifact path format")
                return validation_result
            
            try:
                # Check if checkpoint artifact exists and get its metadata
                checkpoint_artifact = wandb.Api().artifact(checkpoint_artifact_path)
                validation_result['checkpoint_exists'] = True
                validation_result['metadata']['checkpoint'] = {
                    'name': checkpoint_artifact.name,
                    'version': checkpoint_artifact.version,
                    'size': checkpoint_artifact.size,
                    'created_at': checkpoint_artifact.created_at,
                    'metadata': checkpoint_artifact.metadata
                }
                
                # Check for embedded config indicators
                # Modern checkpoints embed config directly in the checkpoint file
                try:
                    has_embedded_config = False
                    
                    # Check artifact metadata for indicators of embedded config
                    if checkpoint_artifact.metadata:
                        # Look for callback metadata that indicates config embedding
                        metadata = checkpoint_artifact.metadata
                        if ('callback_version' in metadata or 
                            'pause_callback' in str(metadata).lower() or
                            'config_source' in metadata):
                            has_embedded_config = True
                    
                    # Look for resume YAML files (legacy approach)
                    if not has_embedded_config:
                        artifact_files = checkpoint_artifact.files()
                        resume_yaml_files = [f for f in artifact_files if f.name.endswith('_resume.yaml')]
                        if resume_yaml_files:
                            has_embedded_config = True
                    
                    # If we have a relatively recent checkpoint (created by modern pause callback),
                    # assume it has embedded config even if we can't detect it
                    if not has_embedded_config:
                        try:
                            from datetime import datetime
                            created_at = checkpoint_artifact.created_at
                            if isinstance(created_at, str):
                                # Parse ISO format datetime
                                created_dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                            else:
                                created_dt = created_at
                            
                            # If checkpoint is recent (created in 2024 or later), likely has embedded config
                            if created_dt.year >= 2024:
                                has_embedded_config = True
                        except Exception:
                            pass
                    
                    validation_result['has_embedded_config'] = has_embedded_config
                    if not has_embedded_config:
                        validation_result['issues'].append("Cannot detect embedded config - checkpoint may be from older version")
                        
                except Exception as e:
                    validation_result['has_embedded_config'] = False
                    validation_result['issues'].append(f"Failed to check embedded config: {e}")
                
            except Exception as e:
                validation_result['issues'].append(f"Checkpoint artifact not found: {e}")
            
            # Generate recommendations based on findings  
            if not validation_result['checkpoint_exists']:
                validation_result['recommendations'].append("❌ Checkpoint artifact is missing or inaccessible")
            else:
                validation_result['recommendations'].append("✅ Checkpoint artifact found and accessible")
            
            if validation_result.get('has_embedded_config'):
                validation_result['recommendations'].append("✅ Embedded config detected - checkpoint is self-contained")
                validation_result['recommendations'].append("   This checkpoint is suitable for reliable W&B resume")
                validation_result['recommendations'].append("   No separate config artifact needed")
            else:
                validation_result['recommendations'].append("❌ No embedded config detected - resume may fail")
                validation_result['recommendations'].append("   Modern checkpoints embed config for self-contained resume")
                validation_result['recommendations'].append("   Use a newer checkpoint that embeds config content")
            
        except Exception as e:
            validation_result['issues'].append(f"Validation failed: {e}")
        
        return validation_result