"""
Unified Artifact Manager for W&B uploads.

This module consolidates all artifact upload functionality to eliminate duplication
between checkpoint and config uploads, providing a single consistent interface.
"""

import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import wandb
from lightning.pytorch import Trainer
from lightning.pytorch.core import LightningModule
from lightning.pytorch.loggers import WandbLogger

from ..logging.logging_config import get_logger


class UnifiedArtifactManager:
    """
    Consolidated artifact manager that handles all types of W&B uploads.
    
    This class eliminates duplication between checkpoint and config uploads
    by providing a unified interface for all artifact operations.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the unified artifact manager.
        
        Args:
            verbose: Whether to log verbose messages
        """
        self.verbose = verbose
        self.logger = get_logger(__name__)
    
    @staticmethod
    def get_wandb_run(trainer: Trainer) -> Optional[wandb.sdk.wandb_run.Run]:
        """
        Get W&B run from trainer's logger.
        
        Args:
            trainer: Lightning trainer instance
            
        Returns:
            W&B run object if found, None otherwise
        """
        if hasattr(trainer, 'logger') and isinstance(trainer.logger, WandbLogger):
            return trainer.logger.experiment
        return None
    
    def upload_artifact(
        self,
        trainer: Trainer,
        files: Dict[str, str],
        artifact_name: str,
        artifact_type: str,
        aliases: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        wandb_run: Optional[wandb.sdk.wandb_run.Run] = None,
        pl_module: Optional[LightningModule] = None
    ) -> Optional[str]:
        """
        Upload files as W&B artifact with unified handling.
        
        Args:
            trainer: Lightning trainer instance
            files: Dictionary mapping artifact paths to local file paths
            artifact_name: Name of the artifact
            artifact_type: Type of artifact (checkpoint, config, etc.)
            aliases: List of aliases for the artifact
            metadata: Additional metadata to include
            wandb_run: Optional W&B run (will auto-detect if not provided)
            pl_module: Optional Lightning module for checkpoint metadata
            
        Returns:
            Artifact path (name:version) if successful, None otherwise
        """
        try:
            # Validate files
            validated_files = self._validate_files(files)
            if not validated_files:
                if self.verbose and trainer.is_global_zero:
                    self.logger.warning(f"No valid files to upload for {artifact_name}")
                return None
            
            # Get W&B run
            if wandb_run is None:
                wandb_run = self.get_wandb_run(trainer)
            
            if not wandb_run:
                if self.verbose and trainer.is_global_zero:
                    self.logger.warning(f"No W&B run available for {artifact_name} upload")
                return None
            
            # Create artifact with metadata
            artifact = self._create_artifact(
                wandb_run=wandb_run,
                name=artifact_name,
                artifact_type=artifact_type,
                metadata=metadata,
                trainer=trainer,
                pl_module=pl_module
            )
            
            # Add files to artifact
            for artifact_path, local_path in validated_files.items():
                if self.verbose and trainer.is_global_zero:
                    self.logger.info(f"Adding {local_path} to {artifact_name}")
                artifact.add_file(local_path, artifact_path)
            
            # Upload with aliases
            if aliases:
                artifact.aliases = aliases
            
            if self.verbose and trainer.is_global_zero:
                self.logger.info(f"Uploading {artifact_name} artifact...")
            
            artifact_path = wandb_run.log_artifact(artifact)
            
            if self.verbose and trainer.is_global_zero:
                self.logger.info(f"Successfully uploaded {artifact_name} artifact: {artifact_path}")
            
            return artifact_path
            
        except Exception as e:
            if self.verbose and trainer.is_global_zero:
                self.logger.warning(f"Failed to upload {artifact_name} artifact: {e}")
            return None
    
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
        Upload checkpoint as W&B artifact using unified interface.
        
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
        # Generate artifact name
        if wandb_run is None:
            wandb_run = self.get_wandb_run(trainer)
        
        if not wandb_run:
            return None
        
        artifact_name = f"{wandb_run.id}-{ckpt_type}"
        
        # Prepare files dictionary
        files = {Path(filepath).name: filepath}
        
        # Prepare metadata
        metadata = self._create_checkpoint_metadata(
            trainer=trainer,
            pl_module=pl_module,
            ckpt_type=ckpt_type,
            score=score,
            epoch=epoch,
            step=step,
            extra_metadata=extra_metadata
        )
        
        return self.upload_artifact(
            trainer=trainer,
            files=files,
            artifact_name=artifact_name,
            artifact_type="model",
            aliases=aliases,
            metadata=metadata,
            wandb_run=wandb_run,
            pl_module=pl_module
        )
    
    def upload_config_artifact(
        self,
        trainer: Trainer,
        config_paths: Union[str, List[str]],
        run_id: Optional[str] = None,
        wandb_run: Optional[wandb.sdk.wandb_run.Run] = None,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Upload config files as W&B artifact using unified interface.
        
        Args:
            trainer: Lightning trainer instance
            config_paths: Path to config file or list of config file paths
            run_id: Optional run ID (will auto-detect if not provided)
            wandb_run: Optional W&B run (will auto-detect if not provided)
            extra_metadata: Additional metadata to include
            
        Returns:
            Artifact path (name:version) if successful, None otherwise
        """
        # Normalize config_paths to list
        if isinstance(config_paths, str):
            config_paths = [config_paths]
        
        # Prepare files dictionary
        files = {}
        for config_path in config_paths:
            if config_path and Path(config_path).exists():
                files[Path(config_path).name] = config_path
        
        if not files:
            if self.verbose and trainer.is_global_zero:
                self.logger.warning("No valid config files found")
            return None
        
        # Get run ID
        if run_id is None:
            if wandb_run is None:
                wandb_run = self.get_wandb_run(trainer)
            run_id = wandb_run.id if wandb_run else "unknown"
        
        # Generate artifact name
        artifact_name = f"{run_id}-config"
        
        # Prepare metadata
        metadata = self._create_config_metadata(
            trainer=trainer,
            config_paths=list(files.values()),
            extra_metadata=extra_metadata
        )
        
        return self.upload_artifact(
            trainer=trainer,
            files=files,
            artifact_name=artifact_name,
            artifact_type="config",
            aliases=None,
            metadata=metadata,
            wandb_run=wandb_run
        )
    
    def _validate_files(self, files: Dict[str, str]) -> Dict[str, str]:
        """Validate that all files exist and are not empty."""
        validated_files = {}
        for artifact_path, local_path in files.items():
            if Path(local_path).exists() and Path(local_path).stat().st_size > 0:
                validated_files[artifact_path] = local_path
            else:
                if self.verbose:
                    self.logger.warning(f"Invalid file: {local_path}")
        return validated_files
    
    def _create_artifact(
        self,
        wandb_run: wandb.sdk.wandb_run.Run,
        name: str,
        artifact_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        trainer: Optional[Trainer] = None,
        pl_module: Optional[LightningModule] = None
    ) -> wandb.Artifact:
        """Create W&B artifact with standardized metadata."""
        artifact = wandb.Artifact(name=name, type=artifact_type, metadata=metadata or {})
        
        # Add standard metadata
        artifact.metadata.update({
            "created_by": "DiffusionFlow",
            "created_timestamp": time.time(),
            "wandb_run_id": wandb_run.id
        })
        
        # Add W&B relationship metadata for checkpoints
        if artifact_type == "model" and wandb_run:
            artifact.metadata.update({
                'entity': getattr(wandb_run, 'entity', 'unknown'),
                'project': getattr(wandb_run, 'project', 'unknown'),
                'artifact_relationships': {
                    'expected_config_artifact': f"{getattr(wandb_run, 'entity', 'unknown')}/{getattr(wandb_run, 'project', 'unknown')}/{wandb_run.id}-config",
                    'expected_config_artifact_name': f"{wandb_run.id}-config"
                }
            })
        
        # Add trainer metadata if available
        if trainer:
            artifact.metadata.update({
                "current_epoch": trainer.current_epoch,
                "global_step": trainer.global_step,
                "trainer_state": str(trainer.state)
            })
        
        # Add model metadata if available
        if pl_module:
            artifact.metadata.update({
                "model_class": pl_module.__class__.__name__,
                "model_hparams": getattr(pl_module, 'hparams', {})
            })
        
        return artifact
    
    def _create_checkpoint_metadata(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        ckpt_type: str,
        score: Optional[float] = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create checkpoint-specific metadata."""
        metadata = {
            "checkpoint_type": ckpt_type,
            "epoch": epoch or trainer.current_epoch,
            "global_step": step or trainer.global_step,
            "model_class": pl_module.__class__.__name__,
            "pytorch_version": torch.__version__,
        }
        
        if score is not None:
            metadata["score"] = score
        
        if extra_metadata:
            metadata.update(extra_metadata)
        
        return metadata
    
    def _create_config_metadata(
        self,
        trainer: Trainer,
        config_paths: List[str],
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create config-specific metadata."""
        metadata = {
            "config_files": [Path(p).name for p in config_paths],
            "num_config_files": len(config_paths),
            "current_epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
        }
        
        if extra_metadata:
            metadata.update(extra_metadata)
        
        return metadata