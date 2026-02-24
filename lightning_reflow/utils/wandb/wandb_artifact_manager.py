"""
Shared W&B artifact management utilities for all callbacks.

This module provides centralized W&B artifact operations to eliminate code duplication
between different callbacks that need to upload checkpoints, configs, or other artifacts.

Note: This module delegates to UnifiedArtifactManager for all core functionality.
It provides a backward-compatible interface for existing callers.
"""

import logging
from typing import Any, Dict, List, Optional

import wandb
from lightning.pytorch import Trainer
from lightning.pytorch.core import LightningModule

from .unified_artifact_manager import UnifiedArtifactManager


class WandbArtifactManager:
    """
    Centralized W&B artifact management for all callbacks.

    This class provides shared functionality for uploading checkpoints, configs,
    and other artifacts to W&B, ensuring consistency across different callbacks.

    Note: This class delegates to UnifiedArtifactManager for all operations.
    It is maintained for backward compatibility with existing callers.
    """

    def __init__(self, verbose: bool = True, keep_n_versions: Optional[int] = None):
        """
        Initialize the artifact manager.

        Args:
            verbose: Whether to log verbose messages
            keep_n_versions: If set, delete older artifact versions after upload,
                keeping only the N most recent.
        """
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        self._unified_manager = UnifiedArtifactManager(
            verbose=verbose, keep_n_versions=keep_n_versions
        )

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
        return UnifiedArtifactManager.get_wandb_run_id(trainer)

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
            Full artifact path (entity/project/name:version) if successful, None otherwise
        """
        return self._unified_manager.upload_checkpoint_artifact(
            trainer=trainer,
            pl_module=pl_module,
            filepath=filepath,
            ckpt_type=ckpt_type,
            aliases=aliases,
            score=score,
            epoch=epoch,
            step=step,
            wandb_run=wandb_run,
            extra_metadata=extra_metadata
        )

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
        return self._unified_manager.extract_score_from_trainer(trainer, metric_name)

    def upload_config_artifact(
        self,
        trainer: Trainer,
        config_paths,
        run_id: Optional[str] = None,
        wandb_run: Optional[wandb.sdk.wandb_run.Run] = None,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Upload config files as W&B artifact.

        Args:
            trainer: Lightning trainer instance
            config_paths: Path to config file or list of config file paths
            run_id: Optional run ID (will auto-detect if not provided)
            wandb_run: Optional W&B run (will auto-detect if not provided)
            extra_metadata: Additional metadata to include

        Returns:
            Artifact path if successful, None otherwise
        """
        return self._unified_manager.upload_config_artifact(
            trainer=trainer,
            config_paths=config_paths,
            run_id=run_id,
            wandb_run=wandb_run,
            extra_metadata=extra_metadata
        )
