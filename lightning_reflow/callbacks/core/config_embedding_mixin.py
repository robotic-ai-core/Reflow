"""
Config embedding mixin for embedding configurations in checkpoints.

This mixin provides functionality to embed Lightning's auto-generated config.yaml
directly in checkpoints for reproducibility.
"""

import time
import hashlib
from typing import Dict, Any, Optional
from pathlib import Path
import sys
import logging
import os

from lightning.pytorch import Trainer, LightningModule

logger = logging.getLogger(__name__)


class ConfigEmbeddingMixin:
    """
    Mixin to add config embedding functionality to any checkpoint callback.
    
    This mixin provides:
    - Direct embedding of Lightning's auto-generated config.yaml
    - Config hash validation
    - W&B run ID storage
    
    Usage:
        class MyCheckpoint(ModelCheckpoint, ConfigEmbeddingMixin):
            def on_save_checkpoint(self, trainer, pl_module, checkpoint):
                super().on_save_checkpoint(trainer, pl_module, checkpoint)
                self.add_config_metadata(trainer, pl_module, checkpoint)
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._original_argv = sys.argv.copy()
    
    def _can_embed_config(self, trainer: Trainer) -> bool:
        """
        Check if the context allows for config embedding.
        
        Returns True if trainer has a valid CLI reference, False otherwise.
        """
        if hasattr(trainer, 'cli') and trainer.cli is not None:
            cli = trainer.cli
            if hasattr(cli, 'save_config_kwargs') and cli.save_config_kwargs is False:
                logger.warning("Config embedding disabled (save_config_kwargs=False).")
                return False
            return True
        else:
            logger.warning("No CLI context on trainer, cannot embed config.")
            return False
    
    def add_config_metadata(self, trainer: Trainer, pl_module: LightningModule, 
                           checkpoint: Dict[str, Any], metadata_key: str = 'self_contained_metadata') -> None:
        """
        Add config-related metadata to checkpoint.
        
        Args:
            trainer: PyTorch Lightning trainer
            pl_module: Lightning module being trained
            checkpoint: Checkpoint dictionary to modify
            metadata_key: Key to store metadata under in checkpoint
        """
        # Store trainer reference for config capture
        self._trainer_ref = trainer
        
        # Only embed config if the context is valid
        if not self._can_embed_config(trainer):
            logger.warning("Skipping config embedding due to invalid context.")
            return
            
        metadata = checkpoint.get(metadata_key, {})
        
        # Add basic metadata
        metadata.update({
            'timestamp': time.time(),
            'original_command': self._original_argv,
            'checkpoint_version': '1.0.0',
            'global_step': trainer.global_step,
            'current_epoch': trainer.current_epoch,
        })
        
        # Add W&B run ID if available
        wandb_run_id = self._get_wandb_run_id()
        if wandb_run_id:
            metadata['wandb_run_id'] = wandb_run_id
            logger.info(f"📋 Storing W&B run ID in checkpoint: {wandb_run_id}")
        
        # Embed Lightning's config.yaml directly - MANDATORY for reproducibility
        lightning_config = self._capture_lightning_config(trainer)
        if not lightning_config:
            logger.warning(
                "Could not capture Lightning's config. "
                "Checkpoint will not be self-contained."
            )
            return
        
        metadata['embedded_config_content'] = lightning_config
        metadata['config_hash'] = self._calculate_config_hash(lightning_config)
        metadata['config_source'] = 'lightning_auto_generated'
        logger.info(f"📝 Embedded Lightning's auto-generated config.yaml ({len(lightning_config)} chars)")
        
        checkpoint[metadata_key] = metadata
    
    def restore_config_metadata(self, trainer: Trainer, pl_module: LightningModule,
                               checkpoint: Dict[str, Any], metadata_key: str = 'self_contained_metadata') -> None:
        """
        Restore config-related metadata from checkpoint.
        
        Args:
            trainer: PyTorch Lightning trainer
            pl_module: Lightning module being trained
            checkpoint: Checkpoint dictionary
            metadata_key: Key where metadata is stored in checkpoint
        """
        metadata = checkpoint.get(metadata_key, {})
        if not metadata:
            return
        
        logger.info(f"✅ Loading config metadata from checkpoint")
        logger.info(f"   Created at: {time.ctime(metadata.get('timestamp', 0))}")
        
        # Config metadata is automatically available in checkpoint
        # No specific restoration needed for config content
    
    def _capture_lightning_config(self, trainer) -> Optional[str]:
        """
        Capture Lightning's configuration with config.yaml file as primary source.
        
        Returns:
            YAML string of Lightning's config, or None if not available
        """
        # First, try to load from the trainer's config_path if available
        config_path = self._get_config_path_from_cli(trainer)
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    # Return the raw YAML content
                    config_yaml = f.read()
                    logger.info(f"📋 Captured config from '{config_path}' for embedding")
                    return config_yaml
            except Exception as e:
                logger.warning(f"⚠️ Failed to read config from '{config_path}': {e}")
        
        logger.warning("No config source found for embedding.")
        return None

    def _get_config_path_from_cli(self, trainer) -> Optional[str]:
        """Attempt to get the config path from the LightningCLI object."""
        try:
            argv = self._original_argv
            for i, arg in enumerate(argv):
                if arg in ['--config', '-c'] and i + 1 < len(argv):
                    return argv[i + 1]
                elif arg.startswith('--config='):
                    return arg.split('=', 1)[1]
            return None
        except Exception as e:
            logger.warning(f"Failed to extract config path: {e}")
            return None
    
    def _calculate_config_hash(self, config_content: str) -> str:
        """Calculate SHA256 hash of config content for validation."""
        return hashlib.sha256(config_content.encode()).hexdigest()
    
    def _get_wandb_run_id(self) -> Optional[str]:
        """Extract W&B run ID if available."""
        try:
            import wandb
            if wandb.run and wandb.run.id:
                return wandb.run.id
        except ImportError:
            pass
        return None