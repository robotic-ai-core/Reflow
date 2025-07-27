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
        self._cli_config_validated = False
        self._trainer_ref = None
    
    def _validate_cli_configuration(self, trainer: Trainer) -> None:
        """
        Validate that the CLI is properly configured for self-contained checkpoints.
        
        This ensures Lightning CLI will create a config file that we can read.
        Fails fast if CLI context is not available.
        """
        if self._cli_config_validated:
            return
            
        # Check if trainer has CLI reference
        if hasattr(trainer, 'cli') and trainer.cli is not None:
            cli = trainer.cli
            
            # Check save_config_kwargs
            if hasattr(cli, 'save_config_kwargs'):
                save_config_kwargs = cli.save_config_kwargs
                
                # Check if config saving is disabled
                if save_config_kwargs is False:
                    raise RuntimeError(
                        "❌ CRITICAL: DiffusionFlowLightningCLI has config saving disabled (save_config_kwargs=False). "
                        "Self-contained checkpoints require Lightning to save a config file. "
                        "Please remove save_config_kwargs=False or set it to a dict of save options."
                    )
                
                # If it's not False, it's either None (default), True, or a dict - all of which should work
                if save_config_kwargs is None:
                    logger.info("✅ CLI configuration validated: using default config saving behavior")
                elif save_config_kwargs is True or isinstance(save_config_kwargs, dict):
                    logger.info(f"✅ CLI configuration validated: save_config_kwargs={save_config_kwargs}")
                else:
                    logger.warning(f"⚠️ Unexpected save_config_kwargs type: {type(save_config_kwargs).__name__}")
            else:
                # If save_config_kwargs doesn't exist, Lightning uses default behavior (saves config)
                logger.info("✅ CLI configuration validated: save_config_kwargs not set, using Lightning defaults")
        else:
            # No CLI context - this is a critical error for self-contained checkpoints
            raise RuntimeError(
                "❌ CRITICAL: ConfigEmbeddingMixin requires LightningReflowCLI context for self-contained checkpoints. "
                "This callback cannot create reproducible checkpoints without CLI-generated config.yaml. "
                "Either use LightningReflowCLI instead of raw PyTorch Lightning Trainer, "
                "or remove ConfigEmbeddingMixin from callbacks that don't require config embedding."
            )
        
        self._cli_config_validated = True
    
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
        
        # Validate CLI configuration on first use
        self._validate_cli_configuration(trainer)
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
        lightning_config = self._capture_lightning_config()
        if not lightning_config:
            error_msg = (
                "❌ CRITICAL: Failed to capture Lightning's auto-generated config.yaml. "
                "Cannot create reproducible checkpoint without embedded config. "
                "Lightning should have created config.yaml automatically."
            )
            logger.error(error_msg)
            raise RuntimeError(
                "Lightning config capture failed - cannot guarantee reproducible checkpoint. "
                "Check that Lightning's config.yaml exists and is readable."
            )
        
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
    
    def _capture_lightning_config(self) -> Optional[str]:
        """
        Capture Lightning's configuration with config.yaml file as primary source.
        
        Returns:
            YAML string of Lightning's config, or None if not available
        """
        try:
            # Primary: Try to read config.yaml file from current run
            config_path = Path("config.yaml")
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_content = f.read()
                
                # Validate content is reasonable
                if config_content.strip():
                    logger.info("📋 Captured Lightning's config from config.yaml file (primary source)")
                    return config_content
                else:
                    logger.warning("Lightning's config.yaml is empty, trying CLI context fallback")
            else:
                logger.info(f"Lightning's config.yaml not found at {config_path.absolute()}, trying CLI context fallback")
            
            # Fallback: Try to get from CLI context if config.yaml unavailable
            if hasattr(self, '_trainer_ref') and self._trainer_ref and hasattr(self._trainer_ref, 'cli'):
                cli = self._trainer_ref.cli
                if hasattr(cli, 'config') and cli.config:
                    import yaml
                    # Convert the merged config to YAML string
                    config_yaml = yaml.dump(cli.config, default_flow_style=False, sort_keys=False)
                    logger.info("📋 Captured Lightning's merged config from CLI context (fallback)")
                    return config_yaml
                else:
                    logger.warning("CLI context available but no config found")
            else:
                logger.warning("No CLI context available for config fallback")
                
            return None
                
        except Exception as e:
            logger.error(f"Failed to capture Lightning's config: {e}")
            return None
    
    def _calculate_config_hash(self, config_content: str) -> str:
        """Calculate SHA256 hash of config content for validation."""
        return hashlib.sha256(config_content.encode()).hexdigest()
    
    def _extract_config_path_from_argv(self) -> Optional[str]:
        """Extract config file path from original command line arguments."""
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
    
    def _get_wandb_run_id(self) -> Optional[str]:
        """Extract W&B run ID if available."""
        try:
            import wandb
            if wandb.run and wandb.run.id:
                return wandb.run.id
        except ImportError:
            pass
        return None