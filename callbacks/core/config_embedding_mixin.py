"""
Config embedding mixin for embedding configurations in checkpoints.

This mixin provides functionality to embed resolved configurations in checkpoints:
1. Embedding the resolved configuration (including CLI overrides)
2. Config synthesis and validation
3. W&B run ID storage
4. Config hash validation

This is separate from manager state operations and can be used by callbacks
that need config persistence without triggering global manager state operations.
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
    - Embedded resolved configuration with CLI overrides
    - Config synthesis and validation
    - W&B run ID storage
    - Config hash validation
    
    Usage:
        class MyCheckpoint(ModelCheckpoint, ConfigEmbeddingMixin):
            def on_save_checkpoint(self, trainer, pl_module, checkpoint):
                super().on_save_checkpoint(trainer, pl_module, checkpoint)
                self.add_config_metadata(trainer, pl_module, checkpoint)
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._synthesized_config: Optional[str] = None
        self._config_synthesis_attempted = False
        self._original_argv = sys.argv.copy()
        self._cli_config_validated = False
    
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
                "❌ CRITICAL: ConfigEmbeddingMixin requires DiffusionFlowLightningCLI context for self-contained checkpoints. "
                "This callback cannot create reproducible checkpoints without CLI-generated config.yaml. "
                "Either use DiffusionFlowLightningCLI instead of raw PyTorch Lightning Trainer, "
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
        
        # Embed resolved config - MANDATORY for reproducibility
        resolved_config = self._capture_resolved_config(trainer)
        if not resolved_config:
            error_msg = (
                "❌ CRITICAL: Failed to capture CLI configuration with overrides. "
                "Cannot create reproducible checkpoint without resolved config. "
                "CLI reference may not be stored in trainer or config synthesis failed."
            )
            logger.error(error_msg)
            raise RuntimeError(
                "CLI config capture failed - cannot guarantee reproducible checkpoint. "
                "Check that trainer.cli is properly set and accessible."
            )
        
        metadata['embedded_config_content'] = resolved_config
        metadata['config_hash'] = self._calculate_config_hash(resolved_config)
        metadata['config_source'] = 'resolved_with_overrides'
        logger.info(f"📝 Embedded resolved config with CLI overrides ({len(resolved_config)} chars)")
        
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
    
    def _synthesize_resolved_config(self, trainer: Trainer, context: str = "checkpoint save") -> bool:
        """
        Get the resolved configuration using Lightning's auto-generated config.yaml file.
        
        Returns:
            bool: True if config retrieval succeeded
        """
        if self._synthesized_config is not None:
            return True
            
        if self._config_synthesis_attempted:
            return self._synthesized_config is not None
            
        self._config_synthesis_attempted = True
        
        try:
            # Use Lightning's auto-generated config.yaml file (guaranteed to exist)
            logger.info(f"📄 Reading Lightning's auto-generated config.yaml for {context}")
            
            # Lightning CLI automatically saves config.yaml in current directory
            config_path = Path("config.yaml")
            
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_yaml = f.read()
                
                # Validate config content before accepting it
                self._validate_config_content(config_yaml, context)
                
                self._synthesized_config = config_yaml
                logger.info(f"✅ Using Lightning's auto-generated config.yaml for {context} ({len(config_yaml)} chars)")
                return True
            else:
                # This should never happen unless config saving is disabled
                error_msg = (
                    f"Lightning config.yaml not found at {config_path.absolute()}. "
                    "This file is required for self-contained checkpoints. "
                    "Ensure DiffusionFlowLightningCLI is not initialized with save_config_kwargs=False."
                )
                logger.error(f"❌ CRITICAL: {error_msg}")
                raise RuntimeError(error_msg)
                
        except RuntimeError:
            # Re-raise RuntimeError to ensure proper configuration
            raise
        except PermissionError as e:
            # Configuration error - should fail fast, not mask
            error_msg = f"Cannot read Lightning config due to permissions: {e}. Check file permissions on {config_path.absolute()}"
            logger.error(f"❌ CRITICAL: {error_msg}")
            raise RuntimeError(error_msg)
        except FileNotFoundError as e:
            # Should be caught by exists() check above, but just in case
            error_msg = f"Lightning config file disappeared during read: {e}"
            logger.error(f"❌ CRITICAL: {error_msg}")
            raise RuntimeError(error_msg)
        except IOError as e:
            # IO errors indicate serious system issues
            error_msg = f"IO error reading Lightning config: {e}. Check disk space and file system health"
            logger.error(f"❌ CRITICAL: {error_msg}")
            raise RuntimeError(error_msg)
        except Exception as e:
            # Unknown errors should not be masked
            error_msg = f"Unexpected error reading Lightning config: {e}"
            logger.error(f"❌ CRITICAL: {error_msg}")
            raise RuntimeError(error_msg)
    
    def _validate_config_content(self, config_yaml: str, context: str) -> None:
        """
        Validate that config content is reasonable and usable.
        
        Args:
            config_yaml: The raw YAML content to validate
            context: Context for error messages
            
        Raises:
            RuntimeError: If config content is invalid or unusable
        """
        # Check for empty content
        if not config_yaml.strip():
            raise RuntimeError(f"Lightning config file is empty at {context}. This indicates a serious configuration issue.")
        
        # Basic YAML syntax validation
        try:
            import yaml
            parsed_config = yaml.safe_load(config_yaml)
        except Exception as e:
            raise RuntimeError(f"Lightning config contains invalid YAML at {context}: {e}")
        
        # Validate it's actually a config structure
        if not isinstance(parsed_config, dict):
            raise RuntimeError(f"Lightning config is not a valid configuration object at {context}")
        
        # Check for minimum required content that Lightning should always generate
        required_sections = ['trainer', 'model', 'data']
        if not any(section in parsed_config for section in required_sections):
            raise RuntimeError(
                f"Lightning config missing required sections at {context}. "
                f"Expected at least one of: {required_sections}. "
                f"Found sections: {list(parsed_config.keys())}"
            )
        
        # Check reasonable file size (Lightning configs should not be tiny or huge)
        config_size = len(config_yaml)
        if config_size < 100:  # Less than 100 chars is suspicious
            logger.warning(f"⚠️ Lightning config is unusually small ({config_size} chars) at {context}")
        elif config_size > 10 * 1024 * 1024:  # More than 10MB is suspicious
            logger.warning(f"⚠️ Lightning config is unusually large ({config_size} chars) at {context}")
    
    def _capture_resolved_config(self, trainer: Trainer) -> Optional[str]:
        """
        Capture the resolved configuration.
        
        Returns:
            YAML string of the complete resolved configuration, or None if synthesis failed
        """
        # Try to synthesize if not done yet
        if self._synthesized_config is None:
            self._synthesize_resolved_config(trainer, context="config capture")
        
        return self._synthesized_config
    
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