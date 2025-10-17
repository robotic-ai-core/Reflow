"""
Config embedding mixin for embedding configurations in checkpoints.

CRITICAL: This mixin embeds ONLY Lightning's auto-generated config.yaml file,
which contains the COMPLETE merged configuration (trainer_defaults + config files).

DO NOT attempt to capture Lightning state ourselves. The only source of
embedded config should be Lightning's own config.yaml file that it automatically
generates in the log directory on every run.

This approach ensures:
1. Complete configuration reproducibility (includes trainer_defaults)
2. Consistency between initial training and pause/resume sessions
3. Single source of truth from Lightning's own config management
"""

import time
import hashlib
from typing import Dict, Any, Optional
from pathlib import Path
import sys
import logging
import os
import yaml

from lightning.pytorch import Trainer, LightningModule

logger = logging.getLogger(__name__)


class ConfigEmbeddingMixin:
    """
    Mixin to add config embedding functionality to any checkpoint callback.
    
    DESIGN PRINCIPLE: Only embed Lightning's auto-generated config.yaml
    
    Lightning automatically saves the COMPLETE merged configuration to:
    {log_dir}/config.yaml
    
    This file contains:
    - trainer_defaults (set programmatically)
    - Config file settings
    - All other Lightning CLI settings
    
    We embed this COMPLETE config, not individual pieces we try to capture ourselves.
    
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

        # CRITICAL: Embed Lightning's auto-generated config.yaml ONLY
        # This is the COMPLETE merged configuration including trainer_defaults
        try:
            lightning_config = self._capture_lightning_auto_config(trainer)
        except RuntimeError as e:
            # Config validation failed - this is CRITICAL and should stop checkpoint saving
            logger.error(f"❌ Config embedding failed: {e}")
            # Re-raise to fail the checkpoint save - better to fail loudly than silently embed bad config
            raise RuntimeError(
                f"Cannot save checkpoint: config embedding failed. {e}\n\n"
                f"This is a safety measure to prevent embedding stale/invalid configuration "
                f"that would cause resume failures later."
            ) from e

        metadata = checkpoint.get(metadata_key, {})
        
        # Add basic metadata (these are our custom metadata for resume)
        metadata.update({
            'timestamp': time.time(),
            'original_command': self._original_argv,
            'checkpoint_version': '1.0.0',
            'global_step': trainer.global_step,
            'current_epoch': trainer.current_epoch,
        })
        
        # Add W&B run ID if available (custom metadata for resume)
        wandb_run_id = self._get_wandb_run_id()
        if wandb_run_id:
            metadata['wandb_run_id'] = wandb_run_id
            logger.info(f"📋 Storing W&B run ID in checkpoint: {wandb_run_id}")
        
        metadata['embedded_config_content'] = lightning_config
        metadata['config_hash'] = self._calculate_config_hash(lightning_config)
        metadata['config_source'] = 'lightning_auto_generated'
        logger.debug(f"📝 Embedded Lightning's auto-generated config.yaml ({len(lightning_config)} chars)")
        
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
    

    def _capture_lightning_auto_config(self, trainer: Trainer) -> str:
        """
        Capture Lightning's auto-generated config.yaml file ONLY.

        Lightning automatically saves the COMPLETE merged configuration
        (trainer_defaults + config files) to {log_dir}/config.yaml.

        This is our SINGLE SOURCE OF TRUTH for embedded config.

        Returns:
            Raw YAML string content of Lightning's complete config

        Raises:
            RuntimeError: If config is missing, stale, or invalid
        """
        try:
            # Step 1: Find Lightning's log directory
            lightning_log_dir = self._get_lightning_log_dir(trainer)
            if not lightning_log_dir:
                raise RuntimeError(
                    "CRITICAL: Could not find Lightning log directory. "
                    "Config embedding requires a valid log directory. "
                    "Ensure trainer has a logger configured."
                )

            # Step 2: Read Lightning's auto-generated config.yaml as raw string
            config_path = lightning_log_dir / "config.yaml"
            if not config_path.exists():
                raise RuntimeError(
                    f"CRITICAL: Lightning config not found at {config_path}. "
                    f"This usually means save_config_callback=None was set in LightningCLI, "
                    f"which disables config saving. Config embedding REQUIRES Lightning to save "
                    f"the config file. Remove save_config_callback=None from your CLI initialization."
                )

            # Step 3: Validate config is fresh (not stale from previous runs)
            self._validate_config_freshness(config_path, lightning_log_dir, trainer)

            with open(config_path, 'r') as f:
                lightning_config_string = f.read().strip()

            if not lightning_config_string:
                raise RuntimeError(
                    f"CRITICAL: Lightning config at {config_path} is empty. "
                    f"This indicates a problem with Lightning's config saving."
                )

            # Step 4: Return raw Lightning config string (preserve exact formatting)
            logger.info(f"✅ Captured fresh config from {config_path}")
            logger.debug(f"📝 Config size: {len(lightning_config_string)} chars")

            return lightning_config_string

        except RuntimeError:
            # Re-raise RuntimeError as-is (these are our validation errors)
            raise
        except Exception as e:
            raise RuntimeError(
                f"CRITICAL: Failed to capture Lightning auto config: {e}. "
                f"Config embedding is required for checkpoint resume."
            )

    def _validate_config_freshness(self, config_path: Path, log_dir: Path, trainer: Trainer) -> None:
        """
        Validate that the config file is fresh and not stale from previous runs.

        Args:
            config_path: Path to the config.yaml file
            log_dir: Lightning's log directory
            trainer: PyTorch Lightning trainer

        Raises:
            RuntimeError: If stale config is detected
        """
        try:
            # Get config file modification time
            config_mtime = config_path.stat().st_mtime
            config_age_seconds = time.time() - config_mtime

            # Check 1: Detect if config is suspiciously old (older than 1 hour before current run)
            # Allow some buffer since trainer.start_time might not be set yet
            if config_age_seconds > 3600:  # 1 hour
                logger.warning(
                    f"⚠️  Config file is {config_age_seconds / 60:.1f} minutes old. "
                    f"This may indicate a stale config from a previous run. "
                    f"Config path: {config_path}"
                )
                # Don't raise yet, check if it's in a parent directory (stronger signal)

            # Check 2: Detect if config is in a parent/fallback directory (VERY suspicious)
            # Lightning should create config in the run-specific log directory
            # If we're reading from a parent directory, it's likely stale
            config_parent = config_path.parent

            # Check if config_parent is a proper subdirectory of the logging root
            # For example:
            #   Good: logs/protoworld/run_xyz/config.yaml  (run-specific)
            #   Bad:  logs/config.yaml  (parent directory - STALE!)

            # Simple heuristic: if log_dir has multiple path components compared to config_parent,
            # we're likely in a run-specific directory
            try:
                # Get relative path - if config is in parent, this will have '..'
                rel_path = config_path.relative_to(log_dir)
                if str(rel_path) == "config.yaml":
                    # Config is directly in log_dir - this is expected
                    logger.debug(f"✅ Config is in run-specific directory: {log_dir}")
                else:
                    # Config is in a subdirectory or weird location
                    logger.warning(f"⚠️  Unexpected config location: {config_path} relative to {log_dir}")
            except ValueError:
                # config_path is not relative to log_dir - VERY suspicious
                raise RuntimeError(
                    f"CRITICAL: Config file is outside the expected log directory!\n"
                    f"  Config path: {config_path}\n"
                    f"  Expected log dir: {log_dir}\n"
                    f"This indicates config is being loaded from a fallback location, "
                    f"which likely means it's STALE from a previous run. "
                    f"Ensure save_config_callback is NOT disabled in your CLI initialization."
                )

            # Check 3: Warn if config looks like it's from a shared parent directory
            # For example: logs/config.yaml instead of logs/protoworld/run_xyz/config.yaml
            if len(log_dir.parts) < 3:
                # Very short path like "logs" - suspicious
                logger.warning(
                    f"⚠️  Log directory path is very short: {log_dir}. "
                    f"Expected a run-specific subdirectory. "
                    f"Config at {config_path} may be stale."
                )

        except Exception as e:
            # Don't fail validation on unexpected errors, just log
            logger.warning(f"Could not validate config freshness: {e}")

    def _get_lightning_log_dir(self, trainer: Trainer) -> Optional[Path]:
        """
        Get Lightning's log directory where it saves config.yaml.

        Returns:
            Path to Lightning's log directory, or None if not available
        """
        try:
            # Try to get log_dir from trainer's logger
            if hasattr(trainer, 'logger') and trainer.logger is not None:
                if hasattr(trainer.logger, 'log_dir') and trainer.logger.log_dir:
                    return Path(trainer.logger.log_dir)
                elif hasattr(trainer.logger, 'save_dir') and trainer.logger.save_dir:
                    # Some loggers use save_dir instead of log_dir
                    return Path(trainer.logger.save_dir)

            # Fallback to trainer's default_root_dir
            if hasattr(trainer, 'default_root_dir') and trainer.default_root_dir:
                return Path(trainer.default_root_dir)

            logger.warning("Could not determine Lightning log directory from trainer")
            return None
        except Exception as e:
            logger.warning(f"Failed to get Lightning log directory: {e}")
            return None
    
    def _calculate_config_hash(self, config_content: str) -> str:
        """Calculate SHA256 hash of config content for validation."""
        return hashlib.sha256(config_content.encode()).hexdigest()
    
    def _get_wandb_run_id(self) -> Optional[str]:
        """Extract W&B run ID from trainer's logger if available."""
        try:
            if hasattr(self, '_trainer_ref') and self._trainer_ref:
                trainer = self._trainer_ref
                if hasattr(trainer, 'logger') and trainer.logger:
                    # Check for WandB logger
                    if hasattr(trainer.logger, 'experiment') and hasattr(trainer.logger.experiment, 'id'):
                        return trainer.logger.experiment.id
                    # Check for multi-logger case
                    elif hasattr(trainer.logger, 'experiment') and isinstance(trainer.logger.experiment, list):
                        for exp in trainer.logger.experiment:
                            if hasattr(exp, 'id'):
                                return exp.id
            return None
        except Exception as e:
            logger.warning(f"Failed to extract W&B run ID: {e}")
        return None