"""
Lightning Reflow CLI - Command-line interface wrapper.

This module provides a thin CLI wrapper around the core LightningReflow class,
handling command-line argument parsing and translating them to method calls.
"""

import argparse
import logging
import os
import subprocess
import sys
import tempfile
import yaml
from pathlib import Path
from lightning.pytorch.cli import LightningCLI

from ..core import LightningReflow

logger = logging.getLogger(__name__)

# Constants
RESUME_COMMAND = 'resume'
FIT_COMMAND = 'fit'
CHECKPOINT_PATH_ARG = '--ckpt_path'
CONFIG_ARG = '--config'
TEMP_CONFIG_PREFIX = 'resume_config_'
TEMP_CONFIG_SUFFIX = '.yaml'

# Import shared configurations to eliminate duplication
from ..core.shared_config import DEFAULT_PAUSE_CALLBACK_CONFIG, DEFAULT_STEP_LOGGER_CONFIG


class LightningReflowCLI(LightningCLI):
    """
    Command-line interface for Lightning Reflow.
    
    This class extends Lightning's CLI with Lightning Reflow's advanced features
    like pause/resume, W&B integration, and sophisticated configuration management.
    
    It acts as a thin wrapper around the core LightningReflow class, parsing
    command-line arguments and translating them to method calls.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the Lightning Reflow CLI.
        
        For resume commands, spawns a new subprocess with the fit command.
        """
        logger.info("🎯 Initializing Lightning Reflow CLI")
        
        # Handle resume command by spawning subprocess with fit command
        if self._is_resume_command():
            self._execute_resume_as_subprocess()
            return
        
        # For fit commands with a checkpoint, enable config overwrite to avoid errors
        if self._is_fit_with_checkpoint():
            logger.info("🔧 Detected fit from checkpoint, enabling config overwrite.")
            self._enable_config_overwrite(kwargs)
        
        # Initialize the Lightning CLI for standard commands (fit, validate, test, predict)
        super().__init__(*args, **kwargs)
    
    def before_instantiate_classes(self) -> None:
        """
        Hook called BEFORE Lightning instantiates any classes.
        
        This is critical for setting environment variables like PYTORCH_CUDA_ALLOC_CONF
        that MUST be set before CUDA/PyTorch initialization occurs.
        """
        logger.info("🔧 Processing environment variables (BEFORE instantiation)")
        self._process_environment_callback_config()
    
    def before_fit(self) -> None:
        """Hook called before fit starts - add essential callbacks."""
        self._add_essential_callbacks()
    
    def _add_essential_callbacks(self) -> None:
        """Add essential callbacks if not already present using shared logic."""
        if not hasattr(self, 'trainer') or not self.trainer:
            return
        
        from ..core.shared_config import ensure_essential_callbacks
        
        # Use shared logic to ensure essential callbacks
        self.trainer.callbacks = ensure_essential_callbacks(self.trainer.callbacks, self.trainer)
    
    def instantiate_trainer(self, **kwargs):
        """Override to apply shared defaults and set CLI reference in trainer."""
        from ..core.shared_config import get_trainer_defaults
        
        # Apply shared trainer defaults before calling parent
        current_trainer_config = self.config.get('trainer', {})
        
        # Check for progress bar conflicts in CLI config
        if current_trainer_config.get('enable_progress_bar') is True:
            logger.warning("⚠️ CLI config is trying to enable Lightning's default progress bar!")
            logger.warning("⚠️ This will conflict with Reflow's custom progress bar (PauseCallback).")
            logger.warning("⚠️ Ignoring enable_progress_bar=true from CLI config to prevent UI conflicts.")
            logger.warning("⚠️ If you need to disable progress bars entirely, set enable_pause=False in PauseCallback config.")
            
            # Remove the conflicting setting
            current_trainer_config = current_trainer_config.copy()
            current_trainer_config.pop('enable_progress_bar', None)
        
        merged_config = get_trainer_defaults(current_trainer_config)
        self.config['trainer'] = merged_config
        logger.info("✅ Applied shared trainer defaults including enable_progress_bar=False")
        
        # Call parent implementation to create trainer
        trainer = super().instantiate_trainer(**kwargs)
        
        # Store CLI reference for callbacks that need it (like ConfigEmbeddingMixin)
        trainer.cli = self
        logger.info("✅ Stored CLI reference in trainer for checkpoint compatibility")
        
        # Register TrainerConfigState manager for systematic config preservation
        self._register_trainer_config_state(trainer)
        
        return trainer
    
    def _process_environment_callback_config(self) -> None:
        """
        Process environment variables from config files EARLY before instantiation.
        
        This ensures variables like PYTORCH_CUDA_ALLOC_CONF are set before CUDA init.
        """
        try:
            # Extract config paths from command line arguments
            config_paths = self._extract_config_paths_from_sys_argv()
            
            if config_paths:
                logger.info(f"📋 Processing environment variables from {len(config_paths)} config files")
                
                # Extract and set environment variables early
                from ..utils.logging.environment_manager import EnvironmentManager
                env_vars, processed_configs = EnvironmentManager.extract_environment_from_configs(config_paths)
                
                if env_vars:
                    EnvironmentManager.set_environment_variables(env_vars, processed_configs)
                    logger.info(f"✅ Set {len(env_vars)} environment variables EARLY (before instantiation)")
                    
                    # Log critical environment variables for debugging
                    for var_name, value in env_vars.items():
                        if any(keyword in var_name.upper() for keyword in ['CUDA', 'ALLOC', 'MALLOC', 'PYTORCH']):
                            logger.info(f"   🎯 CRITICAL: {var_name}={value}")
                        else:
                            logger.debug(f"   {var_name}={value}")
                else:
                    logger.debug("No environment variables found in config files")
            else:
                logger.debug("No config files found for environment variable processing")
                
        except Exception as e:
            logger.warning(f"Failed to process environment variables early: {e}")
    
    def _extract_config_paths_from_sys_argv(self) -> list:
        """Extract config file paths from sys.argv for early processing."""
        config_paths = []
        i = 0
        while i < len(sys.argv):
            if sys.argv[i] == '--config' and i + 1 < len(sys.argv):
                config_path = Path(sys.argv[i + 1])
                if config_path.exists():
                    config_paths.append(config_path)
                i += 2
            elif sys.argv[i].startswith('--config='):
                config_path = Path(sys.argv[i].split('=', 1)[1])
                if config_path.exists():
                    config_paths.append(config_path)
                i += 1
            else:
                i += 1
        return config_paths

    def _register_trainer_config_state(self, trainer):
        """Register TrainerConfigState manager for systematic trainer config preservation."""
        try:
            from lightning_reflow.utils.checkpoint.manager_state import register_manager
            from lightning_reflow.utils.checkpoint.trainer_config_state import TrainerConfigState
            from lightning_reflow.utils.checkpoint.datamodule_state import DataModuleState
            
            # Create and register trainer config state manager
            trainer_config_state = TrainerConfigState(trainer)
            register_manager(trainer_config_state)
            
            # Create and register datamodule state manager
            datamodule_state = DataModuleState(trainer)
            register_manager(datamodule_state)
            
            logger.info("✅ Registered TrainerConfigState and DataModuleState managers for systematic state preservation")
        except Exception as e:
            logger.warning(f"⚠️ Failed to register state managers: {e}")
    
    def _is_resume_command(self) -> bool:
        """Check if the command is a resume command."""
        return len(sys.argv) > 1 and sys.argv[1] == RESUME_COMMAND
    
    def _is_fit_with_checkpoint(self) -> bool:
        """Check if this is a fit command with a checkpoint path."""
        is_fit_command = len(sys.argv) > 1 and sys.argv[1] == FIT_COMMAND
        has_ckpt_path = CHECKPOINT_PATH_ARG in sys.argv or any(a.startswith(f'{CHECKPOINT_PATH_ARG}=') for a in sys.argv)
        return is_fit_command and has_ckpt_path
    
    def _enable_config_overwrite(self, kwargs):
        """Enable config overwrite for fit commands with checkpoints."""
        if 'save_config_kwargs' not in kwargs:
            kwargs['save_config_kwargs'] = {}
        kwargs['save_config_kwargs']['overwrite'] = True
    
    def _execute_resume_as_subprocess(self) -> None:
        """Execute resume command by spawning a subprocess with fit command."""
        try:
            # Parse resume arguments
            parser = self._create_resume_parser()
            args, unknown_args = parser.parse_known_args(sys.argv[2:])
            
            # Validate arguments
            self._validate_resume_args(parser, args)
            
            # Prepare resume using LightningReflow
            resume_source = args.checkpoint_path or args.checkpoint_artifact
            checkpoint_path, embedded_config_yaml, wandb_run_id, temp_reflow = self._prepare_resume(args, resume_source)
            
            # Execute subprocess
            self._execute_fit_subprocess(args, unknown_args, checkpoint_path, embedded_config_yaml, wandb_run_id, temp_reflow)
            
        except Exception as e:
            logger.error(f"❌ Failed to execute resume command: {e}")
            sys.exit(1)
    
    def _create_resume_parser(self) -> argparse.ArgumentParser:
        """Create argument parser for resume command."""
        parser = argparse.ArgumentParser(
            prog="lightning-reflow resume",
            description="Resume Lightning Reflow training from a checkpoint"
        )
        
        parser.add_argument(
            "--checkpoint-path",
            type=str,
            help="Path to checkpoint file to resume from"
        )
        
        parser.add_argument(
            "--checkpoint-artifact", 
            type=str,
            help="W&B artifact to resume from (e.g., 'entity/project/run-id:latest')"
        )
        
        parser.add_argument(
            CONFIG_ARG, "-c",
            action="append",
            type=str,
            help="Path to one or more override configuration files. Can be specified multiple times."
        )
        
        parser.add_argument(
            "--use-wandb-config",
            action="store_true",
            help="Use configuration from W&B run (for W&B artifacts)"
        )
        
        parser.add_argument(
            "--entity",
            type=str,
            help="W&B entity (for artifact resumption)"
        )
        
        parser.add_argument(
            "--project", 
            type=str,
            help="W&B project (for artifact resumption)"
        )
        
        return parser
    
    def _validate_resume_args(self, parser, args):
        """Validate resume command arguments."""
        if args.checkpoint_path and args.checkpoint_artifact:
            parser.error("Cannot specify both --checkpoint-path and --checkpoint-artifact")
        
        if not args.checkpoint_path and not args.checkpoint_artifact:
            parser.error("Must specify either --checkpoint-path or --checkpoint-artifact")
    
    def _prepare_resume(self, args, resume_source):
        """Prepare resume by creating temp LightningReflow and getting strategy."""
        temp_reflow = LightningReflow(
            config_files=args.config if args.config else None,
            auto_configure_logging=False
        )
        
        # Use resume strategy to prepare checkpoint and config
        strategy = temp_reflow._select_resume_strategy(resume_source)
        checkpoint_path, embedded_config_yaml = strategy.prepare_resume(
            resume_source=resume_source,
            use_wandb_config=args.use_wandb_config,
            entity=args.entity,
            project=args.project
        )
        
        # Extract W&B run ID from checkpoint for proper resumption
        wandb_run_id = self._extract_wandb_run_id_from_checkpoint(checkpoint_path)
        
        logger.info(f"🔄 Preparing subprocess resume command")
        logger.info(f"   Checkpoint: {checkpoint_path}")
        if wandb_run_id:
            logger.info(f"   W&B Run ID: {wandb_run_id}")
        else:
            logger.info("   W&B Run ID: Not found in checkpoint")
        
        # Store the temp_reflow for cleanup AFTER subprocess completes
        return checkpoint_path, embedded_config_yaml, wandb_run_id, temp_reflow
    
    def _extract_wandb_run_id_from_checkpoint(self, checkpoint_path):
        """Extract W&B run ID from checkpoint file."""
        try:
            import torch
            from ..utils.checkpoint.checkpoint_utils import extract_wandb_run_id
            
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            run_id = extract_wandb_run_id(checkpoint)
            
            if run_id:
                logger.info(f"✅ Extracted W&B run ID from checkpoint: {run_id}")
                return run_id
            else:
                logger.info("ℹ️ No W&B run ID found in checkpoint metadata")
                return None
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to extract W&B run ID from checkpoint: {e}")
            return None
    
    def _execute_fit_subprocess(self, args, unknown_args, checkpoint_path, embedded_config_yaml, wandb_run_id=None, temp_reflow=None):
        """Execute the fit command in a subprocess."""
        # Build command for subprocess
        cmd = [sys.executable, '-m', 'lightning_reflow.cli', FIT_COMMAND]
        
        # Handle embedded config from checkpoint FIRST (preserves --config --ckpt_path order)
        temp_config_path = self._write_temp_config(embedded_config_yaml)
        
        try:
            # Add temp config as the BASE config file
            if temp_config_path:
                cmd.extend([CONFIG_ARG, temp_config_path])
                logger.info(f"📄 Using Lightning's original merged config from checkpoint as base")
            else:
                logger.info("📄 No embedded config found in checkpoint, resuming without it.")
            
            # Add any user-provided override configs
            if args.config:
                for config_file in args.config:
                    cmd.extend([CONFIG_ARG, config_file])
                logger.info(f"🔧 Applying override configs: {args.config}")
            
            # Add checkpoint path LAST so it overrides any ckpt_path in configs
            cmd.extend([CHECKPOINT_PATH_ARG, str(checkpoint_path)])
            
            # Configure W&B logger for run resumption
            if wandb_run_id:
                self._add_wandb_resume_config(cmd, wandb_run_id, embedded_config_yaml)
            else:
                logger.info("ℹ️ No W&B run ID found - will create new W&B run")
            
            # Pass through any additional Lightning CLI arguments
            if unknown_args:
                cmd.extend(unknown_args)
                logger.info(f"🔧 Passing through additional arguments: {unknown_args}")
            
            logger.info(f"🚀 Executing: {' '.join(cmd)}")
            
            # Execute the fit command in subprocess
            result = subprocess.run(cmd, check=True)
            sys.exit(result.returncode)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Subprocess failed with return code {e.returncode}")
            sys.exit(e.returncode)
        finally:
            # Cleanup temp config file
            self._cleanup_temp_config(temp_config_path)
            
            # Cleanup temp W&B config file if created
            if hasattr(self, '_temp_wandb_config_path'):
                self._cleanup_temp_config(self._temp_wandb_config_path)
                delattr(self, '_temp_wandb_config_path')
            
            # Cleanup temp reflow strategies AFTER subprocess completes
            if temp_reflow:
                try:
                    temp_reflow._cleanup_strategies()
                    logger.info(f"🗑️ Cleaned up temporary reflow strategies")
                except Exception:
                    pass
    
    def _write_temp_config(self, embedded_config_yaml):
        """Write embedded config YAML to temporary file."""
        if not embedded_config_yaml:
            return None
            
        try:
            temp_config_fd, temp_config_path = tempfile.mkstemp(
                suffix=TEMP_CONFIG_SUFFIX, 
                prefix=TEMP_CONFIG_PREFIX
            )
            with os.fdopen(temp_config_fd, 'w') as f:
                f.write(embedded_config_yaml)
            return temp_config_path
        except Exception as e:
            logger.error(f"Failed to create temporary config file: {e}")
            return None
    
    def _cleanup_temp_config(self, temp_config_path):
        """Clean up temporary config file."""
        if temp_config_path and os.path.exists(temp_config_path):
            try:
                os.unlink(temp_config_path)
                logger.info(f"🗑️ Cleaned up temporary config: {temp_config_path}")
            except Exception:
                pass
    
    def _add_wandb_resume_config(self, cmd, wandb_run_id, embedded_config_yaml):
        """Add W&B logger configuration for resuming a run."""
        try:
            # Parse embedded config to check existing logger configuration
            existing_config = {}
            if embedded_config_yaml:
                existing_config = yaml.safe_load(embedded_config_yaml) or {}
            
            # Check if there's already a logger configured
            trainer_config = existing_config.get('trainer', {})
            existing_logger = trainer_config.get('logger', None)
            
            # Prepare W&B logger config
            if isinstance(existing_logger, dict) and existing_logger.get('class_path', '').endswith('WandbLogger'):
                # Update existing W&B logger config
                logger.info("📝 Updating existing W&B logger configuration for resume")
                if 'init_args' not in existing_logger:
                    existing_logger['init_args'] = {}
                existing_logger['init_args']['id'] = wandb_run_id
                existing_logger['init_args']['resume'] = 'allow'
                wandb_config = {'trainer': {'logger': existing_logger}}
            else:
                # Create new W&B logger config
                logger.info("📝 Creating new W&B logger configuration for resume")
                wandb_logger_config = {
                    'class_path': 'lightning.pytorch.loggers.WandbLogger',
                    'init_args': {
                        'id': wandb_run_id,
                        'resume': 'allow',
                        'log_model': False  # Don't log models by default during resume
                    }
                }
                wandb_config = {'trainer': {'logger': wandb_logger_config}}
            
            # Write config to temporary file
            wandb_config_yaml = yaml.dump(wandb_config)
            temp_wandb_config_fd, temp_wandb_config_path = tempfile.mkstemp(
                suffix='.yaml', 
                prefix='wandb_logger_config_'
            )
            
            with os.fdopen(temp_wandb_config_fd, 'w') as f:
                f.write(wandb_config_yaml)
            
            # Add as config file (will be merged with others)
            cmd.extend([CONFIG_ARG, temp_wandb_config_path])
            logger.info(f"🔄 Configuring W&B logger to resume run: {wandb_run_id}")
            
            # Store path for cleanup
            self._temp_wandb_config_path = temp_wandb_config_path
            
        except Exception as e:
            logger.error(f"❌ Failed to create W&B logger config: {e}")
            if 'temp_wandb_config_path' in locals() and os.path.exists(temp_wandb_config_path):
                os.unlink(temp_wandb_config_path)


def main():
    """Main entry point for the lightning-reflow CLI."""
    cli = LightningReflowCLI()


if __name__ == "__main__":
    main() 