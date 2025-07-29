"""
Lightning Reflow CLI - Command-line interface wrapper.

This module provides a thin CLI wrapper around the core LightningReflow class,
handling command-line argument parsing and translating them to method calls.
"""

import argparse
import logging
import sys
from pathlib import Path
from lightning.pytorch.cli import LightningCLI

from ..core import LightningReflow

logger = logging.getLogger(__name__)

# Constants
RESUME_COMMAND = 'resume'
CONFIG_ARG = '--config'

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
        is_fit_command = len(sys.argv) > 1 and sys.argv[1] == 'fit'
        has_ckpt_path = '--ckpt_path' in sys.argv or any(a.startswith('--ckpt_path=') for a in sys.argv)
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
            
            # Create temporary LightningReflow instance and use its resume_cli method
            resume_source = args.checkpoint_path or args.checkpoint_artifact
            temp_reflow = LightningReflow(
                config_files=args.config if args.config else None,
                auto_configure_logging=False
            )
            
            # Use the consolidated resume_cli method
            temp_reflow.resume_cli(
                resume_source=resume_source,
                config_overrides=args.config,
                use_wandb_config=args.use_wandb_config,
                entity=args.entity,
                project=args.project,
                extra_cli_args=unknown_args
            )
            
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
    



def main():
    """Main entry point for the lightning-reflow CLI."""
    cli = LightningReflowCLI()


if __name__ == "__main__":
    main() 