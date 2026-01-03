"""
Lightning Reflow CLI - Command-line interface wrapper.

This module provides a thin CLI wrapper around the core LightningReflow class,
handling command-line argument parsing and translating them to method calls.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from lightning.pytorch.cli import LightningCLI
from typing import Optional, Type, Any

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
    
    def __init__(self, *args, parser_kwargs=None, **kwargs):
        """
        Initialize the Lightning Reflow CLI.

        For resume commands, spawns a new subprocess with the fit command.

        Args:
            parser_kwargs: Optional dict of kwargs to pass to jsonargparse ArgumentParser.
                          Can be used to configure parser behavior (e.g., less strict validation).
            **kwargs: Additional arguments passed to LightningCLI.
        """
        logger.info("🎯 Initializing Lightning Reflow CLI")

        # Store model_class and datamodule_class from positional args for _dump_config fix
        # LightningCLI signature: __init__(model_class, datamodule_class=None, ...)
        self._user_model_class = args[0] if len(args) > 0 else kwargs.get('model_class')
        self._user_datamodule_class = args[1] if len(args) > 1 else kwargs.get('datamodule_class')

        # CRITICAL: Register safe globals BEFORE any checkpoint loading
        # Lightning CLI's _parse_ckpt_path() loads checkpoints with weights_only=True
        # This must happen before super().__init__() to allow numpy/torch objects in checkpoints
        self._register_checkpoint_safe_globals()

        # Handle resume command by spawning subprocess with fit command
        if self._is_resume_command():
            self._execute_resume_as_subprocess()
            return

        # For ALL fit commands, enable config overwrite to handle stale configs gracefully
        # This prevents Lightning from aborting when logs/config.yaml exists from previous runs
        if self._is_fit_command():
            logger.info("🔧 Enabling config overwrite for fit command (handles stale configs)")
            self._enable_config_overwrite(kwargs)

        # Configure parser kwargs for less strict validation if requested
        if parser_kwargs is not None:
            # Merge user-provided parser_kwargs with any existing ones
            existing_parser_kwargs = kwargs.get('parser_kwargs', {})
            merged_parser_kwargs = {**existing_parser_kwargs, **parser_kwargs}
            kwargs['parser_kwargs'] = merged_parser_kwargs
            logger.info(f"🔧 Using custom parser configuration: {merged_parser_kwargs}")

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

    def _dump_config(self) -> None:
        """
        Override to fix class_path when classes are provided programmatically.

        CRITICAL FIX: When subclass_mode_model=True with classes provided programmatically,
        Lightning CLI saves the base class path (e.g., 'lightning.LightningModule') instead
        of the actual subclass. This breaks checkpoint resume because it tries to instantiate
        the base class with subclass-specific parameters.

        This method generically fixes class_path for model, datamodule, and any other
        class parameters that may be added in the future.
        """
        # Call parent to generate config_dump
        super()._dump_config()

        # Generic fix for all class parameters
        # Maps config_key → (user_class, expected_base_class_path)
        class_fixes = [
            ('model', self._user_model_class, 'lightning.LightningModule'),
            ('data', self._user_datamodule_class, 'lightning.LightningDataModule'),
            # Future: Add trainer_class, save_config_callback, etc. here
        ]

        for config_key, user_class, expected_base_path in class_fixes:
            self._fix_class_path_if_needed(config_key, user_class, expected_base_path)

    def _fix_class_path_if_needed(
        self,
        config_key: str,
        user_class: Optional[Type[Any]],
        expected_base_path: str
    ) -> None:
        """
        Fix class_path in config_dump for a given config section.

        Args:
            config_key: The config section key ('model', 'data', etc.)
            user_class: The actual class provided by the user (None if not provided)
            expected_base_path: The base class path that Lightning incorrectly saves
        """
        if user_class is None or not hasattr(self, 'config_dump'):
            return

        if config_key not in self.config_dump:
            return

        section = self.config_dump[config_key]
        if not isinstance(section, dict):
            return

        # Get the correct class_path from the user-provided class
        correct_class_path = f"{user_class.__module__}.{user_class.__name__}"
        current_class_path = section.get('class_path')

        # Fix if needed
        if current_class_path == expected_base_path:
            logger.info(f"🔧 Fixing {config_key}.class_path: '{current_class_path}' → '{correct_class_path}'")
            section['class_path'] = correct_class_path
        elif current_class_path and current_class_path != correct_class_path:
            logger.warning(f"⚠️ Unexpected {config_key}.class_path: '{current_class_path}' (expected '{correct_class_path}')")
            logger.info(f"🔧 Fixing {config_key}.class_path: '{current_class_path}' → '{correct_class_path}'")
            section['class_path'] = correct_class_path
        else:
            logger.debug(f"✅ {config_key}.class_path already correct: '{correct_class_path}'")
    
    def before_fit(self) -> None:
        """Hook called before fit starts - add essential callbacks."""
        self._add_essential_callbacks()
        # Proactively create checkpoint directories to avoid FileNotFoundError
        self._ensure_checkpoint_directories()
    
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

    def _parse_ckpt_path(self) -> None:
        """
        Override Lightning CLI's checkpoint hyperparameter parsing.

        LightningReflow uses the embedded config (self_contained_metadata) as the
        authoritative source for model configuration during resume. This approach:

        1. Avoids format mismatches between ConfigMixin and jsonargparse
        2. Handles complex nested objects (like dynamics_model) correctly
        3. Prevents partial hparams from overwriting complete config values

        The embedded config is extracted by the resume command and passed via
        --config, so the full model specification is already available. We skip
        checkpoint hparam parsing entirely to avoid conflicts.

        Note: This is a deliberate design choice. The embedded config mechanism
        provides a more robust and complete solution than checkpoint hyperparameters,
        which are inherently partial (nn.Module instances can't be serialized to hparams).
        """
        # Skip checkpoint hparam parsing entirely.
        # The embedded config (extracted during resume) provides the complete
        # model configuration including dynamics_model with class_path + init_args.
        # Merging partial checkpoint hparams would overwrite these with None values.
        logger.debug("Skipping checkpoint hparam parsing - using embedded config instead")

    def _ensure_checkpoint_directories(self) -> None:
        """Ensure checkpoint directories exist for ModelCheckpoint and default root.

        Lightning's atomic save via fsspec may not auto-create parent directories.
        This method creates them ahead of time to prevent save errors like
        FileNotFoundError for 'checkpoints/last.ckpt'.
        """
        try:
            if not hasattr(self, 'trainer') or not self.trainer:
                return

            # Ensure trainer.default_root_dir exists
            try:
                default_root = getattr(self.trainer, 'default_root_dir', None)
                if default_root:
                    Path(default_root).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.debug(f"Could not create default_root_dir '{default_root}': {e}")

            # Ensure each ModelCheckpoint.dirpath exists
            try:
                from lightning.pytorch.callbacks import ModelCheckpoint
                for cb in (self.trainer.callbacks or []):
                    if isinstance(cb, ModelCheckpoint):
                        dirpath = getattr(cb, 'dirpath', None)
                        if dirpath:
                            Path(dirpath).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.debug(f"Could not ensure ModelCheckpoint directories: {e}")

        except Exception as e:
            logger.warning(f"Failed to ensure checkpoint directories: {e}")
    
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
                env_vars, processed_files = EnvironmentManager.extract_environment_from_configs(config_paths)
                
                if env_vars:
                    EnvironmentManager.set_environment_variables(env_vars, processed_files)
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
    
    def _register_checkpoint_safe_globals(self) -> None:
        """
        Register safe globals for torch.load with weights_only=True.

        MUST be called BEFORE super().__init__() because Lightning CLI's _parse_ckpt_path()
        loads checkpoints with weights_only=True, which will fail if numpy or torch
        version objects are present.

        Registers:
        - numpy arrays and dtypes (for checkpoint data)
        - torch.torch_version.TorchVersion (Lightning stores pytorch version in checkpoints)

        This is safe for our own checkpoints that may contain these objects.
        """
        try:
            import numpy as np
            import torch

            # Collect all numpy dtype classes for safe loading
            numpy_dtypes = []
            if hasattr(np, 'dtypes'):
                numpy_dtypes = [getattr(np.dtypes, attr) for attr in dir(np.dtypes) if 'DType' in attr]

            # Register safe globals for numpy objects
            safe_globals = [
                np._core.multiarray._reconstruct,
                np.ndarray,
                np.dtype,
            ] + numpy_dtypes

            # Register torch version class - Lightning stores this in checkpoints
            # as 'pytorch-lightning_version' which uses TorchVersion type
            if hasattr(torch, 'torch_version') and hasattr(torch.torch_version, 'TorchVersion'):
                safe_globals.append(torch.torch_version.TorchVersion)

            torch.serialization.add_safe_globals(safe_globals)
            logger.debug(f"✅ Registered {len(safe_globals)} safe globals for checkpoint loading")

        except Exception as e:
            # Don't fail initialization if this doesn't work
            logger.warning(f"⚠️  Could not register safe globals: {e}")

    def _is_resume_command(self) -> bool:
        """Check if the command is a resume command."""
        return len(sys.argv) > 1 and sys.argv[1] == RESUME_COMMAND

    def _is_fit_command(self) -> bool:
        """Check if this is a fit command (with or without checkpoint)."""
        return len(sys.argv) > 1 and sys.argv[1] == 'fit'

    def _is_fit_with_checkpoint(self) -> bool:
        """Check if this is a fit command with a checkpoint path."""
        is_fit_command = self._is_fit_command()
        has_ckpt_path = '--ckpt_path' in sys.argv or any(a.startswith('--ckpt_path=') for a in sys.argv)
        return is_fit_command and has_ckpt_path
    
    def _enable_config_overwrite(self, kwargs):
        """Enable config overwrite for fit commands and warn if overwriting stale config."""
        # Check if we're about to overwrite an existing config
        try:
            # Try to determine log directory from command line args
            log_dir = self._get_log_dir_from_args()
            if log_dir:
                config_path = Path(log_dir) / 'config.yaml'
                if config_path.exists():
                    config_age = time.time() - config_path.stat().st_mtime
                    logger.warning(
                        f"⚠️  Overwriting stale config.yaml from previous run\n"
                        f"   Path: {config_path}\n"
                        f"   Age: {config_age:.1f} seconds ({config_age/3600:.1f} hours)\n"
                        f"   This is normal and safe - a fresh config will be saved."
                    )
        except Exception as e:
            # Don't fail if we can't check - this is just a warning
            logger.debug(f"Could not check for stale config: {e}")

        if 'save_config_kwargs' not in kwargs:
            kwargs['save_config_kwargs'] = {}
        kwargs['save_config_kwargs']['overwrite'] = True

    def _get_log_dir_from_args(self) -> Optional[Path]:
        """Extract log directory from command line arguments."""
        try:
            # Look for --trainer.default_root_dir or similar in sys.argv
            for i, arg in enumerate(sys.argv):
                if 'default_root_dir' in arg:
                    if '=' in arg:
                        # Format: --trainer.default_root_dir=logs
                        return Path(arg.split('=')[1])
                    elif i + 1 < len(sys.argv):
                        # Format: --trainer.default_root_dir logs
                        return Path(sys.argv[i + 1])

            # Default fallback
            return Path('logs')
        except Exception:
            return None
    
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
                wandb_run_id=args.wandb_run_id if hasattr(args, 'wandb_run_id') else None,
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
        
        parser.add_argument(
            "--wandb-run-id",
            nargs='?',
            const='new',  # Value when flag is present without argument
            default=None,  # Value when flag is not present at all
            help="W&B run ID control: Provide ID to resume specific run, use flag alone to force new run, omit to use checkpoint's ID."
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