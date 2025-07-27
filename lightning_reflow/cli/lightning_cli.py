"""
Lightning Reflow CLI - Command-line interface wrapper.

This module provides a thin CLI wrapper around the core LightningReflow class,
handling command-line argument parsing and translating them to method calls.
"""

import logging
import subprocess
import sys
from lightning.pytorch.cli import LightningCLI

from ..core import LightningReflow

logger = logging.getLogger(__name__)


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
        is_fit_command = len(sys.argv) > 1 and sys.argv[1] == 'fit'
        has_ckpt_path = '--ckpt_path' in sys.argv or any(a.startswith('--ckpt_path=') for a in sys.argv)
        
        if is_fit_command and has_ckpt_path:
            logger.info("🔧 Detected fit from checkpoint, enabling config overwrite.")
            if 'save_config_kwargs' not in kwargs:
                kwargs['save_config_kwargs'] = {}
            kwargs['save_config_kwargs']['overwrite'] = True
        
        # Initialize the Lightning CLI for standard commands (fit, validate, test, predict)
        super().__init__(*args, **kwargs)
    
    def before_fit(self) -> None:
        """Hook called before fit starts - add essential callbacks."""
        self._add_essential_callbacks()
    
    def _add_essential_callbacks(self) -> None:
        """Add essential callbacks if not already present."""
        if not hasattr(self, 'trainer') or not self.trainer:
            return
            
        self._add_step_output_logger()
        self._add_pause_callback()
    
    def _add_pause_callback(self) -> None:
        """Add PauseCallback if not already present."""
        from ..callbacks.pause import PauseCallback
        
        has_pause_callback = any(isinstance(cb, PauseCallback) for cb in self.trainer.callbacks)
        if not has_pause_callback:
            pause_callback = PauseCallback(
                checkpoint_dir='pause_checkpoints',
                enable_pause=True,
                pause_key='p',
                upload_key='w',
                debounce_interval=0.3,
                refresh_rate=1,
                bar_colour='#fcac17',
                global_bar_metrics=['*lr*'],
                interval_bar_metrics=['loss', 'train/loss', 'train_loss'],
                logging_interval='step',
            )
            self.trainer.callbacks.append(pause_callback)
            logger.info("✅ Added PauseCallback for progress bar functionality")
    
    def _add_step_output_logger(self) -> None:
        """Add StepOutputLoggerCallback if not already present."""
        from ..callbacks.logging import StepOutputLoggerCallback
        
        has_step_logger = any(isinstance(cb, StepOutputLoggerCallback) for cb in self.trainer.callbacks)
        if not has_step_logger:
            step_logger = StepOutputLoggerCallback(
                train_prog_bar_metrics=['loss', 'train/loss'],
                val_prog_bar_metrics=['val_loss', 'val/val_loss']
            )
            self.trainer.callbacks.append(step_logger)
            logger.info("✅ Added StepOutputLoggerCallback for metrics logging")
    
    def instantiate_trainer(self, **kwargs):
        """Override to set CLI reference in trainer for ConfigEmbeddingMixin compatibility."""
        # Call parent implementation to create trainer
        trainer = super().instantiate_trainer(**kwargs)
        
        # Store CLI reference for callbacks that need it (like ConfigEmbeddingMixin)
        trainer.cli = self
        logger.info("✅ Stored CLI reference in trainer for checkpoint compatibility")
        
        # Register TrainerConfigState manager for systematic config preservation
        self._register_trainer_config_state(trainer)
        
        return trainer
    
    def _register_trainer_config_state(self, trainer):
        """Register TrainerConfigState manager for systematic trainer config preservation."""
        try:
            from modules.utils.checkpoint.manager_state import register_manager
            from modules.utils.checkpoint.trainer_config_state import TrainerConfigState
            from modules.utils.checkpoint.datamodule_state import DataModuleState
            
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
        import sys
        return len(sys.argv) > 1 and sys.argv[1] == 'resume'
    
    def _execute_resume_as_subprocess(self) -> None:
        """Execute resume command by spawning a subprocess with fit command."""
        import argparse
        import tempfile
        import yaml
        import os
        
        # Parse resume arguments first
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
            "--config", "-c",
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
        
        # Parse resume arguments (skip 'resume' subcommand)
        args, unknown_args = parser.parse_known_args(sys.argv[2:])
        
        # Validate arguments
        if args.checkpoint_path and args.checkpoint_artifact:
            parser.error("Cannot specify both --checkpoint-path and --checkpoint-artifact")
        
        if not args.checkpoint_path and not args.checkpoint_artifact:
            parser.error("Must specify either --checkpoint-path or --checkpoint-artifact")
        
        resume_source = args.checkpoint_path or args.checkpoint_artifact
        
        try:
            # Create temporary LightningReflow to handle resume preparation
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
            
            logger.info(f"🔄 Preparing subprocess resume command")
            logger.info(f"   Checkpoint: {checkpoint_path}")
            
            # Build command for subprocess
            cmd = [sys.executable, '-m', 'lightning_reflow.cli.lightning_cli', 'fit']
            
            # Handle embedded config from checkpoint FIRST
            temp_config_path = None
            if embedded_config_yaml:
                # Write embedded config YAML string to temporary file
                temp_config_fd, temp_config_path = tempfile.mkstemp(suffix='.yaml', prefix='resume_config_')
                try:
                    with os.fdopen(temp_config_fd, 'w') as f:
                        f.write(embedded_config_yaml)
                    
                    # Add temp config as the BASE config file
                    cmd.extend(['--config', temp_config_path])
                    logger.info(f"📄 Using Lightning's original merged config from checkpoint as base")
                    
                except Exception as e:
                    logger.error(f"Failed to create temporary config file: {e}")
                    if temp_config_path:
                        os.unlink(temp_config_path)
                        temp_config_path = None
            else:
                logger.info("📄 No embedded config found in checkpoint, resuming without it.")
            
            # Add any user-provided override configs
            if args.config:
                for config_file in args.config:
                    cmd.extend(['--config', config_file])
                logger.info(f"🔧 Applying override configs: {args.config}")
            
            # Add checkpoint path LAST so it overrides any ckpt_path in configs
            cmd.extend(['--ckpt_path', str(checkpoint_path)])
            
            # Pass through any additional Lightning CLI arguments
            if unknown_args:
                cmd.extend(unknown_args)
                logger.info(f"🔧 Passing through additional arguments: {unknown_args}")
            
            logger.info(f"🚀 Executing: {' '.join(cmd)}")
            
            # Execute the fit command in subprocess
            try:
                result = subprocess.run(cmd, check=True)
                sys.exit(result.returncode)
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Subprocess failed with return code {e.returncode}")
                sys.exit(e.returncode)
            finally:
                # Cleanup temporary config file
                if temp_config_path and os.path.exists(temp_config_path):
                    try:
                        os.unlink(temp_config_path)
                        logger.info(f"🗑️ Cleaned up temporary config: {temp_config_path}")
                    except Exception:
                        pass
                
                # Cleanup temp reflow strategies
                try:
                    temp_reflow._cleanup_strategies()
                    logger.info(f"🗑️ Cleaned up temporary reflow strategies")
                except Exception:
                    pass
            
        except Exception as e:
            logger.error(f"❌ Failed to execute resume command: {e}")
            sys.exit(1)
    
    
    def add_arguments_to_parser(self, parser) -> None:
        """Add Lightning Reflow specific arguments to the parser."""
        super().add_arguments_to_parser(parser)
        
        # Add custom Lightning Reflow arguments
        parser.add_argument(
            "--wandb-project",
            type=str,
            default="",
            help="W&B project name. Overrides project name in logger config."
        )
        
        parser.add_argument(
            "--wandb-log-model",
            type=bool,
            default=None,
            help="Custom alias to control wandb model logging."
        )
        
        parser.add_argument(
            "--resume-from-wandb",
            type=str,
            default=None,
            help="Resume training from a W&B checkpoint artifact."
        )
        
        parser.add_argument(
            "--weights-only",
            action="store_true",
            help="Load only model weights from checkpoint, not training state."
        )
        
        parser.add_argument(
            "--use-wandb-config",
            action="store_true", 
            help="Use the config file saved in the W&B artifact."
        )
        
        parser.add_argument(
            "--auto-resume-wandb",
            action="store_true",
            default=True,
            help="Automatically detect wandb run ID from checkpoint path."
        )
        
        parser.add_argument(
            "--disable-pause-exit",
            action="store_true",
            help="Disable validation-boundary pause functionality."
        )
        
        parser.add_argument(
            "--wandb-run",
            type=str,
            default=None,
            help="W&B run ID to resume"
        )


def main():
    """Main entry point for the lightning-reflow CLI."""
    cli = LightningReflowCLI()


if __name__ == "__main__":
    main() 