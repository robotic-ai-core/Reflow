"""
Lightning Reflow CLI - Command-line interface wrapper.

This module provides a thin CLI wrapper around the core LightningReflow class,
handling command-line argument parsing and translating them to method calls.
"""

import logging
from typing import Optional, Dict, Any, List
from lightning.pytorch.cli import LightningCLI

from ..core import LightningReflow

logger = logging.getLogger(__name__)


class LightningReflowCLI(LightningCLI):
    """
    Command-line interface for Lightning Reflow.
    
    This class extends Lightning's CLI with Lightning Reflow's advanced features
    like pause/resume, W&B integration, and sophisticated configuration management.
    
    It acts as a thin wrapper around the core LightningReflow class, parsing
    command-line arguments and translating them to the appropriate method calls.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the Lightning Reflow CLI.
        
        This sets up argument parsing and handles the special 'resume' subcommand
        that doesn't exist in the base Lightning CLI.
        """
        logger.info("🎯 Initializing Lightning Reflow CLI")
        
        # Check if this is a resume command before Lightning CLI processes it
        if self._is_resume_command():
            self._handle_resume_command()
            return
        
        # Initialize the Lightning CLI for standard commands (fit, validate, test, predict)
        super().__init__(*args, **kwargs)
    
    def _is_resume_command(self) -> bool:
        """Check if the command is a resume command."""
        import sys
        return len(sys.argv) > 1 and sys.argv[1] == 'resume'
    
    def _handle_resume_command(self) -> None:
        """Handle the resume subcommand."""
        import argparse
        import sys
        
        # Create argument parser for resume command
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
            type=str,
            help="Path to configuration file"
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
        args = parser.parse_args(sys.argv[2:])
        
        # Determine resume source
        if args.checkpoint_path and args.checkpoint_artifact:
            parser.error("Cannot specify both --checkpoint-path and --checkpoint-artifact")
        
        if not args.checkpoint_path and not args.checkpoint_artifact:
            parser.error("Must specify either --checkpoint-path or --checkpoint-artifact")
        
        resume_source = args.checkpoint_path or args.checkpoint_artifact
        
        # Create LightningReflow instance
        config_files = [args.config] if args.config else None
        
        reflow = LightningReflow(
            config_files=config_files,
            auto_configure_logging=True
        )
        
        # Resume training
        logger.info(f"Resuming training from: {resume_source}")
        
        try:
            result = reflow.resume(
                resume_source=resume_source,
                use_wandb_config=args.use_wandb_config,
                entity=args.entity,
                project=args.project
            )
            logger.info("✅ Resume completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Resume failed: {e}")
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