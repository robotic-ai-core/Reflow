"""
Resume Command Handler for DiffusionFlow Training

This module centralizes all resume-related functionality that was previously
scattered across the CLI class. It provides a clean interface for handling
different resume methods (artifact, checkpoint, config file).
"""

import sys
import argparse
import json
import subprocess
import warnings
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import torch

class ResumeCommandHandler:
    """Handles all resume command functionality."""
    
    def __init__(self, cli_instance):
        """Initialize with reference to CLI instance for access to helper methods."""
        self.cli = cli_instance
        self._wandb_checkpoint_path: Optional[str] = None
        self._detected_wandb_run_id: Optional[str] = None
    
    def should_handle_resume_manually(self) -> bool:
        """Check if we should handle resume manually vs Lightning CLI integration."""
        return len(sys.argv) > 1 and sys.argv[1] == 'resume'
    
    def execute_resume_subcommand(self) -> None:
        """Execute the resume subcommand with unified argument parsing."""
        resume_args, extra_args = self._parse_resume_args()
        
        
        # Validate arguments
        self._validate_resume_args(resume_args)
        
        # Handle validation only mode
        if resume_args.validate:
            success = self._validate_resume_request(resume_args)
            sys.exit(0 if success else 1)
        
        # Show info about extra args if provided
        if extra_args:
            print(f"📋 Additional CLI overrides detected: {' '.join(extra_args)}")
            print(f"   These will be forwarded to the training command")
        
        # Execute resume based on method
        success = self._execute_resume_by_method(resume_args, extra_args)
        sys.exit(0 if success else 1)
    
    def _parse_resume_args(self):
        """Parse resume command arguments."""
        parser = argparse.ArgumentParser(
            prog=f"{sys.argv[0]} resume",
            description="Resume training from a configuration file",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Basic resume (uses embedded config)
  python train_lightning.py resume --checkpoint-artifact entity/project/run-abc123-pause:latest
  python train_lightning.py resume --checkpoint-path pause_checkpoints/checkpoint.ckpt
  
  # Partial config overrides (merged with embedded config)
  python train_lightning.py resume --checkpoint-path checkpoint.ckpt --config override.yaml
  python train_lightning.py resume --checkpoint-path checkpoint.ckpt --config base_override.yaml --config specific_override.yaml
  
  # Mixed overrides (files + CLI args)
  python train_lightning.py resume --checkpoint-path checkpoint.ckpt --config override.yaml --model.backbone.hidden_size=768
  
  # Other options
  python train_lightning.py resume --checkpoint-path checkpoint.ckpt --new-wandb-id my-new-run-id
  python train_lightning.py resume --validate --checkpoint-path checkpoint.ckpt
  python train_lightning.py resume --dry-run --checkpoint-artifact entity/project/run-abc123-pause:latest
            """
        )
        
        
        
        parser.add_argument(
            '--validate',
            action='store_true',
            help='Validate resume configuration without executing'
        )
        
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be executed without running it'
        )
        
        parser.add_argument(
            '--checkpoint-artifact',
            type=str,
            help='W&B checkpoint artifact to resume from (e.g., entity/project/run-id-latest:v0). Uses embedded config or specify --config to override.'
        )
        
        parser.add_argument(
            '--checkpoint-path',
            type=Path,
            help='Local checkpoint file path (alternative to --checkpoint-artifact). Uses embedded config or specify --config to override.'
        )
        
        parser.add_argument(
            '--wandb_run', '--wandb-run',
            type=str,
            help='W&B run ID to resume (equivalent to --trainer.logger.init_args.id)'
        )
        
        parser.add_argument(
            '--new-wandb-id',
            type=str,
            help='Create new W&B run with this ID instead of resuming the original run'
        )
        
        parser.add_argument(
            '--config',
            type=Path,
            action='append',
            dest='config_overrides',
            help='Config override file(s). Can be specified multiple times for partial overrides. Later configs override earlier ones. (works with --checkpoint-artifact or --checkpoint-path)'
        )
        
        return parser.parse_known_args(sys.argv[2:])
    
    def _validate_resume_args(self, resume_args) -> None:
        """Validate resume arguments."""
        # Need either checkpoint_artifact or checkpoint_path
        if not any([resume_args.checkpoint_artifact, resume_args.checkpoint_path]):
            raise argparse.ArgumentError(None, "Must specify either --checkpoint-artifact or --checkpoint-path")
        
        # Ensure only one main resume method is specified
        specified_methods = sum(bool(x) for x in [resume_args.checkpoint_artifact, resume_args.checkpoint_path])
        if specified_methods > 1:
            raise argparse.ArgumentError(None, "Cannot specify both --checkpoint-artifact and --checkpoint-path")
        
        # Ensure --wandb-run and --new-wandb-id are mutually exclusive
        if resume_args.wandb_run and resume_args.new_wandb_id:
            raise argparse.ArgumentError(None, "Cannot specify both --wandb-run and --new-wandb-id")
    
    def _validate_resume_request(self, resume_args) -> bool:
        """Validate resume configuration without executing."""
        if resume_args.checkpoint_artifact:
            validation_result = self.cli._validate_checkpoint_artifact(resume_args.checkpoint_artifact)
        elif resume_args.checkpoint_path:
            validation_result = self.cli._validate_checkpoint_path(resume_args.checkpoint_path)
        
        # Also validate config override files if provided
        if hasattr(resume_args, 'config_overrides') and resume_args.config_overrides:
            for config_file in resume_args.config_overrides:
                if not config_file.exists():
                    print(f"\n❌ Config override file not found: {config_file}")
                    return False
                print(f"📋 Config override file validated: {config_file}")
        
        if validation_result['valid']:
            print(f"\n✅ Resume configuration is valid and ready to use")
            if hasattr(resume_args, 'config_overrides') and resume_args.config_overrides:
                print(f"📝 Will use {len(resume_args.config_overrides)} config override file(s)")
            else:
                print(f"📝 Will use embedded config only (no overrides)")
            return True
        else:
            print(f"\n❌ Resume configuration validation failed")
            return False
    
    def _execute_resume_by_method(self, resume_args, extra_args: List[str]) -> bool:
        """Execute resume based on the specified method."""
        # Determine the wandb run ID to use
        wandb_run_override = resume_args.wandb_run or resume_args.new_wandb_id
        
        if resume_args.checkpoint_artifact:
            return self._execute_resume_from_artifact(
                resume_args.checkpoint_artifact, 
                wandb_run_override=wandb_run_override,
                config_overrides=resume_args.config_overrides,
                dry_run=resume_args.dry_run,
                extra_cli_args=extra_args
            )
        elif resume_args.checkpoint_path:
            return self._execute_resume_from_checkpoint(
                resume_args.checkpoint_path,
                wandb_run_override=wandb_run_override,
                config_overrides=resume_args.config_overrides,
                dry_run=resume_args.dry_run,
                extra_cli_args=extra_args
            )
        else:
            raise ValueError("No valid resume method specified")
    
    
    def _execute_resume_from_artifact(self, artifact_path: str, wandb_run_override: str = None, 
                                    config_overrides: List[Path] = None, dry_run: bool = False, 
                                    extra_cli_args: List[str] = None) -> bool:
        """Resume training from W&B checkpoint artifact."""
        return self.cli._execute_resume_from_artifact(
            artifact_path, wandb_run_override, config_overrides, dry_run, extra_cli_args
        )
    
    def _execute_resume_from_checkpoint(self, checkpoint_path: Path, wandb_run_override: str = None,
                                      config_overrides: List[Path] = None, dry_run: bool = False,
                                      extra_cli_args: List[str] = None) -> bool:
        """Resume training from local checkpoint file."""
        return self.cli._execute_resume_from_checkpoint(
            checkpoint_path, wandb_run_override, config_overrides, dry_run, extra_cli_args
        )
    
    


class WandbRunUtils:
    """Utilities for W&B run ID extraction and validation."""
    
    @staticmethod
    def extract_run_id(source: Union[str, Path]) -> Optional[str]:
        """Extract W&B run ID from various sources."""
        if isinstance(source, Path):
            # Extract from file path
            return WandbRunUtils._extract_from_path(str(source))
        elif isinstance(source, str):
            # Extract from artifact path or run ID string
            if '/' in source and ':' in source:
                # Artifact path format: entity/project/run-id-latest:v0
                return WandbRunUtils._extract_from_artifact_path(source)
            else:
                # Direct run ID
                return source
        return None
    
    @staticmethod
    def _extract_from_path(path: str) -> Optional[str]:
        """Extract run ID from file path."""
        import re
        # Look for run ID pattern in path
        match = re.search(r'run-([a-zA-Z0-9]{8})', path)
        return match.group(1) if match else None
    
    @staticmethod
    def _extract_from_artifact_path(artifact_path: str) -> Optional[str]:
        """Extract run ID from W&B artifact path."""
        import re
        # Pattern: entity/project/run-id-latest:v0
        match = re.search(r'/run-([a-zA-Z0-9]{8})', artifact_path)
        return match.group(1) if match else None
    
    @staticmethod
    def verify_run_exists(run_id: str) -> bool:
        """Verify that a W&B run exists and is accessible."""
        try:
            import wandb
            # This is a simplified check - in practice you'd want to verify
            # the run exists in the specific project
            return len(run_id) == 8 and run_id.isalnum()
        except ImportError:
            warnings.warn("W&B not available for run verification")
            return False