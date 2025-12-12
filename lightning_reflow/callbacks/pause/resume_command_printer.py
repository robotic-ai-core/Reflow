"""
Resume command printer for pause functionality.

This module handles generation and printing of resume commands after pause:
- Script name detection
- Command formatting with local and W&B options
- Fallback handling for errors
"""

import sys
from typing import Optional, List

from lightning.pytorch import Trainer


class ResumeCommandPrinter:
    """
    Generates and prints resume commands for paused training.

    Handles script detection, command formatting, and fallback output.
    Designed to be used as a composition component by PauseCallback.

    Args:
        original_argv: Copy of sys.argv from when training started

    Example:
        printer = ResumeCommandPrinter(sys.argv.copy())
        printer.print_resume_commands(trainer, checkpoint_path, artifact_path)
    """

    def __init__(self, original_argv: List[str]):
        """
        Initialize the resume command printer.

        Args:
            original_argv: Copy of sys.argv from when training started
        """
        self._original_argv = original_argv

    def print_resume_commands(
        self,
        trainer: Trainer,
        checkpoint_path: str,
        artifact_path: Optional[str] = None
    ) -> None:
        """
        Print resume commands based on what's available.

        Args:
            trainer: PyTorch Lightning trainer (unused but kept for interface consistency)
            checkpoint_path: Path to local checkpoint file
            artifact_path: Optional W&B artifact path

        Raises:
            ValueError: If required inputs are invalid
        """
        # Fail early: Validate critical inputs
        if not checkpoint_path:
            raise ValueError("Checkpoint path cannot be empty")
        if not self._original_argv:
            raise ValueError("Original argv not stored - cannot generate resume commands")

        # Get the script command
        script_command = self._detect_script_command()

        print(f"\nTraining paused. Resume options:")
        print(f"Local resume:    {script_command} resume --checkpoint-path {checkpoint_path}")

        if artifact_path:
            print(f"W&B resume:      {script_command} resume --checkpoint-artifact {artifact_path}")

        # Also show the legacy method for backward compatibility
        legacy_command = self._build_legacy_command()
        print(f"Legacy method:   {legacy_command} --ckpt_path {checkpoint_path}")
        if artifact_path:
            print(f"Legacy W&B:      {legacy_command} --resume_from_wandb {artifact_path}")

    def print_resume_commands_with_fallback(
        self,
        trainer: Trainer,
        checkpoint_path: str,
        artifact_path: Optional[str] = None
    ) -> None:
        """
        Print resume commands with fallback for errors.

        This method wraps print_resume_commands with additional error handling
        to ensure some useful output is always printed.

        Args:
            trainer: PyTorch Lightning trainer
            checkpoint_path: Path to local checkpoint file
            artifact_path: Optional W&B artifact path
        """
        try:
            self.print_resume_commands(trainer, checkpoint_path, artifact_path)
        except ValueError as e:
            print(f"Could not generate resume commands: {e}")
            # Provide basic fallback information
            print(f"Checkpoint saved at: {checkpoint_path}")
            if artifact_path:
                print(f"W&B artifact: {artifact_path}")
            print(f"Use standard Lightning resume: --ckpt_path {checkpoint_path}")
        except Exception as e:
            print(f"Unexpected error generating resume commands: {e}")
            print(f"Checkpoint saved at: {checkpoint_path}")
            print(f"Manually resume with: --ckpt_path {checkpoint_path}")

    def _detect_script_command(self) -> str:
        """
        Detect the script name and build the command prefix.

        Returns:
            Script command with python prefix if needed
        """
        script_name = self._original_argv[0] if self._original_argv else "train_lightning.py"

        # Detect if running via module and provide user-friendly command
        if ("__main__.py" in script_name or
            script_name.endswith("/lightning_reflow/cli/__main__.py") or
            "lightning_reflow" in script_name and "__main__" in script_name):
            # Running via python -m lightning_reflow.cli - suggest user-friendly command
            if len(sys.argv) > 0 and sys.argv[0].endswith('.py'):
                return f"python {sys.argv[0]}"
            else:
                # Fallback to a generic command if we can't determine the actual script
                return "python train_lightning.py"
        elif not script_name.startswith("python"):
            return f"python {script_name}"
        else:
            return script_name

    def _build_legacy_command(self) -> str:
        """
        Build legacy command by filtering out existing --ckpt_path arguments.

        Returns:
            Filtered command suitable for legacy resume
        """
        # Filter out any existing --ckpt_path arguments to avoid duplicates/conflicts
        filtered_argv = []
        i = 0
        while i < len(self._original_argv):
            if self._original_argv[i] == '--ckpt_path':
                # Skip both the flag and its value
                i += 2
            elif self._original_argv[i].startswith('--ckpt_path='):
                # Skip combined flag=value format
                i += 1
            else:
                filtered_argv.append(self._original_argv[i])
                i += 1

        if not filtered_argv[0].startswith("python"):
            return f"python {' '.join(filtered_argv)}"
        else:
            return ' '.join(filtered_argv)

    def update_original_argv(self, new_argv: List[str]) -> None:
        """
        Update the stored original argv.

        This is useful when resuming training, where the current argv
        should replace the original one.

        Args:
            new_argv: New argv list to store
        """
        self._original_argv = new_argv
