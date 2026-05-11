"""CLI subprocess helpers for resume.

These were previously methods on LightningReflow in core/. They have no
dependency on instance state — they're pure subprocess plumbing — so they
live in cli/ to keep core/ free of CLI-shaped knowledge.

LightningReflow keeps thin wrapper methods for backward compatibility with
tests that monkey-patch the methods.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Union

import yaml

logger = logging.getLogger(__name__)


def extract_original_command(checkpoint_path: Union[str, Path]) -> Optional[List[str]]:
    """Read the originally-invoked CLI command from pause_callback_metadata.

    Critical when model/datamodule classes were passed as positional args —
    we need to invoke the same script to reconstruct the class arguments.
    """
    try:
        import torch
        checkpoint = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
        metadata = checkpoint.get('pause_callback_metadata', {})
        original_cmd = metadata.get('original_command')
        if original_cmd and isinstance(original_cmd, list) and len(original_cmd) > 0:
            logger.info("Extracted original command: %s", ' '.join(original_cmd))
            return original_cmd
        logger.debug("No original command found in checkpoint metadata")
        return None
    except Exception as e:
        logger.warning("Failed to extract original command from checkpoint: %s", e)
        return None


def write_temp_config(embedded_config_yaml: Optional[str]) -> Optional[str]:
    """Materialize embedded config YAML to a temp file. Returns path or None."""
    if not embedded_config_yaml:
        return None
    try:
        fd, path = tempfile.mkstemp(suffix='.yaml', prefix='resume_config_')
        with os.fdopen(fd, 'w') as f:
            f.write(embedded_config_yaml)
        return path
    except Exception as e:
        logger.error("Failed to create temporary config file: %s", e)
        return None


def cleanup_temp_config(temp_config_path: Optional[str]) -> None:
    """Best-effort removal of a temp config file."""
    if temp_config_path and os.path.exists(temp_config_path):
        try:
            os.unlink(temp_config_path)
            logger.debug("Cleaned up temporary config: %s", temp_config_path)
        except OSError as e:
            logger.warning("Could not remove temporary config %s: %s", temp_config_path, e)


def add_wandb_resume_config(
    cmd: List[str],
    wandb_run_id: str,
    embedded_config_yaml: Optional[str],
) -> Optional[str]:
    """Append a `--config <temp>` for the W&B resume logger to *cmd*.

    Returns the temp file path so the caller can clean it up.
    """
    try:
        existing_config = (
            yaml.safe_load(embedded_config_yaml) if embedded_config_yaml else {}
        ) or {}
        trainer_config = existing_config.get('trainer', {})
        existing_logger = trainer_config.get('logger', None)

        if (
            isinstance(existing_logger, dict)
            and existing_logger.get('class_path', '').endswith('WandbLogger')
        ):
            logger.info("Updating existing W&B logger configuration for resume")
            existing_logger.setdefault('init_args', {})
            existing_logger['init_args']['id'] = wandb_run_id
            existing_logger['init_args']['resume'] = 'allow'
            wandb_config = {'trainer': {'logger': existing_logger}}
        else:
            logger.info("Creating new W&B logger configuration for resume")
            wandb_config = {'trainer': {'logger': {
                'class_path': 'lightning.pytorch.loggers.WandbLogger',
                'init_args': {
                    'id': wandb_run_id,
                    'resume': 'allow',
                    'log_model': False,
                },
            }}}

        fd, path = tempfile.mkstemp(suffix='.yaml', prefix='wandb_logger_config_')
        with os.fdopen(fd, 'w') as f:
            f.write(yaml.dump(wandb_config))

        cmd.extend(['--config', path])
        logger.info("Configuring W&B logger to resume run: %s", wandb_run_id)
        return path

    except Exception as e:
        logger.error("Failed to create W&B logger config: %s", e)
        path_local = locals().get('path')
        if path_local and os.path.exists(path_local):
            try:
                os.unlink(path_local)
            except OSError:
                pass
        return None


def execute_fit_subprocess(
    checkpoint_path: Optional[Union[str, Path]],
    embedded_config_yaml: Optional[str],
    config_overrides: Optional[List[Union[str, Path]]] = None,
    wandb_run_id: Optional[str] = None,
    extra_cli_args: Optional[List[str]] = None,
) -> None:
    """Spawn `python <script> fit` as a subprocess to drive the resumed run.

    When *checkpoint_path* is None, this still runs `fit` (used for the
    fallback path when a resume source is unavailable).

    Exits the current process with the subprocess's return code.
    """
    original_cmd = (
        extract_original_command(checkpoint_path) if checkpoint_path is not None else None
    )

    if original_cmd and original_cmd[0].endswith('.py'):
        cmd = [sys.executable, original_cmd[0], 'fit']
        logger.info("Using original training script: %s", original_cmd[0])
    elif sys.argv[0].endswith('.py') and Path(sys.argv[0]).exists():
        cmd = [sys.executable, sys.argv[0], 'fit']
        logger.info("Resume fallback: using invoking script %s", sys.argv[0])
    else:
        cmd = [sys.executable, '-m', 'lightning_reflow.cli', 'fit']
        if checkpoint_path is not None:
            logger.warning(
                "Original command not found, using generic CLI (may fail if model_class was provided)"
            )

    temp_config_path = write_temp_config(embedded_config_yaml)
    temp_wandb_config_path: Optional[str] = None

    try:
        if temp_config_path:
            cmd.extend(['--config', temp_config_path])
            logger.info("Using Lightning's original merged config from checkpoint as base")
        else:
            logger.info("No embedded config found in checkpoint, resuming without it.")

        if wandb_run_id:
            temp_wandb_config_path = add_wandb_resume_config(
                cmd, wandb_run_id, embedded_config_yaml,
            )
        else:
            logger.info("No W&B run ID specified - will create new W&B run if logger is configured")

        if config_overrides:
            for config_file in config_overrides:
                cmd.extend(['--config', str(config_file)])
            logger.info("Applying override configs with highest precedence: %s", config_overrides)

        if checkpoint_path is not None:
            cmd.extend(['--ckpt_path', str(checkpoint_path)])

        if extra_cli_args:
            cmd.extend(extra_cli_args)
            logger.info("Passing through additional arguments: %s", extra_cli_args)

        logger.info("Executing: %s", ' '.join(cmd))

        # Forward the current process's sys.path so the subprocess sees any
        # path additions the invoking script made (e.g. project root).
        env = os.environ.copy()
        extra_paths = os.pathsep.join(p for p in sys.path if p)
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{extra_paths}{os.pathsep}{existing_pythonpath}"
            if existing_pythonpath else extra_paths
        )

        result = subprocess.run(cmd, check=True, env=env)
        sys.exit(result.returncode)

    except subprocess.CalledProcessError as e:
        logger.error("Subprocess failed with return code %d", e.returncode)
        sys.exit(e.returncode)
    finally:
        cleanup_temp_config(temp_config_path)
        if temp_wandb_config_path:
            cleanup_temp_config(temp_wandb_config_path)
