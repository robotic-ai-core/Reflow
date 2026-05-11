"""Tests for the --wandb-run-id CLI argument and the resolution helper.

The W&B run-id resolution logic lives in LightningReflow._resolve_wandb_run_id,
which lets us cover the precedence rules directly instead of monkey-patching
LightningReflow._add_wandb_resume_config (the previous brittle approach).
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

from lightning_reflow.cli import LightningReflowCLI
from lightning_reflow.core import LightningReflow


def _write_checkpoint(payload: dict) -> str:
    """Save a checkpoint to a temp file and return its path."""
    tmp = tempfile.NamedTemporaryFile(suffix='.ckpt', delete=False)
    torch.save(payload, tmp.name)
    tmp.close()
    return tmp.name


CHECKPOINT_WITH_ID = {
    'epoch': 5,
    'global_step': 1000,
    'state_dict': {},
    'self_contained_metadata': {
        'wandb_run_id': 'checkpoint-run-id',
        'embedded_config_content': 'model:\n  class_path: test.Model\n',
    },
}
CHECKPOINT_WITHOUT_ID = {
    'epoch': 5,
    'global_step': 1000,
    'state_dict': {},
    'self_contained_metadata': {
        'embedded_config_content': 'model:\n  class_path: test.Model\n',
    },
}


class TestWandbRunIdParser:

    def test_explicit_value_parsed(self):
        cli = LightningReflowCLI.__new__(LightningReflowCLI)
        parser = cli._create_resume_parser()
        args = parser.parse_args([
            '--checkpoint-path', '/path/to/checkpoint.ckpt',
            '--wandb-run-id', 'custom-run-id-123',
        ])
        assert args.wandb_run_id == 'custom-run-id-123'

    def test_flag_without_value_means_new(self):
        cli = LightningReflowCLI.__new__(LightningReflowCLI)
        parser = cli._create_resume_parser()
        args = parser.parse_args([
            '--checkpoint-path', '/path/to/checkpoint.ckpt',
            '--wandb-run-id',
        ])
        assert args.wandb_run_id == 'new'

    def test_flag_omitted_is_none(self):
        cli = LightningReflowCLI.__new__(LightningReflowCLI)
        parser = cli._create_resume_parser()
        args = parser.parse_args(['--checkpoint-path', '/path/to/checkpoint.ckpt'])
        assert args.wandb_run_id is None


class TestResolveWandbRunId:
    """Direct unit tests of LightningReflow._resolve_wandb_run_id."""

    def test_explicit_id_overrides_checkpoint(self):
        ckpt_path = _write_checkpoint(CHECKPOINT_WITH_ID)
        try:
            reflow = LightningReflow()
            resolved = reflow._resolve_wandb_run_id('override-run-id', ckpt_path)
            assert resolved == 'override-run-id'
        finally:
            Path(ckpt_path).unlink(missing_ok=True)

    def test_new_sentinel_forces_none(self):
        ckpt_path = _write_checkpoint(CHECKPOINT_WITH_ID)
        try:
            reflow = LightningReflow()
            resolved = reflow._resolve_wandb_run_id('new', ckpt_path)
            assert resolved is None
        finally:
            Path(ckpt_path).unlink(missing_ok=True)

    def test_none_extracts_from_checkpoint(self):
        ckpt_path = _write_checkpoint(CHECKPOINT_WITH_ID)
        try:
            reflow = LightningReflow()
            resolved = reflow._resolve_wandb_run_id(None, ckpt_path)
            assert resolved == 'checkpoint-run-id'
        finally:
            Path(ckpt_path).unlink(missing_ok=True)

    def test_none_returns_none_when_checkpoint_has_no_id(self):
        ckpt_path = _write_checkpoint(CHECKPOINT_WITHOUT_ID)
        try:
            reflow = LightningReflow()
            resolved = reflow._resolve_wandb_run_id(None, ckpt_path)
            assert resolved is None
        finally:
            Path(ckpt_path).unlink(missing_ok=True)


class TestResumeCliDispatch:

    def test_cli_passes_wandb_run_id_through_to_resume_cli(self):
        ckpt_path = _write_checkpoint(CHECKPOINT_WITHOUT_ID)
        try:
            with patch('sys.argv', [
                'script', 'resume',
                '--checkpoint-path', ckpt_path,
                '--wandb-run-id', 'my-custom-run',
            ]):
                with patch.object(LightningReflow, 'resume_cli') as mock_resume_cli:
                    with patch('sys.exit'):
                        cli = LightningReflowCLI.__new__(LightningReflowCLI)
                        cli._execute_resume_as_subprocess()
                    mock_resume_cli.assert_called_once()
                    assert mock_resume_cli.call_args.kwargs['wandb_run_id'] == 'my-custom-run'
        finally:
            Path(ckpt_path).unlink(missing_ok=True)

    def test_no_extra_config_added_when_no_run_id_resolved(self):
        """When no run id is resolved (no flag, no checkpoint id), the
        constructed command does not include a second --config for W&B."""
        ckpt_path = _write_checkpoint(CHECKPOINT_WITHOUT_ID)
        try:
            with patch('subprocess.run') as mock_run, patch('sys.exit'):
                mock_run.return_value = Mock(returncode=0)
                reflow = LightningReflow()
                reflow.resume_cli(resume_source=ckpt_path)
            cmd = mock_run.call_args[0][0]
            assert cmd.count('--config') == 1
        finally:
            Path(ckpt_path).unlink(missing_ok=True)
