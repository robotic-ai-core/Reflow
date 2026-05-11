"""
CLI integration tests: error handling and stale config overwrite.

Focuses on behavior that has real failure modes (invalid subcommands, missing
configs, help-text exit, stale config.yaml overwrite). Trivial smoke tests
that exercise only mocks have been removed.
"""

import yaml
from unittest.mock import patch

import pytest

from lightning_reflow.cli.lightning_cli import LightningReflowCLI


class TestCLIErrorHandling:

    def test_invalid_subcommand_exits(self):
        with pytest.raises((ValueError, SystemExit, AttributeError)):
            with patch('sys.argv', ['train.py', 'invalid_command']):
                LightningReflowCLI()

    def test_missing_config_file_exits(self):
        with pytest.raises((FileNotFoundError, ValueError, SystemExit)):
            with patch('sys.argv', ['train.py', 'fit', '--config', 'nonexistent.yaml']):
                LightningReflowCLI()

    def test_help_text_exits_cleanly(self):
        with patch('sys.argv', ['train.py', '--help']):
            with pytest.raises(SystemExit) as exc_info:
                LightningReflowCLI()
            assert exc_info.value.code == 0


class TestCLIStaleConfigHandling:

    def test_fresh_training_enables_config_overwrite_for_stale_yaml(self, temp_dir):
        config_content = {
            'model': {
                'class_path': 'lightning_reflow.models.SimpleReflowModel',
                'init_args': {'input_dim': 10, 'hidden_dim': 16, 'output_dim': 2},
            },
            'data': {
                'class_path': 'lightning_reflow.data.SimpleDataModule',
                'init_args': {
                    'batch_size': 4, 'train_samples': 8, 'val_samples': 4,
                    'input_dim': 10, 'output_dim': 2,
                },
            },
            'trainer': {
                'max_epochs': 1, 'max_steps': 1,
                'enable_checkpointing': False,
                'default_root_dir': str(temp_dir),
            },
        }
        config_path = temp_dir / "config.yaml"
        config_path.write_text(yaml.dump(config_content))

        logs_dir = temp_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        (logs_dir / "config.yaml").write_text(yaml.dump({'old': 'config'}))

        with patch('sys.argv', ['train.py', 'fit', '--config', str(config_path)]):
            with patch('lightning_reflow.cli.lightning_cli.LightningReflowCLI.__init__', return_value=None):
                cli = LightningReflowCLI.__new__(LightningReflowCLI)
                assert cli._is_fit_command()

                kwargs = {}
                cli._enable_config_overwrite(kwargs)
                assert kwargs['save_config_kwargs']['overwrite'] is True

    def test_enable_config_overwrite_preserves_other_kwargs(self):
        with patch('lightning_reflow.cli.lightning_cli.LightningReflowCLI.__init__', return_value=None):
            cli = LightningReflowCLI.__new__(LightningReflowCLI)

            kwargs = {}
            cli._enable_config_overwrite(kwargs)
            assert kwargs == {'save_config_kwargs': {'overwrite': True}}

            kwargs = {'save_config_kwargs': {'skip_none': False}}
            cli._enable_config_overwrite(kwargs)
            assert kwargs['save_config_kwargs']['overwrite'] is True
            assert kwargs['save_config_kwargs']['skip_none'] is False
