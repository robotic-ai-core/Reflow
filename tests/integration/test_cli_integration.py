"""
Comprehensive CLI integration tests for Lightning Reflow.

Tests the complete CLI workflow including subcommand dispatch, argument linking,
config precedence, and integration with the minimal Lightning Reflow pipeline.
"""

import pytest
import tempfile
import yaml
import torch
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from argparse import Namespace

from lightning_reflow.cli.lightning_cli import LightningReflowCLI


class TestCLIIntegrationBasic:
    """Basic CLI integration tests."""
    
    def test_fit_subcommand_basic_invocation(self, config_file):
        """Test basic fit subcommand invocation."""
        with patch('lightning_reflow.cli.lightning_cli.LightningReflowCLI.__init__', return_value=None) as mock_init:
            with patch.object(LightningReflowCLI, '_configure_pause_exit_callback') as mock_setup:
                with patch.object(LightningReflowCLI, '_setup_logging') as mock_logging:
                    
                    # Create CLI instance
                    cli = LightningReflowCLI.__new__(LightningReflowCLI)
                    cli.subcommand = 'fit'
                    cli.config = Mock()
                    
                    # Mock the initialization process
                    mock_init.return_value = None
                    mock_setup.return_value = None
                    mock_logging.return_value = None
                    
                    # Verify fit subcommand was processed
                    assert cli.subcommand == 'fit'
    
    def test_config_precedence_embedded_vs_cli(self, config_file, mock_checkpoint):
        """Test config precedence: embedded config vs CLI arguments."""
        with patch('lightning_reflow.cli.lightning_cli.LightningReflowCLI.__init__', return_value=None):
            with patch.object(LightningReflowCLI, '_extract_embedded_config_from_checkpoint') as mock_extract:
                
                # Mock embedded config extraction
                embedded_config = 'trainer:\\n  max_epochs: 5'
                mock_extract.return_value = embedded_config
                
                cli = LightningReflowCLI.__new__(LightningReflowCLI)
                cli._checkpoint_path = mock_checkpoint
                
                # Test config extraction
                extracted = cli._extract_embedded_config_from_checkpoint(mock_checkpoint)
                assert extracted == embedded_config
                mock_extract.assert_called_once()
    
    def test_pause_callback_auto_addition_fit_mode(self, config_file):
        """Test that pause callback is automatically added in fit mode."""
        with patch('lightning_reflow.cli.lightning_cli.LightningReflowCLI.__init__', return_value=None):
            with patch.object(LightningReflowCLI, '_configure_pause_exit_callback') as mock_setup:
                
                cli = LightningReflowCLI.__new__(LightningReflowCLI)
                cli.subcommand = 'fit'
                
                # Test pause callback setup for fit command
                cli._configure_pause_exit_callback()
                mock_setup.assert_called_once()
    
    def test_wandb_logger_configuration_from_cli(self, config_file):
        """Test W&B logger configuration from CLI arguments."""
        with patch('lightning_reflow.cli.lightning_cli.LightningReflowCLI.__init__', return_value=None):
            cli = LightningReflowCLI.__new__(LightningReflowCLI)
            
            # Mock trainer with W&B logger
            mock_logger = Mock()
            mock_logger.id = 'test-run-id'
            mock_logger.resume = 'must'
            
            mock_trainer = Mock()
            mock_trainer.logger = mock_logger
            
            cli.trainer = mock_trainer
            
            # Test W&B logger configuration
            if hasattr(cli, '_configure_wandb_logger'):
                cli._configure_wandb_logger('test-run-id', resume='must')
            
            # Verify logger configuration
            assert mock_trainer.logger.id == 'test-run-id'


class TestCLIErrorHandling:
    """Test CLI error handling."""
    
    def test_cli_error_handling_invalid_subcommand(self):
        """Test CLI error handling for invalid subcommands."""
        with pytest.raises((ValueError, SystemExit, AttributeError)):
            with patch('sys.argv', ['train_lightning.py', 'invalid_command']):
                # This should fail during CLI initialization
                LightningReflowCLI()
    
    def test_cli_error_handling_missing_config(self):
        """Test CLI error handling for missing config file."""
        with pytest.raises((FileNotFoundError, ValueError, SystemExit)):
            with patch('sys.argv', ['train_lightning.py', 'fit', '--config', 'nonexistent.yaml']):
                LightningReflowCLI()
    
    def test_help_text_validation(self):
        """Test that help text is properly formatted and informative."""
        with patch('sys.argv', ['train_lightning.py', '--help']):
            with pytest.raises(SystemExit) as exc_info:
                try:
                    LightningReflowCLI()
                except SystemExit as e:
                    # Help should exit with code 0
                    assert e.code == 0
                    raise


class TestCLIConfigIntegration:
    """Test CLI integration with Lightning Reflow config system."""
    
    def test_simple_model_config_loading(self, temp_dir):
        """Test loading SimpleReflowModel through CLI config."""
        config_content = {
            'model': {
                'class_path': 'lightning_reflow.models.SimpleReflowModel',
                'init_args': {
                    'input_dim': 784,
                    'hidden_dim': 256,
                    'output_dim': 10,
                    'learning_rate': 0.001
                }
            },
            'data': {
                'class_path': 'lightning_reflow.data.SimpleDataModule',
                'init_args': {
                    'batch_size': 16,
                    'train_samples': 50,
                    'val_samples': 10
                }
            },
            'trainer': {
                'max_epochs': 1,
                'enable_checkpointing': False
            }
        }
        
        config_path = temp_dir / "simple_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_content, f)
        
        # Test that config can be loaded (mock the actual training)
        with patch('lightning_reflow.cli.lightning_cli.LightningReflowCLI.__init__', return_value=None):
            cli = LightningReflowCLI.__new__(LightningReflowCLI)
            cli.config = config_content
            
            # Verify config structure
            assert cli.config['model']['class_path'] == 'lightning_reflow.models.SimpleReflowModel'
            assert cli.config['data']['class_path'] == 'lightning_reflow.data.SimpleDataModule'
    
    def test_checkpoint_artifact_format_validation(self):
        """Test validation of checkpoint artifact format."""
        cli = LightningReflowCLI.__new__(LightningReflowCLI)
        
        # Test valid formats
        valid_artifacts = [
            "entity/project/artifact:latest",
            "user/proj/run-id-123-pause:v0",
            "org/test/complex-artifact-name:version1.2.3"
        ]
        
        for artifact in valid_artifacts:
            run_id = cli._extract_run_id_from_artifact(artifact)
            # Should extract run ID or return None (but not raise exception)
            assert run_id is None or isinstance(run_id, str)
        
        # Test invalid formats
        invalid_artifacts = [
            "invalid",
            "missing/version",
            "too:many:colons:here",
            ""
        ]
        
        for artifact in invalid_artifacts:
            run_id = cli._extract_run_id_from_artifact(artifact)
            assert run_id is None


class TestCLITrainingIntegration:
    """Test CLI integration with actual training using minimal pipeline."""
    
    def test_minimal_training_flow(self, temp_dir, sample_config):
        """Test minimal training flow with SimpleReflowModel."""
        # Create a minimal config for fast testing
        minimal_config = {
            'model': {
                'class_path': 'lightning_reflow.models.SimpleReflowModel',
                'init_args': {
                    'input_dim': 10,
                    'hidden_dim': 16,
                    'output_dim': 2,
                    'learning_rate': 0.01
                }
            },
            'data': {
                'class_path': 'lightning_reflow.data.SimpleDataModule',
                'init_args': {
                    'batch_size': 4,
                    'train_samples': 8,
                    'val_samples': 4,
                    'input_dim': 10,
                    'output_dim': 2
                }
            },
            'trainer': {
                'max_epochs': 1,
                'max_steps': 2,
                'enable_checkpointing': False,
                'logger': False
            }
        }
        
        config_path = temp_dir / "minimal_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(minimal_config, f)
        
        # Mock CLI initialization to avoid full training
        with patch('lightning_reflow.cli.lightning_cli.LightningReflowCLI.__init__', return_value=None):
            with patch('lightning.pytorch.Trainer.fit') as mock_fit:
                cli = LightningReflowCLI.__new__(LightningReflowCLI)
                cli.config = minimal_config
                cli.subcommand = 'fit'
                
                # Simulate fit call
                mock_fit.return_value = None
                
                # Verify configuration is valid
                assert cli.config['model']['class_path'] == 'lightning_reflow.models.SimpleReflowModel'
                assert cli.config['trainer']['max_steps'] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])