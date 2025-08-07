"""
Unit tests for config override priority in LightningReflow.

Tests the correct priority order during resume:
1. CLI/programmatic overrides (highest priority)
2. User config files (medium priority) 
3. Checkpoint embedded config (lowest priority)
"""

import pytest
import tempfile
import yaml
import torch
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from lightning_reflow.core import LightningReflow
from lightning_reflow.core.config_loader import ConfigLoader


class TestConfigOverridePriority:
    """Test config override priority during resume operations."""
    
    @pytest.fixture
    def temp_checkpoint(self, tmp_path):
        """Create a temporary checkpoint with embedded config."""
        checkpoint_path = tmp_path / "test.ckpt"
        
        # Create checkpoint with embedded config
        checkpoint = {
            'epoch': 1,
            'global_step': 100,
            'state_dict': {},
            'lightning_config': {
                'model': {
                    'class_path': 'lightning_reflow.models.SimpleReflowModel',
                    'init_args': {
                        'learning_rate': 0.001,  # Base value in checkpoint
                        'hidden_dim': 64,        # Base value in checkpoint
                        'dropout': 0.1           # Base value in checkpoint
                    }
                },
                'trainer': {
                    'max_epochs': 10,           # Base value in checkpoint
                    'gradient_clip_val': 1.0    # Base value in checkpoint
                }
            }
        }
        
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path
    
    @pytest.fixture
    def override_config_file(self, tmp_path):
        """Create an override config file."""
        config_path = tmp_path / "override.yaml"
        
        override_config = {
            'model': {
                'init_args': {
                    'learning_rate': 0.0001,  # Override to 0.0001
                    'hidden_dim': 128         # Override to 128
                    # dropout not specified, should keep checkpoint value
                }
            },
            'trainer': {
                'max_epochs': 20  # Override to 20
                # gradient_clip_val not specified, should keep checkpoint value
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(override_config, f)
        
        return config_path
    
    def test_resume_config_priority_with_file_override(self, temp_checkpoint, override_config_file):
        """Test that config file overrides are applied on top of checkpoint config."""
        
        # Create LightningReflow with override config
        reflow = LightningReflow(
            config_files=[str(override_config_file)],
            auto_configure_logging=False
        )
        
        # Mock model creation and trainer.fit to avoid actual instantiation
        with patch.object(reflow, '_create_model') as mock_create_model:
            with patch.object(reflow, '_create_datamodule') as mock_create_data:
                with patch.object(reflow, '_create_trainer') as mock_create_trainer:
                    with patch.object(reflow, '_log_training_summary'):
                        mock_trainer = Mock()
                        mock_trainer.fit.return_value = "Training complete"
                        mock_trainer.callbacks = []
                        mock_trainer.max_epochs = 10
                        mock_create_trainer.return_value = mock_trainer
                        mock_create_model.return_value = Mock()
                        mock_create_data.return_value = Mock()
                        
                        # Resume from checkpoint
                        result = reflow.resume(str(temp_checkpoint))
            
            # Verify the merged config has correct priority
            config = reflow.config
            
            # These should be overridden by the config file
            assert config['model']['init_args']['learning_rate'] == 0.0001
            assert config['model']['init_args']['hidden_dim'] == 128
            assert config['trainer']['max_epochs'] == 20
            
            # These should keep checkpoint values (not in override file)
            assert config['model']['init_args']['dropout'] == 0.1
            assert config['trainer']['gradient_clip_val'] == 1.0
    
    def test_resume_config_priority_with_programmatic_override(self, temp_checkpoint):
        """Test that programmatic overrides have highest priority."""
        
        # Create LightningReflow with programmatic overrides
        reflow = LightningReflow(
            config_overrides={
                'model.init_args.learning_rate': 0.00001,  # Highest priority
                'trainer.gradient_clip_val': 0.5            # Highest priority
            },
            auto_configure_logging=False
        )
        
        # Mock model creation and trainer.fit to avoid actual instantiation
        with patch.object(reflow, '_create_model') as mock_create_model:
            with patch.object(reflow, '_create_datamodule') as mock_create_data:
                with patch.object(reflow, '_create_trainer') as mock_create_trainer:
                    with patch.object(reflow, '_log_training_summary'):
                        mock_trainer = Mock()
                        mock_trainer.fit.return_value = "Training complete"
                        mock_trainer.callbacks = []
                        mock_trainer.max_epochs = 10
                        mock_create_trainer.return_value = mock_trainer
                        mock_create_model.return_value = Mock()
                        mock_create_data.return_value = Mock()
                        
                        # Resume from checkpoint
                        result = reflow.resume(str(temp_checkpoint))
            
            config = reflow.config
            
            # Programmatic overrides should take priority
            assert config['model']['init_args']['learning_rate'] == 0.00001
            assert config['trainer']['gradient_clip_val'] == 0.5
            
            # Other values from checkpoint should be preserved
            assert config['model']['init_args']['hidden_dim'] == 64
            assert config['model']['init_args']['dropout'] == 0.1
            assert config['trainer']['max_epochs'] == 10
    
    def test_resume_config_priority_full_chain(self, temp_checkpoint, override_config_file, tmp_path):
        """Test full priority chain: programmatic > file > checkpoint."""
        
        # Create another config file
        second_override = tmp_path / "second_override.yaml"
        with open(second_override, 'w') as f:
            yaml.dump({
                'model': {
                    'init_args': {
                        'dropout': 0.2  # Override dropout
                    }
                }
            }, f)
        
        # Create LightningReflow with both file and programmatic overrides
        reflow = LightningReflow(
            config_files=[str(override_config_file), str(second_override)],
            config_overrides={
                'model.init_args.learning_rate': 0.00001,  # Highest priority
            },
            auto_configure_logging=False
        )
        
        # Mock model creation and trainer.fit to avoid actual instantiation
        with patch.object(reflow, '_create_model') as mock_create_model:
            with patch.object(reflow, '_create_datamodule') as mock_create_data:
                with patch.object(reflow, '_create_trainer') as mock_create_trainer:
                    with patch.object(reflow, '_log_training_summary'):
                        mock_trainer = Mock()
                        mock_trainer.fit.return_value = "Training complete"
                        mock_trainer.callbacks = []
                        mock_trainer.max_epochs = 10
                        mock_create_trainer.return_value = mock_trainer
                        mock_create_model.return_value = Mock()
                        mock_create_data.return_value = Mock()
                        
                        result = reflow.resume(str(temp_checkpoint))
            
            config = reflow.config
            
            # Programmatic override (highest priority)
            assert config['model']['init_args']['learning_rate'] == 0.00001
            
            # From first override file
            assert config['model']['init_args']['hidden_dim'] == 128
            assert config['trainer']['max_epochs'] == 20
            
            # From second override file (later file has higher priority)
            assert config['model']['init_args']['dropout'] == 0.2
            
            # From checkpoint (lowest priority)
            assert config['trainer']['gradient_clip_val'] == 1.0
    
    def test_multiple_config_files_priority(self, tmp_path):
        """Test that later config files override earlier ones."""
        
        # Create three config files
        config1 = tmp_path / "config1.yaml"
        config2 = tmp_path / "config2.yaml"
        config3 = tmp_path / "config3.yaml"
        
        with open(config1, 'w') as f:
            yaml.dump({
                'model': {'init_args': {'lr': 0.1, 'dim': 32, 'layers': 2}}
            }, f)
        
        with open(config2, 'w') as f:
            yaml.dump({
                'model': {'init_args': {'lr': 0.01, 'dim': 64}}  # Override lr and dim
            }, f)
        
        with open(config3, 'w') as f:
            yaml.dump({
                'model': {'init_args': {'lr': 0.001}}  # Override only lr
            }, f)
        
        # Create LightningReflow with multiple config files
        reflow = LightningReflow(
            config_files=[str(config1), str(config2), str(config3)],
            auto_configure_logging=False
        )
        
        config = reflow.config
        
        # config3 overrides lr (latest file)
        assert config['model']['init_args']['lr'] == 0.001
        
        # config2 overrides dim (config3 doesn't specify it)
        assert config['model']['init_args']['dim'] == 64
        
        # config1's layers remains (not overridden by others)
        assert config['model']['init_args']['layers'] == 2
    
    def test_config_loader_apply_overrides(self):
        """Test the ConfigLoader._apply_overrides method directly."""
        
        loader = ConfigLoader()
        
        # Base config (like from checkpoint)
        base_config = {
            'model': {
                'learning_rate': 0.001,
                'hidden_dim': 64,
                'dropout': 0.1
            },
            'trainer': {
                'max_epochs': 10,
                'gradient_clip_val': 1.0
            }
        }
        
        # User overrides (like from config file)
        user_config = {
            'model': {
                'learning_rate': 0.0001,
                'hidden_dim': 128
            },
            'trainer': {
                'max_epochs': 20
            }
        }
        
        # Apply overrides
        merged = loader._apply_overrides(base_config, user_config)
        
        # Check merged values
        assert merged['model']['learning_rate'] == 0.0001  # Overridden
        assert merged['model']['hidden_dim'] == 128        # Overridden
        assert merged['model']['dropout'] == 0.1           # From base
        assert merged['trainer']['max_epochs'] == 20       # Overridden
        assert merged['trainer']['gradient_clip_val'] == 1.0  # From base


class TestCLIResumeCommand:
    """Test the CLI resume command with multiple config files."""
    
    def test_cli_resume_parser_multiple_configs(self):
        """Test that the resume parser supports multiple --config arguments."""
        from lightning_reflow.cli.lightning_cli import LightningReflowCLI
        
        cli = LightningReflowCLI.__new__(LightningReflowCLI)
        parser = cli._create_resume_parser()
        
        # Test parsing multiple --config arguments
        args = parser.parse_args([
            '--checkpoint-path', '/path/to/checkpoint.ckpt',
            '--config', 'config1.yaml',
            '--config', 'config2.yaml',
            '--config', 'config3.yaml'
        ])
        
        # Should collect all config files in a list
        assert args.config == ['config1.yaml', 'config2.yaml', 'config3.yaml']
        assert args.checkpoint_path == '/path/to/checkpoint.ckpt'
    
    @patch('subprocess.run')
    def test_resume_cli_method_with_multiple_configs(self, mock_subprocess, tmp_path):
        """Test the resume_cli method properly handles multiple config overrides."""
        
        # Create test configs
        config1 = tmp_path / "config1.yaml"
        config2 = tmp_path / "config2.yaml"
        
        with open(config1, 'w') as f:
            yaml.dump({'model': {'lr': 0.01}}, f)
        
        with open(config2, 'w') as f:
            yaml.dump({'model': {'lr': 0.001}}, f)
        
        # Create checkpoint
        checkpoint_path = tmp_path / "checkpoint.ckpt"
        torch.save({
            'state_dict': {},
            'lightning_config': {'model': {'lr': 0.1}}
        }, checkpoint_path)
        
        # Create LightningReflow instance
        reflow = LightningReflow(auto_configure_logging=False)
        
        # Mock the subprocess execution
        mock_subprocess.return_value.returncode = 0
        
        # Call resume_cli with multiple configs
        with patch('sys.exit'):
            reflow.resume_cli(
                resume_source=str(checkpoint_path),
                config_overrides=[str(config1), str(config2)]
            )
        
        # Verify subprocess was called with correct arguments
        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args[0][0]
        
        # Should include both config files in order
        assert '--config' in call_args
        
        # Find all --config arguments
        config_indices = [i for i, arg in enumerate(call_args) if arg == '--config']
        assert len(config_indices) >= 2  # At least embedded config + overrides
        
        # The checkpoint path should be last
        assert '--ckpt_path' in call_args
        ckpt_idx = call_args.index('--ckpt_path')
        assert call_args[ckpt_idx + 1] == str(checkpoint_path)


class TestConfigMergingEdgeCases:
    """Test edge cases in config merging during resume."""
    
    def test_resume_without_embedded_config(self, tmp_path):
        """Test resume when checkpoint has no embedded config."""
        
        # Create checkpoint without lightning_config
        checkpoint_path = tmp_path / "no_config.ckpt"
        torch.save({
            'epoch': 1,
            'state_dict': {},
            # No lightning_config key
        }, checkpoint_path)
        
        # Create override config
        override_config = tmp_path / "override.yaml"
        with open(override_config, 'w') as f:
            yaml.dump({
                'model': {
                    'class_path': 'lightning_reflow.models.SimpleReflowModel',
                    'init_args': {'learning_rate': 0.001}
                }
            }, f)
        
        reflow = LightningReflow(
            config_files=[str(override_config)],
            auto_configure_logging=False
        )
        
        with patch('lightning.pytorch.Trainer.fit') as mock_fit:
            mock_fit.return_value = "Training complete"
            
            # Should handle missing embedded config gracefully
            result = reflow.resume(str(checkpoint_path))
            
            # Config should come from override file
            assert reflow.config['model']['init_args']['learning_rate'] == 0.001
    
    def test_resume_with_empty_override(self, tmp_path):
        """Test resume with empty override config."""
        
        # Create checkpoint with config
        checkpoint_path = tmp_path / "checkpoint.ckpt"
        checkpoint_config = {
            'model': {
                'class_path': 'lightning_reflow.models.SimpleReflowModel',
                'init_args': {'lr': 0.001}
            },
            'trainer': {'max_epochs': 10}
        }
        torch.save({
            'state_dict': {},
            'lightning_config': checkpoint_config
        }, checkpoint_path)
        
        # Create empty override config
        empty_config = tmp_path / "empty.yaml"
        with open(empty_config, 'w') as f:
            yaml.dump({}, f)
        
        reflow = LightningReflow(
            config_files=[str(empty_config)],
            auto_configure_logging=False
        )
        
        # Mock model creation and trainer.fit to avoid actual instantiation
        with patch.object(reflow, '_create_model') as mock_create_model:
            with patch.object(reflow, '_create_datamodule') as mock_create_data:
                with patch.object(reflow, '_create_trainer') as mock_create_trainer:
                    with patch.object(reflow, '_log_training_summary'):
                        mock_trainer = Mock()
                        mock_trainer.fit.return_value = "Training complete"
                        mock_trainer.callbacks = []
                        mock_trainer.max_epochs = 10
                        mock_create_trainer.return_value = mock_trainer
                        mock_create_model.return_value = Mock()
                        mock_create_data.return_value = Mock()
                        
                        result = reflow.resume(str(checkpoint_path))
            
            # Should use checkpoint config since override is empty
            assert reflow.config['model']['init_args']['lr'] == 0.001
            assert reflow.config['trainer']['max_epochs'] == 10
    
    def test_nested_config_override(self, tmp_path):
        """Test deeply nested config overrides."""
        
        # Create checkpoint with nested config
        checkpoint_path = tmp_path / "checkpoint.ckpt"
        torch.save({
            'state_dict': {},
            'lightning_config': {
                'model': {
                    'class_path': 'lightning_reflow.models.SimpleReflowModel',
                    'init_args': {
                        'backbone': {
                            'type': 'resnet',
                            'params': {
                                'layers': 50,
                                'pretrained': True,
                                'freeze': False
                            }
                        }
                    }
                }
            }
        }, checkpoint_path)
        
        # Override only specific nested values
        reflow = LightningReflow(
            config_overrides={
                'model.init_args.backbone.params.layers': 101,
                'model.init_args.backbone.params.freeze': True
            },
            auto_configure_logging=False
        )
        
        # Mock model creation and trainer.fit to avoid actual instantiation
        with patch.object(reflow, '_create_model') as mock_create_model:
            with patch.object(reflow, '_create_datamodule') as mock_create_data:
                with patch.object(reflow, '_create_trainer') as mock_create_trainer:
                    with patch.object(reflow, '_log_training_summary'):
                        mock_trainer = Mock()
                        mock_trainer.fit.return_value = "Training complete"
                        mock_trainer.callbacks = []
                        mock_trainer.max_epochs = 10
                        mock_create_trainer.return_value = mock_trainer
                        mock_create_model.return_value = Mock()
                        mock_create_data.return_value = Mock()
                        
                        result = reflow.resume(str(checkpoint_path))
            
            config = reflow.config
            
            # Check nested overrides applied correctly
            backbone_params = config['model']['init_args']['backbone']['params']
            assert backbone_params['layers'] == 101       # Overridden
            assert backbone_params['freeze'] == True       # Overridden
            assert backbone_params['pretrained'] == True   # From checkpoint
            assert config['model']['init_args']['backbone']['type'] == 'resnet'  # From checkpoint


if __name__ == "__main__":
    pytest.main([__file__, "-v"])