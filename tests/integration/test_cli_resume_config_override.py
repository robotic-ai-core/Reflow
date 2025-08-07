"""
Integration tests for CLI resume with config overrides.

Tests the complete CLI resume flow with multiple --config files and
verifies correct priority ordering.
"""

import pytest
import tempfile
import yaml
import torch
import subprocess
import sys
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lightning_reflow.core import LightningReflow
from lightning_reflow.cli.lightning_cli import LightningReflowCLI


class TestCLIResumeIntegration:
    """Integration tests for CLI resume command with config overrides."""
    
    @pytest.fixture
    def setup_test_environment(self, tmp_path):
        """Setup a complete test environment with configs and checkpoint."""
        
        # Create checkpoint with embedded config
        checkpoint_path = tmp_path / "checkpoint.ckpt"
        checkpoint_config = {
            'model': {
                'class_path': 'lightning_reflow.models.SimpleReflowModel',
                'init_args': {
                    'input_dim': 10,
                    'hidden_dim': 32,
                    'output_dim': 3,
                    'learning_rate': 0.01
                }
            },
            'data': {
                'class_path': 'lightning_reflow.data.SimpleDataModule',
                'init_args': {
                    'batch_size': 32,
                    'train_samples': 100,
                    'val_samples': 20,
                    'input_dim': 10,
                    'output_dim': 3
                }
            },
            'trainer': {
                'max_epochs': 100,
                'gradient_clip_val': 1.0,
                'accumulate_grad_batches': 1
            }
        }
        
        torch.save({
            'epoch': 10,
            'global_step': 500,
            'state_dict': {'dummy': torch.tensor([1.0])},
            'lightning_config': checkpoint_config,
            'wandb_run_id': 'test-run-123'
        }, checkpoint_path)
        
        # Create first override config
        config1_path = tmp_path / "override1.yaml"
        with open(config1_path, 'w') as f:
            yaml.dump({
                'model': {
                    'init_args': {
                        'learning_rate': 0.001,  # Override from 0.01
                        'hidden_dim': 64         # Override from 32
                    }
                },
                'trainer': {
                    'max_epochs': 200        # Override from 100
                }
            }, f)
        
        # Create second override config
        config2_path = tmp_path / "override2.yaml"
        with open(config2_path, 'w') as f:
            yaml.dump({
                'model': {
                    'init_args': {
                        'learning_rate': 0.0001,  # Override from 0.001
                        'output_dim': 5           # Override from 3
                    }
                },
                'data': {
                    'init_args': {
                        'batch_size': 64          # Override from 32
                    }
                }
            }, f)
        
        return {
            'checkpoint_path': checkpoint_path,
            'config1_path': config1_path,
            'config2_path': config2_path,
            'tmp_path': tmp_path,
            'checkpoint_config': checkpoint_config
        }
    
    @patch('subprocess.run')
    def test_cli_resume_with_multiple_configs(self, mock_run, setup_test_environment):
        """Test CLI resume command with multiple --config arguments."""
        
        env = setup_test_environment
        
        # Mock successful subprocess execution
        mock_run.return_value.returncode = 0
        
        # Test the CLI with multiple configs
        with patch('sys.argv', [
            'lightning-reflow', 'resume',
            '--checkpoint-path', str(env['checkpoint_path']),
            '--config', str(env['config1_path']),
            '--config', str(env['config2_path'])
        ]):
            with patch('sys.exit'):
                cli = LightningReflowCLI()
        
        # Verify subprocess.run was called
        mock_run.assert_called_once()
        
        # Get the command that was executed
        cmd = mock_run.call_args[0][0]
        
        # Verify command structure
        assert cmd[0] == sys.executable
        assert '-m' in cmd
        assert 'lightning_reflow.cli' in cmd
        assert 'fit' in cmd
        
        # Verify configs are passed in correct order
        config_indices = [i for i, arg in enumerate(cmd) if arg == '--config']
        
        # Should have embedded config + 2 override configs
        assert len(config_indices) >= 3
        
        # Verify checkpoint path is included
        assert '--ckpt_path' in cmd
        ckpt_idx = cmd.index('--ckpt_path')
        assert str(env['checkpoint_path']) in cmd[ckpt_idx + 1]
    
    def test_cli_resume_config_priority_order(self, setup_test_environment):
        """Test that config priority is correctly applied in CLI resume."""
        
        env = setup_test_environment
        
        # Create a LightningReflow instance to test the merge logic
        reflow = LightningReflow(
            config_files=[str(env['config1_path']), str(env['config2_path'])],
            auto_configure_logging=False
        )
        
        with patch('lightning.pytorch.Trainer') as MockTrainer:
            mock_trainer = Mock()
            mock_trainer.fit.return_value = "Training complete"
            MockTrainer.return_value = mock_trainer
            
            with patch('lightning_reflow.models.SimpleReflowModel') as MockModel:
                mock_model = Mock()
                MockModel.return_value = mock_model
                
                with patch('lightning_reflow.data.SimpleDataModule') as MockDataModule:
                    mock_datamodule = Mock()
                    MockDataModule.return_value = mock_datamodule
                    
                    with patch.object(reflow, '_log_training_summary'):
                        # Resume with the configs
                        result = reflow.resume(str(env['checkpoint_path']))
            
            # Verify final config has correct priority
            config = reflow.config
            
            # From config2 (highest priority file)
            assert config['model']['init_args']['learning_rate'] == 0.0001
            assert config['model']['init_args']['output_dim'] == 5
            assert config['data']['init_args']['batch_size'] == 64
            
            # From config1 (not overridden by config2)
            assert config['model']['init_args']['hidden_dim'] == 64
            assert config['trainer']['max_epochs'] == 200
            
            # From checkpoint (not overridden by any config)
            assert config['model']['init_args']['input_dim'] == 10
            assert config['trainer']['gradient_clip_val'] == 1.0
            assert config['trainer']['accumulate_grad_batches'] == 1
    
    @patch('lightning_reflow.core.LightningReflow._execute_fit_subprocess')
    def test_resume_cli_method(self, mock_execute, setup_test_environment):
        """Test the resume_cli method with multiple config overrides."""
        
        env = setup_test_environment
        
        # Create LightningReflow instance
        reflow = LightningReflow(auto_configure_logging=False)
        
        # Call resume_cli
        reflow.resume_cli(
            resume_source=str(env['checkpoint_path']),
            config_overrides=[str(env['config1_path']), str(env['config2_path'])],
            use_wandb_config=False,
            extra_cli_args=['--trainer.devices', '2']
        )
        
        # Verify _execute_fit_subprocess was called with correct args
        mock_execute.assert_called_once()
        
        call_args = mock_execute.call_args[1]
        
        # Check that config overrides were passed
        assert call_args['config_overrides'] == [str(env['config1_path']), str(env['config2_path'])]
        
        # Check extra CLI args were passed
        assert call_args['extra_cli_args'] == ['--trainer.devices', '2']
        
        # Check W&B run ID was extracted - CRITICAL for resuming to same W&B run!
        assert call_args['wandb_run_id'] == 'test-run-123'
    
    def test_cli_resume_with_wandb_artifact(self, setup_test_environment):
        """Test CLI resume with W&B artifact source."""
        
        env = setup_test_environment
        
        with patch('sys.argv', [
            'lightning-reflow', 'resume',
            '--checkpoint-artifact', 'entity/project/run-123:latest',
            '--config', str(env['config1_path']),
            '--use-wandb-config',
            '--entity', 'test-entity',
            '--project', 'test-project'
        ]):
            with patch('lightning_reflow.strategies.wandb_artifact_resume_strategy.WandbArtifactResumeStrategy.prepare_resume') as mock_prepare:
                mock_prepare.return_value = (env['checkpoint_path'], yaml.dump(env['checkpoint_config']))
                
                with patch('subprocess.run') as mock_run:
                    mock_run.return_value.returncode = 0
                    
                    with patch('sys.exit'):
                        cli = LightningReflowCLI()
                
                # Verify W&B strategy was used
                mock_prepare.assert_called_once()
                call_args = mock_prepare.call_args[1]
                assert call_args['resume_source'] == 'entity/project/run-123:latest'
                assert call_args['use_wandb_config'] == True
                assert call_args['entity'] == 'test-entity'
                assert call_args['project'] == 'test-project'


class TestConfigPriorityWithComplexScenarios:
    """Test config priority with complex real-world scenarios."""
    
    def test_distributed_training_config_override(self, tmp_path):
        """Test config overrides for distributed training settings."""
        
        # Create checkpoint with single-GPU config
        checkpoint_path = tmp_path / "single_gpu.ckpt"
        torch.save({
            'state_dict': {},
            'lightning_config': {
                'model': {
                    'class_path': 'lightning_reflow.models.SimpleReflowModel',
                    'init_args': {'input_dim': 10, 'output_dim': 2}
                },
                'trainer': {
                    'devices': 1,
                    'accelerator': 'gpu',
                    'strategy': 'auto',
                    'precision': 32
                }
            }
        }, checkpoint_path)
        
        # Override for multi-GPU training
        override_config = {
            'trainer': {
                'devices': 4,
                'strategy': 'ddp',
                'precision': 16,
                'sync_batchnorm': True
            }
        }
        
        override_path = tmp_path / "multi_gpu.yaml"
        with open(override_path, 'w') as f:
            yaml.dump(override_config, f)
        
        reflow = LightningReflow(
            config_files=[str(override_path)],
            auto_configure_logging=False
        )
        
        with patch('lightning.pytorch.Trainer') as MockTrainer:
            mock_trainer = Mock()
            mock_trainer.fit.return_value = "Training complete"
            MockTrainer.return_value = mock_trainer
            
            with patch('lightning_reflow.models.SimpleReflowModel') as MockModel:
                mock_model = Mock()
                MockModel.return_value = mock_model
                
                with patch.object(reflow, '_log_training_summary'):
                    result = reflow.resume(str(checkpoint_path))
            
            config = reflow.config
            
            # Distributed settings should be overridden
            assert config['trainer']['devices'] == 4
            assert config['trainer']['strategy'] == 'ddp'
            assert config['trainer']['precision'] == 16
            assert config['trainer']['sync_batchnorm'] == True
            
            # Original accelerator should be preserved
            assert config['trainer']['accelerator'] == 'gpu'
    
    def test_callback_config_override(self, tmp_path):
        """Test that callback configurations can be overridden."""
        
        # Create checkpoint with callbacks
        checkpoint_path = tmp_path / "checkpoint.ckpt"
        torch.save({
            'state_dict': {},
            'lightning_config': {
                'model': {
                    'class_path': 'lightning_reflow.models.SimpleReflowModel',
                    'init_args': {'input_dim': 10, 'output_dim': 2}
                },
                'trainer': {
                    'callbacks': [
                        {
                            'class_path': 'lightning_reflow.callbacks.pause.PauseCallback',
                            'init_args': {
                                'checkpoint_dir': '/old/path',
                                'check_interval': 100
                            }
                        },
                        {
                            'class_path': 'lightning.pytorch.callbacks.ModelCheckpoint',
                            'init_args': {
                                'save_top_k': 1,
                                'monitor': 'val_loss'
                            }
                        }
                    ]
                }
            }
        }, checkpoint_path)
        
        # Override callback settings
        override_config = {
            'trainer': {
                'callbacks': [
                    {
                        'class_path': 'lightning_reflow.callbacks.pause.PauseCallback',
                        'init_args': {
                            'checkpoint_dir': '/new/path',
                            'check_interval': 50,
                            'enable_pause': True
                        }
                    },
                    {
                        'class_path': 'lightning.pytorch.callbacks.ModelCheckpoint',
                        'init_args': {
                            'save_top_k': 3,
                            'monitor': 'val_accuracy',
                            'mode': 'max'
                        }
                    },
                    {
                        'class_path': 'lightning.pytorch.callbacks.EarlyStopping',
                        'init_args': {
                            'monitor': 'val_loss',
                            'patience': 10
                        }
                    }
                ]
            }
        }
        
        override_path = tmp_path / "callbacks.yaml"
        with open(override_path, 'w') as f:
            yaml.dump(override_config, f)
        
        reflow = LightningReflow(
            config_files=[str(override_path)],
            auto_configure_logging=False
        )
        
        with patch('lightning.pytorch.Trainer') as MockTrainer:
            mock_trainer = Mock()
            mock_trainer.fit.return_value = "Training complete"
            MockTrainer.return_value = mock_trainer
            
            with patch('lightning_reflow.models.SimpleReflowModel') as MockModel:
                mock_model = Mock()
                MockModel.return_value = mock_model
                
                with patch.object(reflow, '_log_training_summary'):
                    result = reflow.resume(str(checkpoint_path))
            
            config = reflow.config
            
            # Check callbacks are overridden
            callbacks = config['trainer']['callbacks']
            assert len(callbacks) == 3  # Added EarlyStopping
            
            # Check PauseCallback override
            pause_cb = callbacks[0]
            assert pause_cb['init_args']['checkpoint_dir'] == '/new/path'
            assert pause_cb['init_args']['check_interval'] == 50
            assert pause_cb['init_args']['enable_pause'] == True
            
            # Check ModelCheckpoint override
            ckpt_cb = callbacks[1]
            assert ckpt_cb['init_args']['save_top_k'] == 3
            assert ckpt_cb['init_args']['monitor'] == 'val_accuracy'
            assert ckpt_cb['init_args']['mode'] == 'max'
            
            # Check new EarlyStopping callback
            early_stop = callbacks[2]
            assert early_stop['class_path'] == 'lightning.pytorch.callbacks.EarlyStopping'
            assert early_stop['init_args']['patience'] == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])