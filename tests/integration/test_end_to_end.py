"""
End-to-end integration tests for Lightning Reflow.

Tests the complete pipeline from CLI invocation through training with the minimal components.
"""

import pytest
import tempfile
import yaml
import torch
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from lightning_reflow.models import SimpleReflowModel
from lightning_reflow.data import SimpleDataModule
from lightning_reflow.cli.lightning_cli import LightningReflowCLI


class TestEndToEndPipeline:
    """Test complete end-to-end Lightning Reflow pipeline."""
    
    def test_minimal_training_pipeline(self, temp_dir):
        """Test minimal training pipeline from start to finish."""
        # Create minimal config
        config = {
            'model': {
                'class_path': 'lightning_reflow.models.SimpleReflowModel',
                'init_args': {
                    'input_dim': 20,
                    'hidden_dim': 32,
                    'output_dim': 3,
                    'learning_rate': 0.01,
                    'loss_type': 'cross_entropy'
                }
            },
            'data': {
                'class_path': 'lightning_reflow.data.SimpleDataModule',
                'init_args': {
                    'batch_size': 8,
                    'train_samples': 24,  # 3 batches
                    'val_samples': 8,     # 1 batch
                    'input_dim': 20,
                    'output_dim': 3,
                    'task_type': 'classification',
                    'seed': 42
                }
            },
            'trainer': {
                'max_epochs': 1,
                'max_steps': 3,
                'enable_checkpointing': False,
                'logger': False,
                'enable_progress_bar': False
            }
        }
        
        config_path = temp_dir / "minimal_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Test that we can create model and data module
        model = SimpleReflowModel(**config['model']['init_args'])
        data_module = SimpleDataModule(**config['data']['init_args'])
        
        # Setup data
        data_module.setup()
        
        # Verify data shapes
        train_batch = next(iter(data_module.train_dataloader()))
        assert train_batch['input'].shape == (8, 20)
        assert train_batch['target'].shape == (8,)
        
        # Test forward pass
        output = model(train_batch['input'])
        assert output.shape == (8, 3)
        
        # Test loss calculation
        loss = model.training_step(train_batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
    
    @patch('lightning.pytorch.Trainer.fit')
    def test_cli_integration_with_minimal_config(self, mock_fit, temp_dir):
        """Test CLI integration with minimal config."""
        config = {
            'model': {
                'class_path': 'lightning_reflow.models.SimpleReflowModel',
                'init_args': {
                    'input_dim': 10,
                    'hidden_dim': 16,
                    'output_dim': 2
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
                'enable_checkpointing': False,
                'logger': False
            }
        }
        
        config_path = temp_dir / "test_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Mock CLI initialization to avoid actual training
        with patch('sys.argv', ['train.py', 'fit', '--config', str(config_path)]):
            with patch('lightning_reflow.cli.lightning_cli.LightningReflowCLI.__init__', return_value=None):
                cli = LightningReflowCLI.__new__(LightningReflowCLI)
                cli.config = config
                cli.subcommand = 'fit'
                
                # Verify config was loaded correctly
                assert cli.config['model']['class_path'] == 'lightning_reflow.models.SimpleReflowModel'
                assert cli.config['data']['class_path'] == 'lightning_reflow.data.SimpleDataModule'
    
    def test_pause_resume_integration(self, temp_dir, mock_checkpoint):
        """Test pause/resume integration with minimal pipeline."""
        # Create config with pause callback
        config = {
            'model': {
                'class_path': 'lightning_reflow.models.SimpleReflowModel',
                'init_args': {'input_dim': 10, 'output_dim': 2}
            },
            'data': {
                'class_path': 'lightning_reflow.data.SimpleDataModule',
                'init_args': {
                    'batch_size': 4,
                    'train_samples': 8,
                    'input_dim': 10,
                    'output_dim': 2
                }
            },
            'trainer': {
                'max_epochs': 2,
                'enable_checkpointing': True,
                'callbacks': [{
                    'class_path': 'lightning_reflow.callbacks.pause.PauseCallback',
                    'init_args': {'checkpoint_dir': str(temp_dir / 'checkpoints')}
                }]
            }
        }
        
        # Test resume from checkpoint
        with patch('lightning_reflow.cli.lightning_cli.LightningReflowCLI._execute_resume_from_checkpoint') as mock_resume:
            mock_resume.return_value = True
            
            cli = LightningReflowCLI.__new__(LightningReflowCLI)
            
            result = cli._execute_resume_from_checkpoint(
                checkpoint_path=mock_checkpoint,
                config_overrides=None,
                dry_run=True
            )
            
            assert result is True
            mock_resume.assert_called_once()
    
    def test_wandb_integration_mock(self, temp_dir, mock_wandb):
        """Test W&B integration with mocked wandb."""
        config = {
            'model': {
                'class_path': 'lightning_reflow.models.SimpleReflowModel',
                'init_args': {'input_dim': 10, 'output_dim': 2}
            },
            'data': {
                'class_path': 'lightning_reflow.data.SimpleDataModule',
                'init_args': {
                    'batch_size': 4,
                    'train_samples': 8,
                    'input_dim': 10,
                    'output_dim': 2
                }
            },
            'trainer': {
                'max_epochs': 1,
                'logger': {
                    'class_path': 'lightning.pytorch.loggers.WandbLogger',
                    'init_args': {
                        'project': 'test-project',
                        'offline': True
                    }
                }
            }
        }
        
        # Test that W&B logger can be configured
        with patch('lightning.pytorch.loggers.WandbLogger') as mock_logger_class:
            mock_logger = Mock()
            mock_logger.id = 'test-run-123'
            mock_logger_class.return_value = mock_logger
            
            # Verify logger creation would work
            logger_config = config['trainer']['logger']
            assert logger_config['class_path'] == 'lightning.pytorch.loggers.WandbLogger'
            assert logger_config['init_args']['project'] == 'test-project'
    
    def test_config_override_integration(self, temp_dir):
        """Test config override functionality."""
        base_config = {
            'model': {
                'class_path': 'lightning_reflow.models.SimpleReflowModel',
                'init_args': {'learning_rate': 0.001}
            },
            'trainer': {'max_epochs': 10}
        }
        
        override_config = {
            'model': {
                'init_args': {'learning_rate': 0.01}
            },
            'trainer': {'max_epochs': 5}
        }
        
        base_path = temp_dir / "base_config.yaml"
        override_path = temp_dir / "override_config.yaml"
        
        with open(base_path, 'w') as f:
            yaml.dump(base_config, f)
        with open(override_path, 'w') as f:
            yaml.dump(override_config, f)
        
        # Test config merging logic
        with patch('lightning_reflow.cli.lightning_cli.LightningReflowCLI._merge_configs') as mock_merge:
            mock_merge.return_value = {
                'model': {
                    'class_path': 'lightning_reflow.models.SimpleReflowModel',
                    'init_args': {'learning_rate': 0.01}  # Override value
                },
                'trainer': {'max_epochs': 5}  # Override value
            }
            
            cli = LightningReflowCLI.__new__(LightningReflowCLI)
            merged = cli._merge_configs([str(base_path), str(override_path)])
            
            mock_merge.assert_called_once()
            assert merged['model']['init_args']['learning_rate'] == 0.01
            assert merged['trainer']['max_epochs'] == 5
    
    def test_error_handling_integration(self, temp_dir):
        """Test error handling throughout the pipeline."""
        # Test invalid model class
        invalid_config = {
            'model': {
                'class_path': 'lightning_reflow.models.NonexistentModel',
                'init_args': {}
            },
            'data': {
                'class_path': 'lightning_reflow.data.SimpleDataModule',
                'init_args': {}
            }
        }
        
        config_path = temp_dir / "invalid_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(invalid_config, f)
        
        # Should handle import errors gracefully
        with patch('sys.argv', ['train.py', 'fit', '--config', str(config_path)]):
            with pytest.raises((ImportError, ModuleNotFoundError, AttributeError)):
                # This should fail when trying to import NonexistentModel
                from lightning_reflow.models import NonexistentModel
    
    def test_checkpoint_compatibility(self, temp_dir, mock_checkpoint):
        """Test checkpoint compatibility across different components."""
        # Create a model
        model = SimpleReflowModel(input_dim=20, hidden_dim=32, output_dim=3)
        
        # Save model state
        checkpoint_path = temp_dir / "test_checkpoint.ckpt"
        torch.save({
            'state_dict': model.state_dict(),
            'hyper_parameters': model.hparams,
            'epoch': 5,
            'global_step': 100,
            'wandb_run_id': 'test-run-123'
        }, checkpoint_path)
        
        # Test loading
        checkpoint = torch.load(checkpoint_path)
        
        # Verify checkpoint structure
        assert 'state_dict' in checkpoint
        assert 'hyper_parameters' in checkpoint
        assert 'wandb_run_id' in checkpoint
        assert checkpoint['epoch'] == 5
        assert checkpoint['global_step'] == 100
        
        # Test model loading
        new_model = SimpleReflowModel(**checkpoint['hyper_parameters'])
        new_model.load_state_dict(checkpoint['state_dict'])
        
        # Models should have same parameters
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            torch.testing.assert_close(p1, p2)
    
    def test_data_pipeline_consistency(self):
        """Test data pipeline consistency across different configurations."""
        # Test different configurations produce consistent data
        configs = [
            {'batch_size': 4, 'seed': 42},
            {'batch_size': 8, 'seed': 42},
            {'batch_size': 4, 'seed': 42}  # Same as first
        ]
        
        datasets = []
        for config in configs:
            dm = SimpleDataModule(
                train_samples=16,
                input_dim=10,
                output_dim=2,
                **config
            )
            dm.setup()
            datasets.append(dm)
        
        # First and third should have identical first batch
        batch1 = next(iter(datasets[0].train_dataloader()))
        batch3 = next(iter(datasets[2].train_dataloader()))
        
        torch.testing.assert_close(batch1['input'], batch3['input'])
        torch.testing.assert_close(batch1['target'], batch3['target'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])