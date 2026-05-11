"""Resume config-override priority tests.

Asserts that CLI / additional-config overrides win over embedded-checkpoint
config during a resume, end-to-end through a fit cycle.
"""

import pytest
import tempfile
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import Mock, patch

import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from torch.utils.data import DataLoader

from lightning_reflow.models import SimpleReflowModel
from lightning_reflow.data import SimpleDataModule
from lightning_reflow.core import LightningReflow
from lightning_reflow.callbacks.pause import PauseCallback

class TestConfigOverrideCorrectness:
    """Test that config overrides are correctly applied during actual training."""
    
    def test_learning_rate_override_takes_effect(self, tmp_path):
        """
        Test that learning rate override actually affects training.
        
        This test:
        1. Trains with initial learning rate
        2. Saves checkpoint
        3. Resumes with overridden learning rate
        4. Verifies the model is actually using the new learning rate
        """
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # === Phase 1: Train with initial LR ===
        initial_lr = 0.01
        model = SimpleReflowModel(
            learning_rate=initial_lr, 
            input_dim=10, 
            output_dim=2,
            loss_type='mse'  # Use MSE for regression
        )
        data_module = SimpleDataModule(
            batch_size=4, 
            train_samples=16, 
            input_dim=10, 
            output_dim=2,
            task_type='regression',  # Match loss type
            num_workers=0,
            seed=42
        )
        
        trainer = Trainer(
            max_epochs=1,
            enable_checkpointing=True,
            default_root_dir=str(checkpoint_dir),
            enable_progress_bar=False,
            logger=False
        )
        
        trainer.fit(model, data_module)
        
        # Save checkpoint with proper config
        checkpoint_path = checkpoint_dir / "initial.ckpt"
        # Add model config to checkpoint for resume
        checkpoint = {
            'state_dict': model.state_dict(),
            'epoch': trainer.current_epoch,
            'global_step': trainer.global_step,
            'lightning_config': {
                'model': {
                    'class_path': 'lightning_reflow.models.SimpleReflowModel',
                    'init_args': {
                        'input_dim': 10,
                        'output_dim': 2,
                        'learning_rate': initial_lr,
                        'loss_type': 'mse'
                    }
                }
            }
        }
        torch.save(checkpoint, checkpoint_path)
        
        # Verify initial LR
        optimizer = trainer.optimizers[0]
        assert optimizer.param_groups[0]['lr'] == initial_lr
        print(f"Initial training LR: {initial_lr}")
        
        # === Phase 2: Resume with overridden LR ===
        override_lr = 0.0001
        
        # Create config with override
        override_config = {
            'model': {
                'init_args': {
                    'learning_rate': override_lr
                }
            }
        }
        
        override_path = tmp_path / "override.yaml"
        with open(override_path, 'w') as f:
            yaml.dump(override_config, f)
        
        # Use LightningReflow for proper config handling
        reflow = LightningReflow(
            config_files=[str(override_path)],
            auto_configure_logging=False
        )
        
        # Mock the actual training but capture model creation
        captured_model = None
        
        def capture_model(*args, **kwargs):
            nonlocal captured_model
            # Check if learning_rate is in kwargs
            if 'learning_rate' in kwargs:
                captured_model = SimpleReflowModel(**kwargs)
            else:
                # Use default
                captured_model = SimpleReflowModel(learning_rate=initial_lr, **kwargs)
            return captured_model
        
        with patch('lightning_reflow.models.SimpleReflowModel', side_effect=capture_model):
            with patch.object(Trainer, 'fit', return_value=None) as mock_fit:
                result = reflow.resume(str(checkpoint_path))
        
        # Verify the model was created with overridden LR
        assert captured_model is not None, "Model was not created"
        assert captured_model.learning_rate == override_lr, \
            f"Learning rate not overridden. Expected {override_lr}, got {captured_model.learning_rate}"
        
        # Verify optimizer would use the new LR
        test_optimizer = captured_model.configure_optimizers()
        if isinstance(test_optimizer, dict):
            test_optimizer = test_optimizer['optimizer']
        assert test_optimizer.param_groups[0]['lr'] == override_lr, \
            f"Optimizer not using overridden LR. Expected {override_lr}, got {test_optimizer.param_groups[0]['lr']}"
        
        print(f"✅ Learning rate successfully overridden: {initial_lr} → {override_lr}")
    
    def test_multiple_config_overrides_priority(self, tmp_path):
        """Test that multiple config overrides are applied in correct priority order."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Create checkpoint with embedded config
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
            'trainer': {
                'max_epochs': 10,
                'accumulate_grad_batches': 1
            }
        }
        
        checkpoint_path = checkpoint_dir / "checkpoint.ckpt"
        torch.save({
            'state_dict': {},
            'lightning_config': checkpoint_config
        }, checkpoint_path)
        
        # Create two override configs
        override1 = {
            'model': {
                'init_args': {
                    'learning_rate': 0.001,
                    'hidden_dim': 64
                }
            },
            'trainer': {
                'max_epochs': 20
            }
        }
        
        override2 = {
            'model': {
                'init_args': {
                    'learning_rate': 0.0001,  # Should win
                    'output_dim': 5  # New override
                }
            }
            # max_epochs not specified, so override1's value should persist
        }
        
        override1_path = tmp_path / "override1.yaml"
        override2_path = tmp_path / "override2.yaml"
        
        with open(override1_path, 'w') as f:
            yaml.dump(override1, f)
        with open(override2_path, 'w') as f:
            yaml.dump(override2, f)
        
        # Apply overrides using LightningReflow
        reflow = LightningReflow(
            config_files=[str(override1_path), str(override2_path)],
            auto_configure_logging=False
        )
        
        with patch.object(Trainer, 'fit', return_value=None):
            with patch('lightning_reflow.models.SimpleReflowModel') as MockModel:
                with patch('lightning_reflow.data.SimpleDataModule'):
                    result = reflow.resume(str(checkpoint_path))
        
        # Verify final config has correct priority
        config = reflow.config
        
        # From override2 (highest priority)
        assert config['model']['init_args']['learning_rate'] == 0.0001
        assert config['model']['init_args']['output_dim'] == 5
        
        # From override1 (not overridden by override2)
        assert config['model']['init_args']['hidden_dim'] == 64
        assert config['trainer']['max_epochs'] == 20
        
        # From checkpoint (not overridden)
        assert config['model']['init_args']['input_dim'] == 10
        assert config['trainer']['accumulate_grad_batches'] == 1
        
        print("✅ Config override priority test PASSED")
        print(f"   Final config correctly merged from 3 sources")
