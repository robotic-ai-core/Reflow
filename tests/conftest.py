"""
Pytest configuration and fixtures for Lightning Reflow tests.

This provides comprehensive fixtures for testing all Lightning Reflow functionality
including CLI, pause/resume, W&B integration, and callback systems.
"""

import pytest
import tempfile
import torch
import yaml
from pathlib import Path
from unittest.mock import Mock, patch

# Test data fixtures
@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_config():
    """Sample Lightning config for testing."""
    return {
        'model': {
            'class_path': 'lightning_reflow.models.SimpleReflowModel',
            'init_args': {
                'input_dim': 784,
                'hidden_dim': 128,
                'output_dim': 10,
                'learning_rate': 0.001
            }
        },
        'data': {
            'class_path': 'lightning_reflow.data.SimpleDataModule',
            'init_args': {
                'batch_size': 32,
                'train_samples': 100,
                'val_samples': 20,
                'test_samples': 20
            }
        },
        'trainer': {
            'max_epochs': 2,
            'enable_checkpointing': True,
            'logger': {
                'class_path': 'lightning.pytorch.loggers.WandbLogger',
                'init_args': {
                    'project': 'test-project',
                    'offline': True
                }
            }
        }
    }


@pytest.fixture
def config_file(sample_config, temp_dir):
    """Create a temporary config file."""
    config_path = temp_dir / "test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(sample_config, f)
    return str(config_path)


@pytest.fixture
def mock_wandb_logger():
    """Mock WandbLogger for testing."""
    logger = Mock()
    logger.id = 'test-run-123'
    logger.resume = 'allow'
    logger.project = 'test-project'
    logger.log_metrics = Mock()
    return logger


@pytest.fixture
def mock_trainer(mock_wandb_logger):
    """Mock Lightning Trainer for testing."""
    trainer = Mock()
    trainer.logger = mock_wandb_logger
    trainer.max_epochs = 10
    trainer.current_epoch = 5
    trainer.global_step = 1000
    trainer.checkpoint_callback = Mock()
    trainer.callback_metrics = {}
    return trainer


# Opt-in wandb mocking. Tests that need to avoid hitting real wandb should
# request this fixture explicitly. WandbLogger(offline=True) is sufficient
# for most tests; use this fixture only when bypassing wandb entirely.
@pytest.fixture
def mock_wandb():
    """Mock wandb.{init,log,finish,config} for tests that need to bypass it."""
    with patch('wandb.init') as mock_init, \
         patch('wandb.log') as mock_log, \
         patch('wandb.finish') as mock_finish, \
         patch('wandb.config') as mock_config:
        mock_init.return_value = Mock()
        mock_config.update = Mock()
        yield {
            'init': mock_init,
            'log': mock_log,
            'finish': mock_finish,
            'config': mock_config,
        }


@pytest.fixture
def mock_subprocess():
    """Mock subprocess for CLI testing."""
    with patch('subprocess.run') as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""
        yield mock_run


# Test data generators
def create_sample_batch(batch_size=4, input_dim=784, output_dim=10, task_type="classification"):
    """Create a sample batch for testing."""
    inputs = torch.randn(batch_size, input_dim)
    
    if task_type == "classification":
        targets = torch.randint(0, output_dim, (batch_size,))
    else:
        targets = torch.randn(batch_size, output_dim)
    
    return {"input": inputs, "target": targets}


@pytest.fixture
def sample_batch():
    """Sample batch fixture."""
    return create_sample_batch()


