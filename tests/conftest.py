"""
Pytest configuration and fixtures for Lightning Reflow tests.

This provides comprehensive fixtures for testing all Lightning Reflow functionality
including CLI, pause/resume, W&B integration, and callback systems.
"""

import pytest
import tempfile
import torch
import yaml
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Optional

# Add lightning_reflow to path
sys.path.insert(0, str(Path(__file__).parent.parent))

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
def mock_checkpoint(temp_dir):
    """Create a mock checkpoint with W&B metadata."""
    checkpoint_data = {
        'state_dict': {'model.weight': torch.randn(10, 10)},
        'epoch': 5,
        'global_step': 1000,
        'optimizer_states': [{}],
        'lr_schedulers': [],
        'wandb_run_id': 'test-run-123',
        'pause_callback_metadata': {
            'wandb_run_id': 'test-run-123',
            'pause_timestamp': 1640995200,
            'embedded_config_content': yaml.dump({
                'model': {'class_path': 'lightning_reflow.models.SimpleReflowModel'},
                'trainer': {'max_epochs': 10}
            }),
            'config_source': 'resolved_with_overrides',
            'callback_version': '4.0.0'
        }
    }
    
    checkpoint_path = temp_dir / "test_checkpoint.ckpt"
    torch.save(checkpoint_data, checkpoint_path)
    return str(checkpoint_path)


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


@pytest.fixture
def cli_args_basic():
    """Basic CLI arguments for testing."""
    return [
        'train_lightning.py',
        'fit',
        '--model.learning_rate=0.001',
        '--trainer.max_epochs=2'
    ]


@pytest.fixture
def cli_args_resume(mock_checkpoint):
    """Resume CLI arguments for testing."""
    return [
        'train_lightning.py',
        'resume',
        '--checkpoint-path', mock_checkpoint
    ]


@pytest.fixture
def cli_args_resume_artifact():
    """Resume from artifact CLI arguments."""
    return [
        'train_lightning.py',
        'resume',
        '--checkpoint-artifact', 'entity/project/test-run-123-pause:latest'
    ]


# Mock external dependencies
@pytest.fixture(autouse=True)
def mock_wandb():
    """Auto-mock wandb to avoid external dependencies."""
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
            'config': mock_config
        }


@pytest.fixture
def mock_subprocess():
    """Mock subprocess for CLI testing."""
    with patch('subprocess.run') as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""
        yield mock_run


# Test utilities
class MockKeyboardInput:
    """Mock keyboard input for pause callback testing."""
    
    def __init__(self, inputs=None):
        self.inputs = inputs or []
        self.index = 0
    
    def __call__(self, *args, **kwargs):
        if self.index < len(self.inputs):
            result = self.inputs[self.index]
            self.index += 1
            return result
        return None


@pytest.fixture
def mock_keyboard_input():
    """Fixture for mocking keyboard input."""
    return MockKeyboardInput


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


# Integration test helpers
@pytest.fixture
def integration_test_setup(temp_dir, config_file):
    """Setup for integration tests."""
    return {
        'temp_dir': temp_dir,
        'config_file': config_file,
        'work_dir': temp_dir / 'work',
        'checkpoints_dir': temp_dir / 'checkpoints'
    }


# Error simulation fixtures
@pytest.fixture
def failing_model():
    """Model that fails for error testing."""
    model = Mock()
    model.side_effect = RuntimeError("Simulated model failure")
    return model


@pytest.fixture
def network_error():
    """Simulate network errors for W&B testing."""
    return ConnectionError("Simulated network failure")