"""
Unit tests for SimpleReflowModel.

Tests the basic functionality of the minimal Lightning model used for testing.
"""

import pytest
import torch
from unittest.mock import patch, Mock

from lightning_reflow.models import SimpleReflowModel


class TestSimpleReflowModel:
    """Test SimpleReflowModel functionality."""
    
    def test_model_initialization(self):
        """Test model initialization with default parameters."""
        model = SimpleReflowModel()
        
        assert model.input_dim == 784
        assert model.hidden_dim == 128
        assert model.output_dim == 10
        assert model.learning_rate == 1e-3
        assert model.loss_type == "mse"
    
    def test_model_initialization_custom_params(self):
        """Test model initialization with custom parameters."""
        model = SimpleReflowModel(
            input_dim=512,
            hidden_dim=256,
            output_dim=5,
            learning_rate=0.01,
            loss_type="cross_entropy"
        )
        
        assert model.input_dim == 512
        assert model.hidden_dim == 256
        assert model.output_dim == 5
        assert model.learning_rate == 0.01
        assert model.loss_type == "cross_entropy"
    
    def test_invalid_loss_type(self):
        """Test that invalid loss type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported loss type"):
            SimpleReflowModel(loss_type="invalid_loss")
    
    def test_forward_pass(self):
        """Test forward pass with different input shapes."""
        model = SimpleReflowModel(input_dim=100, output_dim=5)
        
        # Test 2D input (batch_size, input_dim)
        x = torch.randn(4, 100)
        output = model(x)
        assert output.shape == (4, 5)
        
        # Test 3D input (should be flattened)
        x = torch.randn(4, 10, 10)  # Total features = 100
        output = model(x)
        assert output.shape == (4, 5)
        
        # Test 4D input (should be flattened)
        x = torch.randn(4, 5, 5, 4)  # Total features = 100
        output = model(x)
        assert output.shape == (4, 5)
    
    def test_training_step_mse_loss(self, sample_batch):
        """Test training step with MSE loss."""
        model = SimpleReflowModel(
            input_dim=784,
            output_dim=10,
            loss_type="mse"
        )
        
        # Prepare batch for regression
        batch = {
            "input": torch.randn(4, 784),
            "target": torch.randn(4, 10)
        }
        
        # Mock logging
        model.log = Mock()
        
        loss = model.training_step(batch, 0)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        model.log.assert_called()
    
    def test_training_step_cross_entropy_loss(self):
        """Test training step with cross-entropy loss."""
        model = SimpleReflowModel(
            input_dim=784,
            output_dim=10,
            loss_type="cross_entropy"
        )
        
        # Prepare batch for classification
        batch = {
            "input": torch.randn(4, 784),
            "target": torch.randint(0, 10, (4,))
        }
        
        # Mock logging
        model.log = Mock()
        
        loss = model.training_step(batch, 0)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        model.log.assert_called()
    
    def test_training_step_cross_entropy_one_hot_targets(self):
        """Test training step with one-hot encoded targets."""
        model = SimpleReflowModel(
            input_dim=784,
            output_dim=10,
            loss_type="cross_entropy"
        )
        
        # Prepare batch with one-hot targets
        batch = {
            "input": torch.randn(4, 784),
            "target": torch.eye(10)[torch.randint(0, 10, (4,))]  # One-hot encoded
        }
        
        # Mock logging
        model.log = Mock()
        
        loss = model.training_step(batch, 0)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        model.log.assert_called()
    
    def test_validation_step_classification(self):
        """Test validation step with classification task."""
        model = SimpleReflowModel(
            input_dim=784,
            output_dim=10,
            loss_type="cross_entropy"
        )
        
        batch = {
            "input": torch.randn(4, 784),
            "target": torch.randint(0, 10, (4,))
        }
        
        # Mock logging
        model.log = Mock()
        
        loss = model.validation_step(batch, 0)
        
        assert isinstance(loss, torch.Tensor)
        # Should log both loss and accuracy for classification
        assert model.log.call_count >= 2  # val_loss and val_acc
    
    def test_validation_step_regression(self):
        """Test validation step with regression task."""
        model = SimpleReflowModel(
            input_dim=784,
            output_dim=10,
            loss_type="mse"
        )
        
        batch = {
            "input": torch.randn(4, 784),
            "target": torch.randn(4, 10)
        }
        
        # Mock logging
        model.log = Mock()
        
        loss = model.validation_step(batch, 0)
        
        assert isinstance(loss, torch.Tensor)
        # Should only log loss for regression
        model.log.assert_called_with("val_loss", loss, on_epoch=True, prog_bar=True)
    
    def test_test_step(self):
        """Test that test step uses validation step logic."""
        model = SimpleReflowModel()
        batch = {
            "input": torch.randn(4, 784),
            "target": torch.randn(4, 10)
        }
        
        # Mock validation_step
        with patch.object(model, 'validation_step') as mock_val_step:
            mock_val_step.return_value = torch.tensor(0.5)
            
            result = model.test_step(batch, 0)
            
            mock_val_step.assert_called_once_with(batch, 0)
            assert result == torch.tensor(0.5)
    
    def test_configure_optimizers(self):
        """Test optimizer configuration."""
        model = SimpleReflowModel(learning_rate=0.01)
        
        optimizer = model.configure_optimizers()
        
        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.param_groups[0]['lr'] == 0.01
    
    def test_on_train_start_logging(self):
        """Test hyperparameter logging at training start."""
        model = SimpleReflowModel(learning_rate=0.005, hidden_dim=256)
        
        # Mock logging
        model.log = Mock()
        
        model.on_train_start()
        
        # Should log hyperparameters
        model.log.assert_any_call("hp/learning_rate", 0.005)
        model.log.assert_any_call("hp/hidden_dim", 256)
    
    def test_model_gradient_flow(self):
        """Test that gradients flow through the model correctly."""
        model = SimpleReflowModel(input_dim=10, hidden_dim=16, output_dim=2)
        
        # Create sample data
        x = torch.randn(2, 10, requires_grad=True)
        target = torch.randn(2, 2)
        
        # Forward pass
        output = model(x)
        loss = model.criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
    
    def test_model_save_hyperparameters(self):
        """Test that hyperparameters are saved correctly."""
        model = SimpleReflowModel(
            input_dim=512,
            hidden_dim=256,
            learning_rate=0.01
        )
        
        # Check that hparams are saved
        assert hasattr(model, 'hparams')
        assert model.hparams.input_dim == 512
        assert model.hparams.hidden_dim == 256
        assert model.hparams.learning_rate == 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])