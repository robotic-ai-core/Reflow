"""Simple test model for Lightning Reflow testing."""

import torch
import torch.nn as nn
import lightning.pytorch as pl
from typing import Dict, Any, Optional


class SimpleReflowModel(pl.LightningModule):
    """
    Simple model for testing Lightning Reflow functionality.
    
    This model provides a minimal implementation for testing CLI,
    pause/resume, W&B integration, and other reflow features.
    """
    
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 128,
        output_dim: int = 10,
        learning_rate: float = 1e-3,
        loss_type: str = "mse"
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.loss_type = loss_type
        
        # Simple MLP
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Loss function
        if loss_type == "mse":
            self.criterion = nn.MSELoss()
        elif loss_type == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Flatten input if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.model(x)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        x, y = batch["input"], batch["target"]
        y_hat = self(x)
        
        # Handle different target formats
        if self.loss_type == "cross_entropy" and y.dim() > 1:
            y = y.argmax(dim=-1)
        
        loss = self.criterion(y_hat, y)
        
        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        x, y = batch["input"], batch["target"]
        y_hat = self(x)
        
        if self.loss_type == "cross_entropy" and y.dim() > 1:
            y = y.argmax(dim=-1)
        
        loss = self.criterion(y_hat, y)
        
        # Calculate accuracy for classification
        if self.loss_type == "cross_entropy":
            preds = torch.argmax(y_hat, dim=-1)
            acc = (preds == y).float().mean()
            self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step."""
        return self.validation_step(batch, batch_idx)
    
    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def on_train_start(self):
        """Called at the beginning of training."""
        self.log("hp/learning_rate", self.learning_rate)
        self.log("hp/hidden_dim", self.hidden_dim)