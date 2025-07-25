"""Simple data module for Lightning Reflow testing."""

import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Optional, Dict, Any, Tuple
import numpy as np


class SimpleDataset(Dataset):
    """Simple synthetic dataset for testing."""
    
    def __init__(
        self,
        num_samples: int = 1000,
        input_dim: int = 784,
        output_dim: int = 10,
        task_type: str = "classification",
        noise_level: float = 0.1,
        seed: Optional[int] = None
    ):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.task_type = task_type
        self.noise_level = noise_level
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Generate synthetic data
        self.data = self._generate_data()
    
    def _generate_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic data."""
        # Create random input data
        inputs = torch.randn(self.num_samples, self.input_dim)
        
        if self.task_type == "classification":
            # Create classification targets
            # Use a simple linear mapping + softmax for realistic class probabilities
            weight = torch.randn(self.input_dim, self.output_dim)
            logits = inputs @ weight
            targets = F.softmax(logits, dim=-1)
            
            # Add some noise and convert to class indices
            noise = torch.randn_like(targets) * self.noise_level
            targets = (targets + noise).argmax(dim=-1)
            
        elif self.task_type == "regression":
            # Create regression targets
            weight = torch.randn(self.input_dim, self.output_dim)
            targets = inputs @ weight
            
            # Add noise
            noise = torch.randn_like(targets) * self.noise_level
            targets = targets + noise
            
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
        
        return inputs, targets
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Support negative indexing
        if idx < 0:
            idx = self.num_samples + idx
        
        # Check bounds
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.num_samples}")
            
        inputs, targets = self.data
        return {
            "input": inputs[idx],
            "target": targets[idx]
        }


class SimpleDataModule(pl.LightningDataModule):
    """
    Simple data module for testing Lightning Reflow functionality.
    
    Provides synthetic data for testing CLI, pause/resume, and other features.
    """
    
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        input_dim: int = 784,
        output_dim: int = 10,
        task_type: str = "classification",
        train_samples: int = 1000,
        val_samples: int = 200,
        test_samples: int = 200,
        noise_level: float = 0.1,
        train_val_split: float = 0.8,
        seed: Optional[int] = 42
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.task_type = task_type
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples
        self.noise_level = noise_level
        self.train_val_split = train_val_split
        self.seed = seed
        
        # Will be set in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        """Set up datasets."""
        if stage == "fit" or stage is None:
            # Create training data
            full_train_dataset = SimpleDataset(
                num_samples=self.train_samples,
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                task_type=self.task_type,
                noise_level=self.noise_level,
                seed=self.seed
            )
            
            # Split into train/val if needed
            if self.val_samples == 0:
                train_size = int(self.train_val_split * len(full_train_dataset))
                val_size = len(full_train_dataset) - train_size
                self.train_dataset, self.val_dataset = random_split(
                    full_train_dataset, [train_size, val_size],
                    generator=torch.Generator().manual_seed(self.seed or 42)
                )
            else:
                self.train_dataset = full_train_dataset
                self.val_dataset = SimpleDataset(
                    num_samples=self.val_samples,
                    input_dim=self.input_dim,
                    output_dim=self.output_dim,
                    task_type=self.task_type,
                    noise_level=self.noise_level,
                    seed=self.seed + 1 if self.seed else None
                )
        
        if stage == "test" or stage is None:
            self.test_dataset = SimpleDataset(
                num_samples=self.test_samples,
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                task_type=self.task_type,
                noise_level=self.noise_level,
                seed=self.seed + 2 if self.seed else None
            )
    
    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0
        )
    
    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0
        )
    
    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0
        )