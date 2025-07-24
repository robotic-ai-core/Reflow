"""
Unit tests for SimpleDataModule and SimpleDataset.

Tests the data loading functionality of the minimal data pipeline.
"""

import pytest
import torch
from torch.utils.data import DataLoader

from lightning_reflow.data import SimpleDataModule
from lightning_reflow.data.simple_data import SimpleDataset


class TestSimpleDataset:
    """Test SimpleDataset functionality."""
    
    def test_dataset_initialization_defaults(self):
        """Test dataset initialization with default parameters."""
        dataset = SimpleDataset()
        
        assert len(dataset) == 1000
        assert dataset.input_dim == 784
        assert dataset.output_dim == 10
        assert dataset.task_type == "classification"
        assert dataset.noise_level == 0.1
    
    def test_dataset_initialization_custom(self):
        """Test dataset initialization with custom parameters."""
        dataset = SimpleDataset(
            num_samples=500,
            input_dim=100,
            output_dim=5,
            task_type="regression",
            noise_level=0.2,
            seed=42
        )
        
        assert len(dataset) == 500
        assert dataset.input_dim == 100
        assert dataset.output_dim == 5
        assert dataset.task_type == "regression"
        assert dataset.noise_level == 0.2
    
    def test_dataset_classification_data_generation(self):
        """Test data generation for classification task."""
        dataset = SimpleDataset(
            num_samples=100,
            input_dim=50,
            output_dim=3,
            task_type="classification",
            seed=42
        )
        
        # Test data shapes and types
        sample = dataset[0]
        assert "input" in sample
        assert "target" in sample
        
        assert sample["input"].shape == (50,)
        assert sample["target"].dtype == torch.long  # Classification targets should be long
        assert 0 <= sample["target"] < 3  # Should be valid class index
    
    def test_dataset_regression_data_generation(self):
        """Test data generation for regression task."""
        dataset = SimpleDataset(
            num_samples=100,
            input_dim=50,
            output_dim=3,
            task_type="regression",
            seed=42
        )
        
        # Test data shapes and types
        sample = dataset[0]
        assert "input" in sample
        assert "target" in sample
        
        assert sample["input"].shape == (50,)
        assert sample["target"].shape == (3,)
        assert sample["target"].dtype == torch.float32
    
    def test_dataset_reproducibility(self):
        """Test that datasets are reproducible with same seed."""
        dataset1 = SimpleDataset(num_samples=10, seed=42)
        dataset2 = SimpleDataset(num_samples=10, seed=42)
        
        for i in range(10):
            sample1 = dataset1[i]
            sample2 = dataset2[i]
            
            torch.testing.assert_close(sample1["input"], sample2["input"])
            torch.testing.assert_close(sample1["target"], sample2["target"])
    
    def test_dataset_different_seeds(self):
        """Test that different seeds produce different data."""
        dataset1 = SimpleDataset(num_samples=10, seed=42)
        dataset2 = SimpleDataset(num_samples=10, seed=123)
        
        sample1 = dataset1[0]
        sample2 = dataset2[0]
        
        # Should be different with high probability
        assert not torch.equal(sample1["input"], sample2["input"])
    
    def test_dataset_invalid_task_type(self):
        """Test that invalid task type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported task type"):
            SimpleDataset(task_type="invalid_task")
    
    def test_dataset_indexing(self):
        """Test dataset indexing functionality."""
        dataset = SimpleDataset(num_samples=50, seed=42)
        
        # Test valid indices
        for i in [0, 25, 49]:
            sample = dataset[i]
            assert "input" in sample
            assert "target" in sample
        
        # Test invalid indices
        with pytest.raises(IndexError):
            dataset[50]  # Out of bounds
        
        with pytest.raises(IndexError):
            dataset[-1]  # Negative indexing not implemented


class TestSimpleDataModule:
    """Test SimpleDataModule functionality."""
    
    def test_datamodule_initialization_defaults(self):
        """Test data module initialization with defaults."""
        dm = SimpleDataModule()
        
        assert dm.batch_size == 32
        assert dm.num_workers == 0
        assert dm.input_dim == 784
        assert dm.output_dim == 10
        assert dm.task_type == "classification"
        assert dm.train_samples == 1000
        assert dm.val_samples == 200
        assert dm.test_samples == 200
    
    def test_datamodule_initialization_custom(self):
        """Test data module initialization with custom parameters."""
        dm = SimpleDataModule(
            batch_size=16,
            num_workers=2,
            input_dim=128,
            output_dim=5,
            task_type="regression",
            train_samples=500,
            val_samples=100,
            test_samples=50,
            seed=123
        )
        
        assert dm.batch_size == 16
        assert dm.num_workers == 2
        assert dm.input_dim == 128
        assert dm.output_dim == 5
        assert dm.task_type == "regression"
        assert dm.train_samples == 500
        assert dm.val_samples == 100
        assert dm.test_samples == 50
        assert dm.seed == 123
    
    def test_setup_fit_stage(self):
        """Test setup for fit stage."""
        dm = SimpleDataModule(
            train_samples=100,
            val_samples=20,
            input_dim=50,
            output_dim=3,
            seed=42
        )
        
        # Setup should create datasets
        dm.setup(stage="fit")
        
        assert dm.train_dataset is not None
        assert dm.val_dataset is not None
        assert len(dm.train_dataset) == 100
        assert len(dm.val_dataset) == 20
    
    def test_setup_test_stage(self):
        """Test setup for test stage."""
        dm = SimpleDataModule(
            test_samples=50,
            input_dim=50,
            output_dim=3,
            seed=42
        )
        
        # Setup should create test dataset
        dm.setup(stage="test")
        
        assert dm.test_dataset is not None
        assert len(dm.test_dataset) == 50
    
    def test_setup_all_stages(self):
        """Test setup for all stages (None)."""
        dm = SimpleDataModule(
            train_samples=100,
            val_samples=20,
            test_samples=30,
            seed=42
        )
        
        # Setup all stages
        dm.setup(stage=None)
        
        assert dm.train_dataset is not None
        assert dm.val_dataset is not None
        assert dm.test_dataset is not None
        assert len(dm.train_dataset) == 100
        assert len(dm.val_dataset) == 20
        assert len(dm.test_dataset) == 30
    
    def test_setup_train_val_split(self):
        """Test train/val split when val_samples=0."""
        dm = SimpleDataModule(
            train_samples=100,
            val_samples=0,  # Use split instead
            train_val_split=0.8,
            seed=42
        )
        
        dm.setup(stage="fit")
        
        # Should split the training data
        total_samples = len(dm.train_dataset) + len(dm.val_dataset)
        assert total_samples == 100
        assert len(dm.train_dataset) == 80  # 80% of 100
        assert len(dm.val_dataset) == 20   # 20% of 100
    
    def test_dataloaders_creation(self):
        """Test that dataloaders are created correctly."""
        dm = SimpleDataModule(
            batch_size=8,
            num_workers=0,  # Set to 0 for testing
            train_samples=32,
            val_samples=16,
            test_samples=8,
            seed=42
        )
        
        dm.setup()
        
        # Test train dataloader
        train_dl = dm.train_dataloader()
        assert isinstance(train_dl, DataLoader)
        assert train_dl.batch_size == 8
        assert train_dl.dataset == dm.train_dataset
        
        # Test val dataloader
        val_dl = dm.val_dataloader()
        assert isinstance(val_dl, DataLoader)
        assert val_dl.batch_size == 8
        assert val_dl.dataset == dm.val_dataset
        
        # Test test dataloader
        test_dl = dm.test_dataloader()
        assert isinstance(test_dl, DataLoader)
        assert test_dl.batch_size == 8
        assert test_dl.dataset == dm.test_dataset
    
    def test_dataloader_batch_iteration(self):
        """Test iterating through dataloader batches."""
        dm = SimpleDataModule(
            batch_size=4,
            num_workers=0,
            train_samples=12,  # Exactly 3 batches
            input_dim=20,
            output_dim=3,
            task_type="classification",
            seed=42
        )
        
        dm.setup(stage="fit")
        train_dl = dm.train_dataloader()
        
        batches = list(train_dl)
        assert len(batches) == 3  # 12 samples / 4 batch_size
        
        for batch in batches:
            assert "input" in batch
            assert "target" in batch
            assert batch["input"].shape[0] == 4  # batch size
            assert batch["input"].shape[1] == 20  # input dim
            assert batch["target"].shape[0] == 4  # batch size
    
    def test_dataloader_shuffle_behavior(self):
        """Test that train dataloader shuffles while others don't."""
        dm = SimpleDataModule(
            batch_size=4,
            num_workers=0,
            train_samples=16,
            val_samples=8,
            test_samples=8,
            seed=42
        )
        
        dm.setup()
        
        train_dl = dm.train_dataloader()
        val_dl = dm.val_dataloader()
        test_dl = dm.test_dataloader()
        
        # Train should shuffle, val/test should not
        assert train_dl.sampler.shuffle is True
        assert val_dl.sampler.shuffle is False
        assert test_dl.sampler.shuffle is False
    
    def test_hyperparameter_saving(self):
        """Test that hyperparameters are saved correctly."""
        dm = SimpleDataModule(
            batch_size=16,
            input_dim=128,
            learning_rate=0.01  # This would be ignored but should be in hparams
        )
        
        # Check that hparams are saved
        assert hasattr(dm, 'hparams')
        assert dm.hparams.batch_size == 16
        assert dm.hparams.input_dim == 128
    
    def test_consistent_data_across_stages(self):
        """Test that data remains consistent across different setup calls."""
        dm = SimpleDataModule(
            train_samples=50,
            val_samples=10,
            test_samples=20,
            seed=42
        )
        
        # Setup fit stage
        dm.setup(stage="fit")
        train_sample_1 = dm.train_dataset[0]
        
        # Setup again - should be same data
        dm.setup(stage="fit")
        train_sample_2 = dm.train_dataset[0]
        
        torch.testing.assert_close(train_sample_1["input"], train_sample_2["input"])
        torch.testing.assert_close(train_sample_1["target"], train_sample_2["target"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])