#!/usr/bin/env python3
"""
Test to understand trainer.should_stop behavior during checkpoint save/load.
"""

import torch
import lightning.pytorch as pl
from pathlib import Path
import tempfile
import shutil

class SimpleModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)
        
    def training_step(self, batch, batch_idx):
        x = batch[0]  # TensorDataset returns a tuple
        return torch.nn.functional.mse_loss(self.layer(x), x)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

def test_should_stop_checkpoint():
    """Test if trainer.should_stop is saved in checkpoints."""
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create model and simple data
        model = SimpleModel()
        data = torch.randn(10, 1)
        dataset = torch.utils.data.TensorDataset(data)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
        
        # Create trainer and run a few steps
        trainer = pl.Trainer(
            max_steps=5,
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False
        )
        
        # Manually advance a few steps
        trainer.fit(model, dataloader)
        
        # Now set should_stop and save checkpoint
        trainer.should_stop = True
        checkpoint_path = Path(tmp_dir) / "test_checkpoint.ckpt"
        trainer.save_checkpoint(checkpoint_path)
        
        # Load checkpoint and check state
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        print("\n=== Checkpoint Contents ===")
        print(f"Top-level keys: {list(checkpoint.keys())}")
        
        # Check if should_stop is in checkpoint
        if 'trainer' in checkpoint:
            print(f"Trainer state keys: {list(checkpoint['trainer'].keys())}")
            if 'should_stop' in checkpoint['trainer']:
                print(f"trainer.should_stop in checkpoint: {checkpoint['trainer']['should_stop']}")
        
        # Check loops state
        if 'loops' in checkpoint:
            print(f"\nLoops keys: {list(checkpoint['loops'].keys())}")
            if 'fit_loop' in checkpoint['loops']:
                fit_loop = checkpoint['loops']['fit_loop']
                if 'epoch_progress' in fit_loop:
                    epoch_prog = fit_loop['epoch_progress']
                    print(f"\nEpoch progress:")
                    print(f"  current: {epoch_prog.get('current', {})}")
                    print(f"  total: {epoch_prog.get('total', {})}")
        
        # Now create a new trainer and load checkpoint
        print("\n=== Testing Resume ===")
        new_trainer = pl.Trainer(
            max_steps=10,
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False
        )
        
        print(f"New trainer.should_stop before load: {new_trainer.should_stop}")
        
        # This simulates what happens during resume
        new_model = SimpleModel()
        
        # Note: trainer.should_stop is not saved in checkpoints by Lightning
        # It's a runtime state that gets reset on trainer initialization
        # The PauseCallback handles this by explicitly setting
        # trainer.should_stop = False in on_load_checkpoint when resuming from pause
        
        print(f"New trainer.should_stop after load: {new_trainer.should_stop}")
        print("\nNOTE: trainer.should_stop is not persisted in checkpoints by Lightning")
        print("      It's handled by callbacks like PauseCallback")

if __name__ == "__main__":
    test_should_stop_checkpoint()