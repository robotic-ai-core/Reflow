#!/usr/bin/env python3
"""
Minimal test to verify trainer.should_stop behavior in checkpoints.
"""

import torch
import lightning.pytorch as pl
from pathlib import Path
import tempfile

class MinimalCallback(pl.Callback):
    """Minimal pause callback that only handles should_stop."""
    
    def __init__(self):
        self.pause_requested = False
    
    def on_validation_end(self, trainer, pl_module):
        if self.pause_requested:
            print(f"Setting trainer.should_stop = True at step {trainer.global_step}")
            trainer.should_stop = True
    
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # Add metadata to identify pause checkpoints
        checkpoint['pause_metadata'] = {
            'is_pause': trainer.should_stop,
            'global_step': trainer.global_step
        }
        print(f"Saving checkpoint with should_stop={trainer.should_stop}")
    
    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        metadata = checkpoint.get('pause_metadata', {})
        if metadata.get('is_pause'):
            print(f"Detected pause checkpoint, ensuring trainer.should_stop = False")
            trainer.should_stop = False

class SimpleModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)
        
    def training_step(self, batch, batch_idx):
        x = batch[0]
        return torch.nn.functional.mse_loss(self.layer(x), x)
    
    def validation_step(self, batch, batch_idx):
        x = batch[0]
        return torch.nn.functional.mse_loss(self.layer(x), x)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

def test_should_stop_in_checkpoint():
    """Test if setting should_stop=False in on_load_checkpoint works."""
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create data
        data = torch.randn(50, 1)
        dataset = torch.utils.data.TensorDataset(data)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=5)
        val_loader = torch.utils.data.DataLoader(dataset[:10], batch_size=5)
        
        # Phase 1: Train and pause
        print("\n=== Phase 1: Train and pause ===")
        model1 = SimpleModel()
        callback = MinimalCallback()
        
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=tmp_dir,
            filename='pause_checkpoint',
            save_last=True
        )
        
        trainer1 = pl.Trainer(
            max_epochs=3,
            val_check_interval=5,
            callbacks=[callback, checkpoint_callback],
            enable_progress_bar=False,
            logger=False,
            enable_checkpointing=True
        )
        
        # Trigger pause after first validation
        class PauseTrigger(pl.Callback):
            def on_validation_end(self, trainer, pl_module):
                if trainer.global_step >= 5:
                    print(f"Requesting pause at step {trainer.global_step}")
                    callback.pause_requested = True
        
        trainer1.callbacks.append(PauseTrigger())
        
        # Train until pause
        trainer1.fit(model1, train_loader, val_loader)
        
        print(f"\nPhase 1 results:")
        print(f"  Final step: {trainer1.global_step}")
        print(f"  Final should_stop: {trainer1.should_stop}")
        print(f"  Checkpoint saved: {checkpoint_callback.last_model_path}")
        
        # Phase 2: Resume without epoch manipulation
        print("\n=== Phase 2: Resume from checkpoint ===")
        model2 = SimpleModel()
        
        trainer2 = pl.Trainer(
            max_epochs=3,
            val_check_interval=5,
            callbacks=[MinimalCallback()],  # Fresh callback instance
            enable_progress_bar=False,
            logger=False
        )
        
        print(f"Before resume: trainer.should_stop = {trainer2.should_stop}")
        
        # Test direct checkpoint loading
        checkpoint = torch.load(checkpoint_callback.last_model_path, map_location='cpu', weights_only=False)
        print(f"\nCheckpoint contents:")
        print(f"  pause_metadata: {checkpoint.get('pause_metadata', {})}")
        print(f"  epoch: {checkpoint.get('epoch')}")
        print(f"  global_step: {checkpoint.get('global_step')}")
        
        # Check epoch progress
        if 'loops' in checkpoint and 'fit_loop' in checkpoint['loops']:
            epoch_prog = checkpoint['loops']['fit_loop']['epoch_progress']
            print(f"  epoch_progress.current: {epoch_prog['current']}")
            print(f"  epoch_progress.total: {epoch_prog['total']}")
        
        # Resume training
        try:
            trainer2.fit(model2, train_loader, val_loader, ckpt_path=checkpoint_callback.last_model_path)
            print(f"\nPhase 2 completed successfully!")
            print(f"  Final step: {trainer2.global_step}")
            print(f"  Final epoch: {trainer2.current_epoch}")
        except Exception as e:
            print(f"\nPhase 2 failed with error: {e}")
            print(f"  Trainer state at failure:")
            print(f"    should_stop: {trainer2.should_stop}")
            print(f"    current_epoch: {trainer2.current_epoch}")
            print(f"    global_step: {trainer2.global_step}")

if __name__ == "__main__":
    test_should_stop_in_checkpoint()