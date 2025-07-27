#!/usr/bin/env python3
"""
Minimal script to replicate the class path validation issue that prevents
checkpoint restoration when using embedded configs.

This demonstrates how Lightning CLI validation fails before checkpoint loading
when embedded configs contain class paths that don't exist in the current context.
"""

import os
import sys
import torch
import yaml
import tempfile
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def create_checkpoint_with_embedded_config():
    """Create a checkpoint file with embedded config containing incompatible class paths."""
    logger.info("Creating checkpoint with embedded config...")
    
    # Create a minimal checkpoint
    checkpoint = {
        'epoch': 4,
        'global_step': 8435,
        'pytorch-lightning_version': '2.5.0',
        'state_dict': {
            'model.weight': torch.randn(10, 10),
            'model.bias': torch.randn(10)
        },
        'optimizer_states': [{
            'state': {},
            'param_groups': [{'lr': 0.001}]
        }],
        'lr_schedulers': [],
        'callbacks': {},
        # Embedded config with incompatible class paths
        'hparams': {
            'trainer': {
                'callbacks': [
                    {
                        'class_path': 'modules.callbacks.logging.InputImageLogger',
                        'init_args': {'log_every_n_steps': 100}
                    },
                    {
                        'class_path': 'modules.callbacks.LatentValidationSampleGenerator',
                        'init_args': {'num_samples': 8}
                    }
                ]
            },
            'model': {
                'class_path': 'modules.models.DiffusionModel',
                'init_args': {'in_channels': 3}
            }
        }
    }
    
    # Save checkpoint
    ckpt_path = tempfile.mktemp(suffix='.ckpt')
    torch.save(checkpoint, ckpt_path)
    logger.info(f"Created checkpoint at: {ckpt_path}")
    logger.info(f"  - global_step: {checkpoint['global_step']}")
    logger.info(f"  - epoch: {checkpoint['epoch']}")
    
    return ckpt_path


def create_minimal_training_script():
    """Create a minimal training script that uses Lightning CLI."""
    script_content = '''#!/usr/bin/env python3
import sys
import pytorch_lightning as pl
import torch
from torch import nn
from lightning.pytorch.cli import LightningCLI


class MinimalModel(pl.LightningModule):
    def __init__(self, input_dim=10, hidden_dim=20):
        super().__init__()
        self.save_hyperparameters()
        self.layer = nn.Linear(input_dim, hidden_dim)
    
    def forward(self, x):
        return self.layer(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        print(f"Step {self.global_step}: loss={loss:.4f}")
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


class MinimalDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
    
    def train_dataloader(self):
        # Create dummy data
        dataset = torch.utils.data.TensorDataset(
            torch.randn(100, 10),
            torch.randn(100, 20)
        )
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)


if __name__ == '__main__':
    print(f"Running with args: {sys.argv}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Check if we have a checkpoint path argument
    ckpt_path = None
    for i, arg in enumerate(sys.argv):
        if arg == '--ckpt_path' and i + 1 < len(sys.argv):
            ckpt_path = sys.argv[i + 1]
            print(f"Found checkpoint path: {ckpt_path}")
            
            # Try to load and inspect the checkpoint
            import torch
            ckpt = torch.load(ckpt_path, map_location='cpu')
            print(f"Checkpoint contains global_step: {ckpt.get('global_step', 'MISSING')}")
            print(f"Checkpoint contains epoch: {ckpt.get('epoch', 'MISSING')}")
    
    # Use Lightning CLI
    cli = LightningCLI(
        MinimalModel,
        MinimalDataModule,
        save_config_callback=None,
        parser_kwargs={"parser_mode": "omegaconf"}
    )
'''
    
    script_path = tempfile.mktemp(suffix='_train.py')
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    logger.info(f"Created training script at: {script_path}")
    return script_path


def demonstrate_embedded_config_issue(ckpt_path, train_script):
    """Demonstrate how embedded config causes validation failure."""
    logger.info("\n=== DEMONSTRATING THE ISSUE ===")
    
    # Extract embedded config from checkpoint
    import torch
    ckpt = torch.load(ckpt_path, map_location='cpu')
    embedded_config = ckpt.get('hparams', {})
    
    # Write embedded config to file
    embedded_config_path = tempfile.mktemp(suffix='_embedded.yaml')
    with open(embedded_config_path, 'w') as f:
        yaml.dump(embedded_config, f)
    
    logger.info(f"Extracted embedded config to: {embedded_config_path}")
    logger.info("Embedded config contains:")
    with open(embedded_config_path, 'r') as f:
        for line in f:
            logger.info(f"  {line.rstrip()}")
    
    # Try to resume with embedded config (THIS WILL FAIL)
    logger.info("\n--- Attempt 1: Resume WITH embedded config (will fail) ---")
    cmd = [
        sys.executable, train_script, 'fit',
        '--config', embedded_config_path,
        '--ckpt_path', ckpt_path,
        '--trainer.max_steps', '1'
    ]
    
    logger.info(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error("❌ FAILED as expected!")
        logger.error("Error output:")
        for line in result.stderr.split('\n'):
            if 'error:' in line or 'module' in line:
                logger.error(f"  {line}")
    else:
        logger.info("✅ Succeeded (unexpected!)")
    
    # Try to resume WITHOUT embedded config (THIS SHOULD WORK)
    logger.info("\n--- Attempt 2: Resume WITHOUT embedded config (should work) ---")
    
    # Create minimal config
    minimal_config = {
        'trainer': {
            'max_steps': 1,
            'logger': False
        }
    }
    minimal_config_path = tempfile.mktemp(suffix='_minimal.yaml')
    with open(minimal_config_path, 'w') as f:
        yaml.dump(minimal_config, f)
    
    cmd = [
        sys.executable, train_script, 'fit',
        '--config', minimal_config_path,
        '--ckpt_path', ckpt_path
    ]
    
    logger.info(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        logger.info("✅ SUCCESS! Training resumed properly")
        # Check if global_step was restored
        if 'Step 8435' in result.stdout or 'global_step: 8435' in result.stdout:
            logger.info("✅ global_step was properly restored to 8435!")
        else:
            logger.warning("⚠️  global_step might not have been restored properly")
            logger.info("Output:")
            for line in result.stdout.split('\n'):
                if 'Step' in line or 'global_step' in line:
                    logger.info(f"  {line}")
    else:
        logger.error("❌ FAILED!")
        logger.error(f"Error: {result.stderr}")
    
    # Cleanup
    os.unlink(embedded_config_path)
    os.unlink(minimal_config_path)


def main():
    """Main function to demonstrate the issue."""
    logger.info("=== CLASS PATH VALIDATION ISSUE DEMONSTRATION ===")
    logger.info("This shows how embedded configs with incompatible class paths")
    logger.info("prevent Lightning from loading checkpoints and restoring global_step")
    
    # Create checkpoint with problematic embedded config
    ckpt_path = create_checkpoint_with_embedded_config()
    
    # Create minimal training script
    train_script = create_minimal_training_script()
    
    # Demonstrate the issue
    try:
        demonstrate_embedded_config_issue(ckpt_path, train_script)
    finally:
        # Cleanup
        if os.path.exists(ckpt_path):
            os.unlink(ckpt_path)
        if os.path.exists(train_script):
            os.unlink(train_script)
    
    logger.info("\n=== SUMMARY ===")
    logger.info("1. Embedded config contains class paths like 'modules.callbacks.logging.InputImageLogger'")
    logger.info("2. Lightning CLI validates ALL configs before loading checkpoint")
    logger.info("3. Validation fails because those module paths don't exist")
    logger.info("4. Checkpoint is never loaded, so global_step is not restored")
    logger.info("5. Solution: Don't use embedded config when resuming")


if __name__ == '__main__':
    main()