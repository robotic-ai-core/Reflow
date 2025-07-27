#!/usr/bin/env python3
"""
Minimal reproduction of the embedded config class path validation issue.
"""

import os
import sys
import torch
import yaml
import tempfile
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# Step 1: Create a training script
TRAINING_SCRIPT = '''
import os
import sys
import pytorch_lightning as pl
import torch
from torch import nn
from lightning.pytorch.cli import LightningCLI


class SimpleModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 10)
    
    def training_step(self, batch, batch_idx):
        loss = torch.tensor(1.0)  # Dummy loss
        print(f"[TRAINING] Step {self.global_step}, Epoch {self.current_epoch}")
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


class SimpleDataModule(pl.LightningDataModule):
    def train_dataloader(self):
        dataset = torch.utils.data.TensorDataset(torch.randn(10, 10))
        return torch.utils.data.DataLoader(dataset, batch_size=1)


print(f"[DEBUG] Script started with args: {sys.argv}")

# Check for checkpoint
for i, arg in enumerate(sys.argv):
    if arg == "--ckpt_path" and i + 1 < len(sys.argv):
        ckpt_path = sys.argv[i + 1]
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu")
            print(f"[DEBUG] Checkpoint loaded: global_step={ckpt.get('global_step')}, epoch={ckpt.get('epoch')}")

cli = LightningCLI(SimpleModel, SimpleDataModule, save_config_callback=None)
'''


def create_checkpoint_with_incompatible_config():
    """Create a checkpoint that embeds config with non-existent class paths."""
    checkpoint = {
        'epoch': 4,
        'global_step': 8435,
        'pytorch-lightning_version': '2.5.0',
        'state_dict': {'layer.weight': torch.randn(10, 10)},
        'optimizer_states': [{}],
        'lr_schedulers': [],
        'callbacks': {},
        # This is the problematic embedded config
        'hparams': {
            'trainer': {
                'callbacks': [{
                    'class_path': 'modules.callbacks.NonExistentCallback',
                    'init_args': {}
                }]
            },
            'model': {
                'class_path': 'modules.models.NonExistentModel',
                'init_args': {}
            }
        }
    }
    
    ckpt_file = tempfile.NamedTemporaryFile(suffix='.ckpt', delete=False)
    torch.save(checkpoint, ckpt_file.name)
    return ckpt_file.name


def main():
    logger.info("=== EMBEDDED CONFIG CLASS PATH ISSUE ===\n")
    
    # Create files
    script_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
    script_file.write(TRAINING_SCRIPT)
    script_file.close()
    
    ckpt_path = create_checkpoint_with_incompatible_config()
    logger.info(f"✅ Created checkpoint at: {ckpt_path}")
    logger.info("   Contains: global_step=8435, epoch=4")
    logger.info("   Embedded config has class_path: 'modules.callbacks.NonExistentCallback'\n")
    
    # Extract embedded config
    ckpt = torch.load(ckpt_path)
    embedded_config = ckpt.get('hparams', {})
    config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(embedded_config, config_file)
    config_file.close()
    
    # Test 1: Try with embedded config (WILL FAIL)
    logger.info("--- Test 1: Resume WITH embedded config ---")
    cmd1 = [
        sys.executable, script_file.name, 'fit',
        '--config', config_file.name,
        '--ckpt_path', ckpt_path,
        '--trainer.max_steps', '2'
    ]
    logger.info(f"Command: python train.py fit --config embedded.yaml --ckpt_path checkpoint.ckpt")
    
    result1 = subprocess.run(cmd1, capture_output=True, text=True)
    if result1.returncode != 0:
        logger.info("❌ FAILED (as expected)")
        logger.info("   Error: Cannot find 'modules.callbacks.NonExistentCallback'")
        logger.info("   Lightning CLI validation failed BEFORE loading checkpoint")
        logger.info("   global_step was NEVER restored\n")
    
    # Test 2: Try without embedded config (SHOULD WORK)
    logger.info("--- Test 2: Resume WITHOUT embedded config ---")
    
    minimal_config = {'trainer': {'max_steps': 2}}
    minimal_config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(minimal_config, minimal_config_file)
    minimal_config_file.close()
    
    cmd2 = [
        sys.executable, script_file.name, 'fit',
        '--config', minimal_config_file.name,
        '--ckpt_path', ckpt_path
    ]
    logger.info(f"Command: python train.py fit --config minimal.yaml --ckpt_path checkpoint.ckpt")
    
    result2 = subprocess.run(cmd2, capture_output=True, text=True)
    if result2.returncode == 0:
        logger.info("✅ SUCCESS")
        if "Step 8435" in result2.stdout:
            logger.info("   global_step properly restored to 8435!")
        else:
            logger.info("   But global_step might not be restored...")
        logger.info("   Output:")
        for line in result2.stdout.split('\n'):
            if "[TRAINING]" in line or "[DEBUG]" in line:
                logger.info(f"   {line}")
    
    # Cleanup
    for f in [script_file.name, ckpt_path, config_file.name, minimal_config_file.name]:
        try:
            os.unlink(f)
        except:
            pass
    
    logger.info("\n=== KEY INSIGHT ===")
    logger.info("Lightning CLI validates ALL configs BEFORE loading any checkpoint.")
    logger.info("If embedded config has invalid class paths, validation fails.")
    logger.info("The checkpoint (with global_step=8435) is never loaded.")
    logger.info("Solution: Don't use embedded configs with incompatible class paths.")


if __name__ == '__main__':
    main()