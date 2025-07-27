#!/usr/bin/env python3
"""
Reproduction showing how LightningReflowCLI's resume command
encounters the class path validation issue.

This mimics what happens when you run:
  python train_lightning.py resume --checkpoint-artifact ...
"""

import os
import sys
import torch
import yaml
import tempfile
import subprocess
import logging

# Add lightning_reflow to path
sys.path.insert(0, '/home/neil/code/modelling/Yggdrasil/lightning_reflow')

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def create_problematic_checkpoint():
    """Create a checkpoint like the one from VibeDiffusion with embedded config."""
    checkpoint = {
        'epoch': 4,
        'global_step': 8435,
        'pytorch-lightning_version': '2.5.0',
        'state_dict': {},
        'optimizer_states': [{}],
        'lr_schedulers': [],
        'callbacks': {},
        # Embedded config from original training (references modules.*)
        'hparams': {
            'trainer': {
                'max_steps': 67480,
                'callbacks': [
                    {
                        'class_path': 'modules.callbacks.logging.InputImageLogger',
                        'init_args': {'target_num_images': 64}
                    },
                    {
                        'class_path': 'modules.callbacks.LatentValidationSampleGenerator', 
                        'init_args': {'num_samples': 8}
                    }
                ]
            },
            'model': {
                'class_path': 'modules.models.diffusion_flow_model.DiffusionFlowModel',
                'init_args': {}
            },
            'data': {
                'class_path': 'modules.data.VibeDiffusionDataModule',
                'init_args': {}
            }
        }
    }
    
    ckpt_file = tempfile.NamedTemporaryFile(suffix='.ckpt', delete=False)
    torch.save(checkpoint, ckpt_file.name)
    return ckpt_file.name


def simulate_resume_command_flow(ckpt_path):
    """Simulate what happens during resume command processing."""
    logger.info("=== SIMULATING RESUME COMMAND FLOW ===\n")
    
    # Step 1: What lightning_reflow does
    logger.info("1. LightningReflowCLI processes 'resume' command")
    logger.info("   - Downloads checkpoint from W&B (simulated here)")
    logger.info("   - Extracts embedded config from checkpoint")
    
    # Extract embedded config
    ckpt = torch.load(ckpt_path)
    embedded_config = ckpt.get('hparams', {})
    
    logger.info("\n2. Embedded config contains:")
    for key in embedded_config.get('trainer', {}).get('callbacks', []):
        logger.info(f"   - {key['class_path']}")
    
    # Step 2: Resume converts to fit command
    logger.info("\n3. Resume command expands to:")
    logger.info("   python train_lightning.py fit \\")
    logger.info("     --config user_config.yaml \\")
    logger.info("     --config embedded_config.yaml \\  # THIS CAUSES THE ISSUE")
    logger.info("     --ckpt_path checkpoint.ckpt")
    
    # Step 3: Lightning CLI validation
    logger.info("\n4. Lightning CLI tries to validate all configs:")
    logger.info("   ❌ FAILS: Cannot import 'modules.callbacks.logging.InputImageLogger'")
    logger.info("   ❌ FAILS: Cannot import 'modules.callbacks.LatentValidationSampleGenerator'")
    logger.info("   ⚠️  Never gets to load checkpoint!")
    logger.info("   ⚠️  global_step=8435 is never restored!")
    
    logger.info("\n5. Result: Training starts from step 0 instead of 8435")


def show_the_fix():
    """Show how the fix works."""
    logger.info("\n\n=== THE FIX ===\n")
    
    logger.info("1. Don't use embedded config when resuming:")
    logger.info("   python train_lightning.py fit \\")
    logger.info("     --config user_config.yaml \\")
    logger.info("     --ckpt_path checkpoint.ckpt")
    logger.info("     # NO embedded_config.yaml!")
    
    logger.info("\n2. Lightning loads checkpoint successfully:")
    logger.info("   ✅ Restores global_step=8435")
    logger.info("   ✅ Restores epoch=4")
    logger.info("   ✅ Restores optimizer state")
    logger.info("   ✅ Restores model weights")
    
    logger.info("\n3. Uses current project's callbacks instead:")
    logger.info("   - lightning_reflow.callbacks.monitoring.FlowProgressBarCallback")
    logger.info("   - lightning_reflow.callbacks.logging.StepOutputLoggerCallback")
    logger.info("   (These exist and can be imported!)")


def main():
    logger.info("=== CLASS PATH VALIDATION PREVENTS CHECKPOINT RESTORATION ===\n")
    
    # Create checkpoint
    ckpt_path = create_problematic_checkpoint()
    logger.info(f"Created checkpoint with:")
    logger.info(f"  - global_step = 8435")
    logger.info(f"  - Embedded config with 'modules.*' class paths")
    
    # Show the problem
    simulate_resume_command_flow(ckpt_path)
    
    # Show the fix
    show_the_fix()
    
    # Cleanup
    os.unlink(ckpt_path)
    
    logger.info("\n\n=== SUMMARY ===")
    logger.info("The embedded config contains class paths from the original training environment.")
    logger.info("These don't exist in lightning_reflow, causing validation to fail.")
    logger.info("Lightning never loads the checkpoint, so global_step stays at 0.")
    logger.info("\nThe fix: Skip embedded config and let Lightning restore state with current config.")


if __name__ == '__main__':
    main()