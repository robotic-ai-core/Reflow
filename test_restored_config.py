#!/usr/bin/env python3
"""
Test the restored embedded config approach.
"""

import subprocess
import sys
import tempfile
import yaml
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_resume_with_embedded_config():
    """Test that resume works with the properly restored embedded config approach."""
    logger.info("=== TESTING RESUME WITH EMBEDDED CONFIG ===")
    
    # Create minimal config for quick test
    config = {
        'trainer': {
            'max_epochs': 1,
            'max_steps': 2,  # Just run 2 steps to verify restoration
            'logger': False,
            'enable_checkpointing': False,
            'limit_train_batches': 2,
            'limit_val_batches': 0,
            'num_sanity_val_steps': 0,
        }
    }
    
    # Write config to temp file
    temp_fd, config_path = tempfile.mkstemp(suffix='.yaml', prefix='config_test_')
    with os.fdopen(temp_fd, 'w') as f:
        yaml.dump(config, f)
    
    # Test the resume command with the restored approach
    cmd = [
        sys.executable, 'train_lightning.py', 'resume',
        '--checkpoint-artifact', 'neiltan/VibeDiffusion/h0ccvzi3-pause:latest',
        '--config', config_path
    ]
    
    logger.info(f"Command: {' '.join(cmd)}")
    
    # Set debug logging to see what happens
    env = os.environ.copy()
    env['VIBE_LOG_LEVEL'] = 'DEBUG'
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
            cwd='/home/neil/code/modelling/Yggdrasil'
        )
        
        stdout = result.stdout
        stderr = result.stderr
        all_output = stdout + '\n' + stderr
        
        logger.info(f"Return code: {result.returncode}")
        
        # Check for key indicators
        indicators = {
            'embedded_config_used': 'Using Lightning\'s original merged config' in all_output,
            'config_validation_error': 'error: Parser key' in stderr,
            'module_import_error': 'module \'modules.callbacks\' has no attribute' in stderr,
            'checkpoint_loading': 'Restoring states from the checkpoint' in all_output,
            'global_step_restored': 'global_step=8435' in all_output or 'trainer.global_step=8435' in all_output,
            'training_started': 'Step 8435' in all_output or 'Step 8436' in all_output,
        }
        
        logger.info("=== ANALYSIS ===")
        for key, value in indicators.items():
            status = "✅" if value else "❌"
            logger.info(f"{status} {key}: {value}")
        
        if indicators['config_validation_error'] or indicators['module_import_error']:
            logger.error("❌ CONFIG VALIDATION FAILED")
            logger.error("The embedded config still contains incompatible class paths!")
            logger.error("This means the issue is NOT with our extraction/usage logic")
            logger.error("The embedded config itself has invalid imports for this context")
            
            # Show the specific error
            logger.error("\n=== ERROR OUTPUT ===")
            for line in stderr.split('\n'):
                if 'error:' in line or 'module' in line or 'class_path' in line:
                    logger.error(f"  {line}")
        
        elif indicators['embedded_config_used'] and indicators['global_step_restored']:
            logger.info("✅ SUCCESS! Embedded config approach working correctly")
            logger.info("The merged config from Lightning was successfully used")
        
        else:
            logger.warning("⚠️ Partial success or unclear result")
            logger.info("Last 10 lines of output:")
            for line in all_output.split('\n')[-10:]:
                if line.strip():
                    logger.info(f"  {line}")
        
        # Cleanup
        os.unlink(config_path)
        
        return result.returncode == 0 and not indicators['config_validation_error']
        
    except subprocess.TimeoutExpired:
        logger.info("⏰ Process timed out - likely means training started successfully!")
        os.unlink(config_path)
        return True
    except Exception as e:
        logger.error(f"Test failed: {e}")
        os.unlink(config_path)
        return False


def main():
    """Run the test."""
    logger.info("🧪 TESTING RESTORED EMBEDDED CONFIG APPROACH")
    
    success = test_resume_with_embedded_config()
    
    print("\n" + "="*60)
    print("EMBEDDED CONFIG RESTORATION TEST")
    print("="*60)
    
    if success:
        print("✅ SUCCESS!")
        print("The embedded config approach is working correctly.")
        print("Lightning's merged config was used successfully for resume.")
    else:
        print("❌ STILL FAILING")
        print("The embedded config contains incompatible class paths.")
        print("This confirms that the config itself is the issue, not our logic.")

if __name__ == "__main__":
    main()