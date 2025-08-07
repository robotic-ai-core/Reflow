#!/usr/bin/env python3
"""
Test the --wandb-run-id CLI argument functionality.
"""

import pytest
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from lightning_reflow.cli import LightningReflowCLI
from lightning_reflow.core import LightningReflow


class TestWandbRunIdArgument:
    """Test the --wandb-run-id CLI argument."""
    
    def test_wandb_run_id_in_parser(self):
        """Test that --wandb-run-id is added to the resume parser."""
        cli = LightningReflowCLI.__new__(LightningReflowCLI)
        parser = cli._create_resume_parser()
        
        # Test 1: With explicit ID
        args = parser.parse_args([
            '--checkpoint-path', '/path/to/checkpoint.ckpt',
            '--wandb-run-id', 'custom-run-id-123'
        ])
        assert hasattr(args, 'wandb_run_id')
        assert args.wandb_run_id == 'custom-run-id-123'
        
        # Test 2: Flag without value (force new run)
        args = parser.parse_args([
            '--checkpoint-path', '/path/to/checkpoint.ckpt',
            '--wandb-run-id'
        ])
        assert args.wandb_run_id == 'new'
        
        # Test 3: No flag at all
        args = parser.parse_args([
            '--checkpoint-path', '/path/to/checkpoint.ckpt'
        ])
        assert args.wandb_run_id is None
    
    def test_wandb_run_id_passed_to_resume_cli(self):
        """Test that --wandb-run-id is passed to resume_cli method."""
        with tempfile.NamedTemporaryFile(suffix='.ckpt', delete=False) as tmp_ckpt:
            # Create a minimal checkpoint
            checkpoint = {
                'epoch': 5,
                'global_step': 1000,
                'state_dict': {},
                'self_contained_metadata': {
                    'embedded_config_content': 'model:\n  class_path: test.Model\n'
                }
            }
            torch.save(checkpoint, tmp_ckpt.name)
            
            try:
                with patch('sys.argv', ['script', 'resume', 
                                       '--checkpoint-path', tmp_ckpt.name,
                                       '--wandb-run-id', 'my-custom-run']):
                    
                    with patch.object(LightningReflow, 'resume_cli') as mock_resume_cli:
                        with patch('sys.exit'):
                            cli = LightningReflowCLI.__new__(LightningReflowCLI)
                            cli._execute_resume_as_subprocess()
                        
                        # Check that resume_cli was called with wandb_run_id
                        mock_resume_cli.assert_called_once()
                        call_kwargs = mock_resume_cli.call_args[1]
                        assert 'wandb_run_id' in call_kwargs
                        assert call_kwargs['wandb_run_id'] == 'my-custom-run'
            finally:
                Path(tmp_ckpt.name).unlink(missing_ok=True)
    
    def test_explicit_wandb_run_id_overrides_checkpoint(self):
        """Test that explicit --wandb-run-id overrides the checkpoint's run ID."""
        with tempfile.NamedTemporaryFile(suffix='.ckpt', delete=False) as tmp_ckpt:
            # Create checkpoint with embedded W&B run ID
            checkpoint = {
                'epoch': 5,
                'global_step': 1000,
                'state_dict': {},
                'self_contained_metadata': {
                    'wandb_run_id': 'checkpoint-run-id',
                    'embedded_config_content': 'model:\n  class_path: test.Model\n'
                }
            }
            torch.save(checkpoint, tmp_ckpt.name)
            
            try:
                # Need to capture the W&B config before subprocess runs
                wandb_config_content = None
                original_add_wandb = LightningReflow._add_wandb_resume_config
                
                def capture_wandb_config(self, cmd, wandb_run_id, embedded_config_yaml):
                    nonlocal wandb_config_content
                    # Capture the W&B run ID being used
                    wandb_config_content = wandb_run_id
                    return original_add_wandb(self, cmd, wandb_run_id, embedded_config_yaml)
                
                with patch.object(LightningReflow, '_add_wandb_resume_config', capture_wandb_config):
                    with patch('subprocess.run') as mock_run:
                        mock_run.return_value = Mock(returncode=0)
                        
                        with patch('sys.exit'):
                            reflow = LightningReflow()
                            reflow.resume_cli(
                                resume_source=tmp_ckpt.name,
                                wandb_run_id='override-run-id'  # This should override checkpoint's ID
                            )
                    
                # Check that the override run ID was used
                assert wandb_config_content == 'override-run-id'
                    
            finally:
                Path(tmp_ckpt.name).unlink(missing_ok=True)
    
    def test_no_wandb_config_when_no_run_id(self):
        """Test that no W&B config is added when --wandb-run-id is not provided."""
        with tempfile.NamedTemporaryFile(suffix='.ckpt', delete=False) as tmp_ckpt:
            # Create checkpoint WITHOUT W&B run ID
            checkpoint = {
                'epoch': 5,
                'global_step': 1000,
                'state_dict': {},
                'self_contained_metadata': {
                    'embedded_config_content': 'model:\n  class_path: test.Model\n'
                }
            }
            torch.save(checkpoint, tmp_ckpt.name)
            
            try:
                # Mock subprocess.run to capture the command
                with patch('subprocess.run') as mock_run:
                    mock_run.return_value = Mock(returncode=0)
                    
                    with patch('sys.exit'):
                        reflow = LightningReflow()
                        reflow.resume_cli(
                            resume_source=tmp_ckpt.name
                            # No wandb_run_id provided
                        )
                    
                    # Get the command that was executed
                    cmd = mock_run.call_args[0][0]
                    
                    # Count --config arguments
                    config_count = cmd.count('--config')
                    
                    # Should only have one --config for the embedded config
                    # No second --config for W&B since no run ID
                    assert config_count == 1, f"Expected 1 --config, got {config_count}"
                    
            finally:
                Path(tmp_ckpt.name).unlink(missing_ok=True)
    
    def test_wandb_run_id_with_new_run(self):
        """Test using --wandb-run-id to start a new W&B run (not resuming)."""
        with tempfile.NamedTemporaryFile(suffix='.ckpt', delete=False) as tmp_ckpt:
            checkpoint = {
                'epoch': 5,
                'global_step': 1000,
                'state_dict': {},
                'self_contained_metadata': {
                    'embedded_config_content': 'model:\n  class_path: test.Model\n'
                }
            }
            torch.save(checkpoint, tmp_ckpt.name)
            
            try:
                # Capture the W&B run ID being used
                wandb_config_content = None
                original_add_wandb = LightningReflow._add_wandb_resume_config
                
                def capture_wandb_config(self, cmd, wandb_run_id, embedded_config_yaml):
                    nonlocal wandb_config_content
                    wandb_config_content = wandb_run_id
                    return original_add_wandb(self, cmd, wandb_run_id, embedded_config_yaml)
                
                with patch.object(LightningReflow, '_add_wandb_resume_config', capture_wandb_config):
                    with patch('subprocess.run') as mock_run:
                        mock_run.return_value = Mock(returncode=0)
                        
                        with patch('sys.exit'):
                            reflow = LightningReflow()
                            reflow.resume_cli(
                                resume_source=tmp_ckpt.name,
                                wandb_run_id='brand-new-run-id'  # Start fresh run with specific ID
                            )
                    
                    # Should use the specified run ID
                    assert wandb_config_content == 'brand-new-run-id'
                    
            finally:
                Path(tmp_ckpt.name).unlink(missing_ok=True)


    def test_force_new_run_with_wandb_run_id_flag(self):
        """Test that --wandb-run-id without value forces a new W&B run."""
        with tempfile.NamedTemporaryFile(suffix='.ckpt', delete=False) as tmp_ckpt:
            # Create checkpoint WITH embedded W&B run ID
            checkpoint = {
                'epoch': 5,
                'global_step': 1000,
                'state_dict': {},
                'self_contained_metadata': {
                    'wandb_run_id': 'existing-checkpoint-run-id',
                    'embedded_config_content': 'model:\n  class_path: test.Model\n'
                }
            }
            torch.save(checkpoint, tmp_ckpt.name)
            
            try:
                # Track what happens with W&B run ID
                captured_run_id = None
                original_add_wandb = LightningReflow._add_wandb_resume_config
                
                def capture_wandb_config(self, cmd, wandb_run_id, embedded_config_yaml):
                    nonlocal captured_run_id
                    captured_run_id = wandb_run_id
                    # Don't actually add the config to avoid file issues
                    return None
                
                with patch.object(LightningReflow, '_add_wandb_resume_config', capture_wandb_config):
                    with patch('subprocess.run') as mock_run:
                        mock_run.return_value = Mock(returncode=0)
                        
                        with patch('sys.exit'):
                            reflow = LightningReflow()
                            reflow.resume_cli(
                                resume_source=tmp_ckpt.name,
                                wandb_run_id='new'  # Force new run
                            )
                    
                    # Should NOT use the checkpoint's run ID
                    # Should have None to create a new run
                    assert captured_run_id is None
                    
            finally:
                Path(tmp_ckpt.name).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])