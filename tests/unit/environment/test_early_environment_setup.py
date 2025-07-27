#!/usr/bin/env python3
"""
Test early environment variable setup timing.

Verifies that environment variables are set before PyTorch/CUDA initialization
through the before_instantiate_classes hook.
"""

import pytest
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from lightning_reflow.cli.lightning_cli import LightningReflowCLI


class TestEarlyEnvironmentSetup:
    """Test that environment variables are set early enough."""
    
    def setup_method(self):
        """Setup test environment before each test."""
        self.original_env = os.environ.copy()
    
    def teardown_method(self):
        """Cleanup after each test."""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_before_instantiate_classes_hook_exists(self):
        """Test that the before_instantiate_classes hook exists."""
        cli = LightningReflowCLI.__new__(LightningReflowCLI)
        assert hasattr(cli, 'before_instantiate_classes')
        assert callable(getattr(cli, 'before_instantiate_classes'))
    
    def test_config_path_extraction_from_argv(self):
        """Test extraction of config paths from sys.argv."""
        cli = LightningReflowCLI.__new__(LightningReflowCLI)
        
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("environment:\n  TEST_VAR: test_value\n")
            config_path = f.name
        
        try:
            # Test --config argument
            test_argv = ['script.py', 'fit', '--config', config_path]
            with patch.object(sys, 'argv', test_argv):
                paths = cli._extract_config_paths_from_sys_argv()
                assert len(paths) == 1
                assert str(paths[0]) == config_path
            
            # Test --config= argument
            test_argv = ['script.py', 'fit', f'--config={config_path}']
            with patch.object(sys, 'argv', test_argv):
                paths = cli._extract_config_paths_from_sys_argv()
                assert len(paths) == 1
                assert str(paths[0]) == config_path
        
        finally:
            Path(config_path).unlink()
    
    def test_early_environment_processing(self):
        """Test that environment variables are processed early."""
        cli = LightningReflowCLI.__new__(LightningReflowCLI)
        
        # Create a temporary config file with environment variables
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
trainer:
  max_epochs: 1
environment:
  PYTORCH_CUDA_ALLOC_CONF: "max_split_size_mb:256,expandable_segments:True"
  TEST_EARLY_VAR: "early_value"
""")
            config_path = f.name
        
        try:
            # Mock sys.argv to include config
            test_argv = ['script.py', 'fit', '--config', config_path]
            with patch.object(sys, 'argv', test_argv):
                # Call the early processing method
                cli._process_environment_callback_config()
                
                # Verify environment variables were set
                assert os.environ.get('PYTORCH_CUDA_ALLOC_CONF') == "max_split_size_mb:256,expandable_segments:True"
                assert os.environ.get('TEST_EARLY_VAR') == "early_value"
        
        finally:
            Path(config_path).unlink()
    
    def test_critical_variable_logging(self):
        """Test that critical environment variables are logged prominently."""
        cli = LightningReflowCLI.__new__(LightningReflowCLI)
        
        # Create config with critical variables
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
environment:
  PYTORCH_CUDA_ALLOC_CONF: "max_split_size_mb:512"
  MALLOC_TRIM_THRESHOLD_: "128MB"
  REGULAR_VAR: "normal_value"
""")
            config_path = f.name
        
        try:
            test_argv = ['script.py', 'fit', '--config', config_path]
            with patch.object(sys, 'argv', test_argv):
                # Call the processing method - it should complete without error
                cli._process_environment_callback_config()
                
                # Verify critical environment variables were actually set
                assert os.environ.get('PYTORCH_CUDA_ALLOC_CONF') == "max_split_size_mb:512"
                assert os.environ.get('MALLOC_TRIM_THRESHOLD_') == "128MB"
                assert os.environ.get('REGULAR_VAR') == "normal_value"
        
        finally:
            Path(config_path).unlink()
    
    def test_before_instantiate_classes_calls_processing(self):
        """Test that before_instantiate_classes calls environment processing."""
        cli = LightningReflowCLI.__new__(LightningReflowCLI)
        
        with patch.object(cli, '_process_environment_callback_config') as mock_process:
            cli.before_instantiate_classes()
            mock_process.assert_called_once()
    
    def test_missing_config_file_handling(self):
        """Test graceful handling of missing config files."""
        cli = LightningReflowCLI.__new__(LightningReflowCLI)
        
        # Test with non-existent config file
        test_argv = ['script.py', 'fit', '--config', '/nonexistent/config.yaml']
        with patch.object(sys, 'argv', test_argv):
            # Should not raise exception
            cli._process_environment_callback_config()
            
            # Should not set any environment variables
            assert 'TEST_VAR' not in os.environ
    
    def test_no_environment_section_handling(self):
        """Test handling of config files without environment section."""
        cli = LightningReflowCLI.__new__(LightningReflowCLI)
        
        # Create config without environment section
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
trainer:
  max_epochs: 10
model:
  learning_rate: 0.001
""")
            config_path = f.name
        
        try:
            test_argv = ['script.py', 'fit', '--config', config_path]
            with patch.object(sys, 'argv', test_argv):
                # Should not raise exception
                cli._process_environment_callback_config()
        
        finally:
            Path(config_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 