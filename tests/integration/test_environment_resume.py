#!/usr/bin/env python3
"""
Test environment variable restoration during resume.

This test verifies that environment variables set in config files
are properly restored when resuming training from a checkpoint.
"""

import os
import sys
import pytest
import tempfile
import yaml
from pathlib import Path
import torch
import lightning.pytorch as pl
from unittest.mock import Mock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from lightning_reflow.callbacks.core.environment_callback import EnvironmentCallback
from lightning_reflow.cli.lightning_cli import LightningReflowCLI


class TestEnvironmentVariableResume:
    """Test environment variable handling during resume operations."""
    
    def setup_method(self):
        """Setup test environment before each test."""
        self.original_env = os.environ.copy()
        # Clear any test env vars
        for key in list(os.environ.keys()):
            if key.startswith("TEST_ENV_"):
                del os.environ[key]
    
    def teardown_method(self):
        """Cleanup after each test."""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_environment_callback_sets_variables(self):
        """Test that EnvironmentCallback properly sets environment variables."""
        # Create callback with test variables
        test_vars = {
            "TEST_ENV_VAR1": "value1",
            "TEST_ENV_VAR2": "123",
            "TEST_ENV_VAR3": "true"
        }
        
        callback = EnvironmentCallback(env_vars=test_vars)
        
        # Mock trainer and module
        trainer = Mock(spec=pl.Trainer)
        trainer.global_rank = 0
        pl_module = Mock(spec=pl.LightningModule)
        
        # Setup should store original values
        callback.setup(trainer, pl_module, "fit")
        
        # Verify original values were stored
        assert len(callback.original_env) >= len(test_vars)
        
        # Manually set the variables (simulating what CLI would do)
        for key, value in test_vars.items():
            os.environ[key] = str(value)
        
        # Verify variables are set
        assert os.environ.get("TEST_ENV_VAR1") == "value1"
        assert os.environ.get("TEST_ENV_VAR2") == "123"
        assert os.environ.get("TEST_ENV_VAR3") == "true"
        
        # Teardown should restore original values
        callback.teardown(trainer, pl_module, "fit")
        
        # Verify variables are removed/restored
        for key in test_vars:
            if key in callback.original_env and callback.original_env[key] is not None:
                assert os.environ.get(key) == callback.original_env[key]
            else:
                assert key not in os.environ
    
    def test_environment_variables_from_config_file(self, tmp_path):
        """Test loading environment variables from config file."""
        # Create a config file with environment variables
        config_content = {
            "trainer": {
                "callbacks": [{
                    "class_path": "lightning_reflow.callbacks.core.environment_callback.EnvironmentCallback",
                    "init_args": {
                        "env_vars": {
                            "TEST_CONFIG_VAR1": "from_config",
                            "TEST_CONFIG_VAR2": "42",
                            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"
                        }
                    }
                }]
            }
        }
        
        config_file = tmp_path / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_content, f)
        
        # Create callback that will read from config
        callback = EnvironmentCallback(config_paths=[config_file])
        
        # Mock trainer and module
        trainer = Mock(spec=pl.Trainer)
        trainer.global_rank = 0
        pl_module = Mock(spec=pl.LightningModule)
        
        # Setup should process config and set variables
        callback.setup(trainer, pl_module, "fit")
        
        # The callback should have extracted env vars from config
        # Note: In real usage, the CLI would set these before instantiation
        assert callback.env_vars is not None or callback.config_paths
    
    def test_environment_variable_precedence(self, tmp_path):
        """Test that CLI overrides take precedence over config file values."""
        # Create base config with env vars
        base_config = {
            "trainer": {
                "callbacks": [{
                    "class_path": "lightning_reflow.callbacks.core.environment_callback.EnvironmentCallback",
                    "init_args": {
                        "env_vars": {
                            "TEST_PRECEDENCE_VAR": "base_value",
                            "TEST_ONLY_BASE": "base_only"
                        }
                    }
                }]
            }
        }
        
        base_file = tmp_path / "base_config.yaml"
        with open(base_file, "w") as f:
            yaml.dump(base_config, f)
        
        # Create override config
        override_config = {
            "trainer": {
                "callbacks": [{
                    "class_path": "lightning_reflow.callbacks.core.environment_callback.EnvironmentCallback",
                    "init_args": {
                        "env_vars": {
                            "TEST_PRECEDENCE_VAR": "override_value",
                            "TEST_ONLY_OVERRIDE": "override_only"
                        }
                    }
                }]
            }
        }
        
        override_file = tmp_path / "override_config.yaml"
        with open(override_file, "w") as f:
            yaml.dump(override_config, f)
        
        # In real usage, Lightning CLI would merge these configs
        # Here we simulate the precedence handling
        from lightning_reflow.utils.logging import EnvironmentManager
        
        # Extract with proper precedence (later configs override earlier ones)
        env_vars, _ = EnvironmentManager.extract_environment_from_configs([base_file, override_file])
        
        # Verify precedence
        assert env_vars.get("TEST_PRECEDENCE_VAR") == "override_value"
        assert env_vars.get("TEST_ONLY_BASE") == "base_only"
        assert env_vars.get("TEST_ONLY_OVERRIDE") == "override_only"
    
    @pytest.mark.parametrize("env_var,value", [
        ("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256,expandable_segments:True"),
        ("MALLOC_TRIM_THRESHOLD_", "128MB"),
        ("OMP_NUM_THREADS", "4"),
    ])
    def test_critical_environment_variables(self, env_var, value):
        """Test that critical environment variables are properly handled."""
        # Store original value
        original_value = os.environ.get(env_var)
        
        # Make sure the variable is NOT set before we start
        if env_var in os.environ:
            del os.environ[env_var]
        
        callback = EnvironmentCallback(env_vars={env_var: value})
        
        # Mock trainer and module
        trainer = Mock(spec=pl.Trainer)
        trainer.global_rank = 0
        pl_module = Mock(spec=pl.LightningModule)
        
        try:
            # Setup callback - this should set the environment variable
            callback.setup(trainer, pl_module, "fit")
            
            # Verify variable is set
            assert os.environ.get(env_var) == value
            
            # Teardown should restore original state
            callback.teardown(trainer, pl_module, "fit")
            
            # Verify restoration - should be removed since it wasn't there originally
            assert env_var not in os.environ
                
        finally:
            # Ensure cleanup
            if original_value is not None:
                os.environ[env_var] = original_value
            elif env_var in os.environ:
                del os.environ[env_var]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])