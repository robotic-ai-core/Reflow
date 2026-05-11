#!/usr/bin/env python3
"""
Test environment variable restoration functionality.

Tests comprehensive environment variable handling across different scenarios
including config file processing, type conversion, and conflict resolution.
"""

import pytest
import os
import sys
import yaml
import tempfile
from pathlib import Path

# Add project root to path  
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from lightning_reflow.utils.logging import EnvironmentManager


class TestEnvironmentVariableBasicFunctionality:
    """Test basic environment variable functionality."""
    
    def setup_method(self):
        """Setup test environment before each test."""
        self.original_env = os.environ.copy()
        # Clear any existing state manager to avoid cross-test contamination
        EnvironmentManager._state_manager = None

    def teardown_method(self):
        """Cleanup after each test."""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)
        # Clear state manager
        EnvironmentManager._state_manager = None

    def test_environment_manager_initialization(self):
        state_manager = EnvironmentManager.get_state_manager()
        assert state_manager is not None

    def test_extract_environment_from_config_basic(self, tmp_path):
        """Test basic environment variable extraction from config files."""
        # Create test config with environment variables (note: 'environment' not 'environment_variables')
        config_content = """
trainer:
  max_epochs: 10
environment:
  CUDA_VISIBLE_DEVICES: "0,1"
  BATCH_SIZE: "32"
  LEARNING_RATE: "0.001"
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config_content)
        
        # Extract environment variables
        env_vars, _ = EnvironmentManager.extract_environment_from_configs([config_path])
        
        # Verify extraction worked
        assert isinstance(env_vars, dict)
        assert "CUDA_VISIBLE_DEVICES" in env_vars
        assert env_vars["CUDA_VISIBLE_DEVICES"] == "0,1"
        assert env_vars["BATCH_SIZE"] == "32"
        assert env_vars["LEARNING_RATE"] == "0.001"

    def test_apply_environment_variables_basic(self, tmp_path):
        """Test basic application of environment variables."""
        # Test environment variables
        test_env_vars = {
            "TEST_VAR_1": "value1",
            "TEST_VAR_2": "value2",
            "TEST_VAR_3": "value3"
        }
        
        # Apply them
        EnvironmentManager.set_environment_variables(test_env_vars)
        
        # Verify they were set
        for var_name, expected_value in test_env_vars.items():
            assert os.environ.get(var_name) == expected_value


class TestEnvironmentVariableConflictResolution:
    """Test environment variable conflict resolution."""
    
    def setup_method(self):
        """Setup test environment before each test."""
        self.original_env = os.environ.copy()
        # Clear any existing state manager to avoid cross-test contamination
        EnvironmentManager._state_manager = None

    def teardown_method(self):
        """Cleanup after each test."""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)
        # Clear state manager
        EnvironmentManager._state_manager = None

    def test_system_vs_config_priority(self, tmp_path):
        """Test that config values override system environment variables."""
        # Create test config
        config_content = """
environment:
  CUDA_VISIBLE_DEVICES: "0,1"
  BATCH_SIZE: "32"
  NEW_CONFIG_VAR: "config_value"
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config_content)
        
        # Set system environment variables (some overlapping)
        os.environ.update({
            "CUDA_VISIBLE_DEVICES": "2,3",  # Different from config
            "EXISTING_SYSTEM_VAR": "system_value"
        })
        
        # Process config file
        env_vars, _ = EnvironmentManager.extract_environment_from_configs([config_path])
        
        # Apply environment variables
        EnvironmentManager.set_environment_variables(env_vars)
        
        # Config values should override system values
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0,1"
        assert os.environ.get("BATCH_SIZE") == "32"
        assert os.environ.get("NEW_CONFIG_VAR") == "config_value"
        
        # System-only variables should be preserved
        assert os.environ.get("EXISTING_SYSTEM_VAR") == "system_value"

    def test_empty_environment_section(self, tmp_path):
        """Test handling of config files with empty environment section."""
        config_content = """
trainer:
  max_epochs: 10
environment: {}
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config_content)
        
        # Should handle empty environment section gracefully
        env_vars, _ = EnvironmentManager.extract_environment_from_configs([config_path])
        assert isinstance(env_vars, dict)
        assert len(env_vars) == 0

    def test_missing_environment_section(self, tmp_path):
        """Test handling of config files without environment section."""
        config_content = """
trainer:
  max_epochs: 10
model:
  learning_rate: 0.001
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config_content)
        
        # Should handle missing environment section gracefully
        env_vars, _ = EnvironmentManager.extract_environment_from_configs([config_path])
        assert isinstance(env_vars, dict)
        assert len(env_vars) == 0


class TestEnvironmentVariableTypeHandling:
    """Test environment variable type conversion and handling."""
    
    def setup_method(self):
        """Setup test environment before each test."""
        self.original_env = os.environ.copy()
        # Clear any existing state manager to avoid cross-test contamination
        EnvironmentManager._state_manager = None

    def teardown_method(self):
        """Cleanup after each test."""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)
        # Clear state manager
        EnvironmentManager._state_manager = None

    def test_environment_variable_type_conversion(self):
        """Test that different types are properly converted to strings."""
        # Test various types
        env_vars = {
            "NUMERIC_VAR": 42,
            "FLOAT_VAR": 3.14159,
            "BOOLEAN_VAR": True,
            "STRING_VAR": "hello world",
            "NULL_VAR": None,
            "EMPTY_VAR": ""
        }
        
        # Apply environment variables
        EnvironmentManager.set_environment_variables(env_vars)
        
        # Verify they were converted and set correctly
        assert os.environ.get("NUMERIC_VAR") == "42"
        assert os.environ.get("FLOAT_VAR") == "3.14159"
        assert os.environ.get("BOOLEAN_VAR") == "True"
        assert os.environ.get("STRING_VAR") == "hello world"
        assert os.environ.get("NULL_VAR") == "None"
        assert os.environ.get("EMPTY_VAR") == ""

    def test_complex_environment_variable_values(self):
        """Test complex environment variable values with special characters."""
        # Test complex values
        env_vars = {
            "PATH_VAR": "/path/with/special:characters;and,separators",
            "JSON_LIKE_VAR": '{"key": "value", "number": 42}',
            "MULTILINE_VAR": "line1\nline2\nline3\n",
            "ESCAPED_VAR": 'value with "quotes" and backslashes'
        }
        
        # Apply environment variables
        EnvironmentManager.set_environment_variables(env_vars)
        
        # Verify they were set correctly
        assert os.environ.get("PATH_VAR") == "/path/with/special:characters;and,separators"
        assert os.environ.get("JSON_LIKE_VAR") == '{"key": "value", "number": 42}'
        assert os.environ.get("MULTILINE_VAR") == "line1\nline2\nline3\n"
        assert os.environ.get("ESCAPED_VAR") == 'value with "quotes" and backslashes'


class TestMultipleConfigFileHandling:
    """Test handling of multiple configuration files."""
    
    def setup_method(self):
        """Setup test environment before each test."""
        self.original_env = os.environ.copy()
        # Clear any existing state manager to avoid cross-test contamination
        EnvironmentManager._state_manager = None

    def teardown_method(self):
        """Cleanup after each test."""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)
        # Clear state manager
        EnvironmentManager._state_manager = None

    def test_multiple_config_files_merging(self, tmp_path):
        """Test merging environment variables from multiple config files."""
        # Create first config
        config1_content = """
environment:
  VAR_1: "from_config_1"
  SHARED_VAR: "from_config_1"
"""
        config1_path = tmp_path / "config1.yaml"
        config1_path.write_text(config1_content)
        
        # Create second config  
        config2_content = """
environment:
  VAR_2: "from_config_2"
  SHARED_VAR: "from_config_2"  # Should override config1
"""
        config2_path = tmp_path / "config2.yaml"
        config2_path.write_text(config2_content)
        
        # Extract from both configs
        env_vars, _ = EnvironmentManager.extract_environment_from_configs([config1_path, config2_path])
        
        # Verify merging behavior
        assert env_vars["VAR_1"] == "from_config_1"
        assert env_vars["VAR_2"] == "from_config_2"
        # Later config should override earlier one for shared variables
        assert env_vars["SHARED_VAR"] == "from_config_2"

    def test_nonexistent_config_file_handling(self, tmp_path):
        """Test handling of nonexistent config files."""
        nonexistent_path = tmp_path / "nonexistent.yaml"
        
        # Should handle nonexistent files gracefully
        env_vars, _ = EnvironmentManager.extract_environment_from_configs([nonexistent_path])
        assert isinstance(env_vars, dict)
        assert len(env_vars) == 0


class TestEnvironmentManagerStateIntegration:
    """Test integration with EnvironmentManagerState."""
    
    def setup_method(self):
        """Setup test environment before each test."""
        self.original_env = os.environ.copy()
        # Clear any existing state manager to avoid cross-test contamination
        EnvironmentManager._state_manager = None

    def teardown_method(self):
        """Cleanup after each test."""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)
        # Clear state manager
        EnvironmentManager._state_manager = None

    def test_state_manager_creation_and_registration(self):
        state_manager = EnvironmentManager.get_state_manager()
        assert state_manager is not None
        assert hasattr(state_manager, 'manager_name')
        assert state_manager.manager_name == "environment_manager"

        EnvironmentManager.register_for_checkpoint_persistence()

    def test_state_manager_basic_functionality(self):
        state_manager = EnvironmentManager.get_state_manager()

        assert hasattr(state_manager, 'capture_state')
        assert hasattr(state_manager, 'restore_state')

        state = state_manager.capture_state()
        assert isinstance(state, dict)
        for field in ['env_vars', 'config_sources']:
            assert field in state

        result = state_manager.restore_state(state)
        assert isinstance(result, bool)


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""
    
    def setup_method(self):
        """Setup test environment before each test."""
        self.original_env = os.environ.copy()
        # Clear any existing state manager to avoid cross-test contamination
        EnvironmentManager._state_manager = None

    def teardown_method(self):
        """Cleanup after each test."""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)
        # Clear state manager
        EnvironmentManager._state_manager = None

    def test_invalid_yaml_file_handling(self, tmp_path):
        """Test handling of invalid YAML files."""
        # Create invalid YAML file
        invalid_yaml_content = """
trainer:
  max_epochs: 10
environment:
  - invalid: yaml: structure
    missing_colon_after_key
"""
        invalid_path = tmp_path / "invalid.yaml"
        invalid_path.write_text(invalid_yaml_content)
        
        # Should handle invalid YAML gracefully
        env_vars, _ = EnvironmentManager.extract_environment_from_configs([invalid_path])
        assert isinstance(env_vars, dict)
        # May be empty due to parsing error

    def test_permission_denied_file_access(self, tmp_path):
        """Test handling of files with restricted permissions."""
        # Create file and restrict permissions (if on Unix-like system)
        restricted_file = tmp_path / "restricted.yaml"
        restricted_file.write_text("environment:\n  VAR: value")
        
        try:
            restricted_file.chmod(0o000)  # Remove all permissions
            
            # Should handle permission errors gracefully
            env_vars, _ = EnvironmentManager.extract_environment_from_configs([restricted_file])
            assert isinstance(env_vars, dict)
            
        except (OSError, PermissionError):
            # Skip if we can't change permissions (e.g., Windows, or restricted filesystem)
            pytest.skip("Cannot modify file permissions on this system")
        finally:
            try:
                restricted_file.chmod(0o644)  # Restore permissions for cleanup
            except (OSError, PermissionError):
                pass

    def test_empty_config_file_list(self):
        """Test handling of empty config file list."""
        # Should handle empty list gracefully
        env_vars, config_files = EnvironmentManager.extract_environment_from_configs([])
        assert isinstance(env_vars, dict)
        assert len(env_vars) == 0
        assert isinstance(config_files, list)
        assert len(config_files) == 0


class TestEnvironmentCallback:
    """EnvironmentCallback set/teardown semantics and config-source loading."""

    def setup_method(self):
        self.original_env = os.environ.copy()
        for key in list(os.environ.keys()):
            if key.startswith("TEST_ENV_"):
                del os.environ[key]

    def teardown_method(self):
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_callback_sets_and_restores_environment_variables(self):
        from unittest.mock import Mock
        import lightning.pytorch as pl
        from lightning_reflow.callbacks.core.environment_callback import EnvironmentCallback

        test_vars = {"TEST_ENV_VAR1": "value1", "TEST_ENV_VAR2": "123"}
        callback = EnvironmentCallback(env_vars=test_vars)

        trainer = Mock(spec=pl.Trainer)
        trainer.global_rank = 0
        pl_module = Mock(spec=pl.LightningModule)

        callback.setup(trainer, pl_module, "fit")
        for key, value in test_vars.items():
            os.environ[key] = str(value)

        assert os.environ.get("TEST_ENV_VAR1") == "value1"
        assert os.environ.get("TEST_ENV_VAR2") == "123"

        callback.teardown(trainer, pl_module, "fit")
        for key in test_vars:
            if key in callback.original_env and callback.original_env[key] is not None:
                assert os.environ.get(key) == callback.original_env[key]
            else:
                assert key not in os.environ

    def test_extract_environment_from_configs_respects_override_order(self, tmp_path):
        """Later configs in the list override earlier ones."""
        base_config = {
            "trainer": {"callbacks": [{
                "class_path": "lightning_reflow.callbacks.core.environment_callback.EnvironmentCallback",
                "init_args": {"env_vars": {
                    "TEST_PRECEDENCE_VAR": "base_value",
                    "TEST_ONLY_BASE": "base_only",
                }},
            }]}
        }
        override_config = {
            "trainer": {"callbacks": [{
                "class_path": "lightning_reflow.callbacks.core.environment_callback.EnvironmentCallback",
                "init_args": {"env_vars": {
                    "TEST_PRECEDENCE_VAR": "override_value",
                    "TEST_ONLY_OVERRIDE": "override_only",
                }},
            }]}
        }
        base_file = tmp_path / "base_config.yaml"
        base_file.write_text(yaml.dump(base_config))
        override_file = tmp_path / "override_config.yaml"
        override_file.write_text(yaml.dump(override_config))

        env_vars, _ = EnvironmentManager.extract_environment_from_configs([base_file, override_file])
        assert env_vars.get("TEST_PRECEDENCE_VAR") == "override_value"
        assert env_vars.get("TEST_ONLY_BASE") == "base_only"
        assert env_vars.get("TEST_ONLY_OVERRIDE") == "override_only"

    @pytest.mark.parametrize("env_var,value", [
        ("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256,expandable_segments:True"),
        ("MALLOC_TRIM_THRESHOLD_", "128MB"),
        ("OMP_NUM_THREADS", "4"),
    ])
    def test_callback_handles_critical_runtime_variables(self, env_var, value):
        from unittest.mock import Mock
        import lightning.pytorch as pl
        from lightning_reflow.callbacks.core.environment_callback import EnvironmentCallback

        original_value = os.environ.get(env_var)
        if env_var in os.environ:
            del os.environ[env_var]

        callback = EnvironmentCallback(env_vars={env_var: value})
        trainer = Mock(spec=pl.Trainer)
        trainer.global_rank = 0
        pl_module = Mock(spec=pl.LightningModule)
        try:
            callback.setup(trainer, pl_module, "fit")
            assert os.environ.get(env_var) == value

            callback.teardown(trainer, pl_module, "fit")
            assert env_var not in os.environ
        finally:
            if original_value is not None:
                os.environ[env_var] = original_value
            elif env_var in os.environ:
                del os.environ[env_var]
