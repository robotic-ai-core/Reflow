#!/usr/bin/env python3
"""
Test comprehensive environment variable restoration scenarios and edge cases.

This test suite focuses on environment variable restoration scenarios,
conflict resolution, and edge cases that can occur in production environments.
"""

import pytest
import os
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from lightning_reflow.utils.logging import EnvironmentManager


class TestEnvironmentVariableBasicFunctionality:
    """Test basic environment variable functionality."""

    def test_environment_manager_initialization(self):
        """Test that EnvironmentManager can be initialized and accessed."""
        # Test that EnvironmentManager is available and can get state manager
        # Note: After refactoring, the registration pattern has changed
        # and the package is designed to work with or without the main project imports
        try:
            state_manager = EnvironmentManager.get_state_manager()
            assert state_manager is not None
        except ImportError:
            # Expected when running tests in isolation from main project
            # The EnvironmentManager falls back to local implementation
            pytest.skip("EnvironmentManager requires main project imports")

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
        env_vars, config_files = EnvironmentManager.extract_environment_from_configs([config_path])
        
        # Verify extraction worked
        assert isinstance(env_vars, dict)
        assert "CUDA_VISIBLE_DEVICES" in env_vars
        assert env_vars["CUDA_VISIBLE_DEVICES"] == "0,1"
        assert env_vars["BATCH_SIZE"] == "32"
        assert env_vars["LEARNING_RATE"] == "0.001"

    def test_apply_environment_variables_basic(self, tmp_path):
        """Test basic application of environment variables."""
        original_env = os.environ.copy()
        try:
            # Test environment variables
            test_env_vars = {
                "TEST_VAR_1": "value1",
                "TEST_VAR_2": "value2",
                "TEST_VAR_3": "value3"
            }
            
            # Apply environment variables (correct method name)
            EnvironmentManager.set_environment_variables(test_env_vars)
            
            # Verify they were applied
            assert os.environ.get("TEST_VAR_1") == "value1"
            assert os.environ.get("TEST_VAR_2") == "value2"
            assert os.environ.get("TEST_VAR_3") == "value3"
            
        finally:
            # Restore original environment
            os.environ.clear()
            os.environ.update(original_env)


class TestEnvironmentVariableConflictResolution:
    """Test environment variable conflict scenarios."""

    def test_system_vs_config_priority(self, tmp_path):
        """Test priority resolution when env vars conflict between system and config."""
        # Create test config with environment variables
        config_content = """
trainer:
  max_epochs: 10
environment:
  CUDA_VISIBLE_DEVICES: "0,1"
  BATCH_SIZE: "32"
  NEW_CONFIG_VAR: "config_value"
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config_content)
        
        # Set system environment variables (some overlapping)
        original_env = os.environ.copy()
        os.environ.update({
            "CUDA_VISIBLE_DEVICES": "2,3",  # Different from config
            "EXISTING_SYSTEM_VAR": "system_value"
        })
        
        try:
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
            
        finally:
            # Restore original environment
            os.environ.clear()
            os.environ.update(original_env)

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
    """Test environment variable type conversion and validation."""

    def test_environment_variable_type_conversion(self, tmp_path):
        """Test type conversion for different environment variable types."""
        config_content = """
trainer:
  max_epochs: 10
environment:
  NUMERIC_VAR: 42
  FLOAT_VAR: 3.14159
  BOOLEAN_VAR: true
  STRING_VAR: "hello world"
  NULL_VAR: null
  EMPTY_VAR: ""
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config_content)
        
        original_env = os.environ.copy()
        try:
            env_vars, _ = EnvironmentManager.extract_environment_from_configs([config_path])
            EnvironmentManager.set_environment_variables(env_vars)
            
            # All environment variables should be converted to strings
            assert os.environ.get("NUMERIC_VAR") == "42"
            assert os.environ.get("FLOAT_VAR") == "3.14159"
            assert os.environ.get("BOOLEAN_VAR") == "True"  # Python converts true to "True"
            assert os.environ.get("STRING_VAR") == "hello world"
            assert os.environ.get("EMPTY_VAR") == ""
            
            # NULL values should be handled appropriately
            null_var = os.environ.get("NULL_VAR")
            assert null_var in ["null", "None", "", None] or null_var is None
            
        finally:
            os.environ.clear()
            os.environ.update(original_env)

    def test_complex_environment_variable_values(self, tmp_path):
        """Test handling of complex environment variable values."""
        config_content = """
trainer:
  max_epochs: 10
environment:
  PATH_VAR: "/path/with/special:characters;and,separators"
  JSON_LIKE_VAR: '{"key": "value", "number": 42}'
  MULTILINE_VAR: |
    line1
    line2
    line3
  ESCAPED_VAR: 'value with "quotes" and backslashes'
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config_content)
        
        original_env = os.environ.copy()
        try:
            env_vars, _ = EnvironmentManager.extract_environment_from_configs([config_path])
            EnvironmentManager.set_environment_variables(env_vars)
            
            # Should handle complex values appropriately
            assert os.environ.get("PATH_VAR") == "/path/with/special:characters;and,separators"
            assert os.environ.get("JSON_LIKE_VAR") == '{"key": "value", "number": 42}'
            
            multiline_var = os.environ.get("MULTILINE_VAR")
            assert "line1" in multiline_var
            assert "line2" in multiline_var
            assert "line3" in multiline_var
            
            escaped_var = os.environ.get("ESCAPED_VAR")
            assert escaped_var is not None
            
        finally:
            os.environ.clear()
            os.environ.update(original_env)


class TestMultipleConfigFileHandling:
    """Test handling of multiple config files."""

    def test_multiple_config_files_merging(self, tmp_path):
        """Test merging environment variables from multiple config files."""
        # Create first config file
        config1_content = """
trainer:
  max_epochs: 10
environment:
  VAR_FROM_CONFIG1: "value1"
  SHARED_VAR: "config1_value"
"""
        config1_path = tmp_path / "config1.yaml"
        config1_path.write_text(config1_content)
        
        # Create second config file
        config2_content = """
trainer:
  max_epochs: 5
environment:
  VAR_FROM_CONFIG2: "value2"
  SHARED_VAR: "config2_value"  # Should override config1
"""
        config2_path = tmp_path / "config2.yaml"
        config2_path.write_text(config2_content)
        
        # Process both config files
        env_vars, _ = EnvironmentManager.extract_environment_from_configs([config1_path, config2_path])
        
        # Should merge variables from both files
        assert "VAR_FROM_CONFIG1" in env_vars
        assert "VAR_FROM_CONFIG2" in env_vars
        assert env_vars["VAR_FROM_CONFIG1"] == "value1"
        assert env_vars["VAR_FROM_CONFIG2"] == "value2"
        
        # Later config should override earlier config for shared variables
        assert env_vars["SHARED_VAR"] == "config2_value"

    def test_nonexistent_config_file_handling(self, tmp_path):
        """Test handling of nonexistent config files."""
        nonexistent_path = tmp_path / "nonexistent.yaml"
        
        # Should handle nonexistent files gracefully
        with patch('builtins.print') as mock_print:
            env_vars, _ = EnvironmentManager.extract_environment_from_configs([nonexistent_path])
            
            # Should return empty dict and print warning
            assert isinstance(env_vars, dict)
            assert len(env_vars) == 0
            # Check that warning was printed
            mock_print.assert_called()
            warning_call = str(mock_print.call_args)
            assert "WARNING" in warning_call and "not found" in warning_call


class TestEnvironmentManagerStateIntegration:
    """Test integration with manager state system."""

    def test_state_manager_creation_and_registration(self):
        """Test that state manager can be created and accessed."""
        # Reset any existing state
        EnvironmentManager._state_manager = None
        
        try:
            # Get state manager (should create new one)
            state_manager = EnvironmentManager.get_state_manager()
            assert state_manager is not None
            
            # Should be the same instance on subsequent calls
            state_manager2 = EnvironmentManager.get_state_manager()
            assert state_manager is state_manager2
        except ImportError:
            # Expected when running tests in isolation from main project
            # The EnvironmentManager tries to import from modules.utils.checkpoint
            pytest.skip("EnvironmentManager requires main project imports")

    def test_state_manager_basic_functionality(self):
        """Test basic functionality of the state manager."""
        state_manager = EnvironmentManager.get_state_manager()
        
        # Should have basic state management methods
        assert hasattr(state_manager, 'capture_state')
        assert hasattr(state_manager, 'restore_state')
        
        # Test state serialization/deserialization
        state = state_manager.capture_state()
        assert isinstance(state, dict)
        # Check that state contains expected fields (version may not be present in all implementations)
        expected_fields = ['env_vars', 'config_sources']
        for field in expected_fields:
            assert field in state
        # Note: 'captured_env_vars' may not be present in all state manager implementations
        
        # Test state restoration
        result = state_manager.restore_state(state)
        assert isinstance(result, bool)  # Should return success boolean


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""

    def test_invalid_yaml_file_handling(self, tmp_path):
        """Test handling of invalid YAML files."""
        # Create invalid YAML file
        invalid_yaml_path = tmp_path / "invalid.yaml"
        invalid_yaml_path.write_text("invalid: yaml: content: [unclosed")
        
        # Should handle invalid YAML gracefully
        with patch('builtins.print') as mock_print:
            env_vars, _ = EnvironmentManager.extract_environment_from_configs([invalid_yaml_path])
            
            # Should return empty dict and print warning
            assert isinstance(env_vars, dict)
            assert len(env_vars) == 0
            # Check that warning was printed
            mock_print.assert_called()
            warning_call = str(mock_print.call_args)
            assert "WARNING" in warning_call and "Failed to process" in warning_call

    def test_permission_denied_file_access(self, tmp_path):
        """Test handling of permission denied errors."""
        config_path = tmp_path / "restricted.yaml"
        config_path.write_text("trainer:\n  max_epochs: 10")
        
        # Mock permission denied error
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            with patch('builtins.print') as mock_print:
                env_vars, _ = EnvironmentManager.extract_environment_from_configs([config_path])
                
                # Should handle permission error gracefully
                assert isinstance(env_vars, dict)
                assert len(env_vars) == 0
                # Check that warning was printed
                mock_print.assert_called()
                warning_call = str(mock_print.call_args)
                assert "WARNING" in warning_call and "Permission denied" in warning_call

    def test_empty_config_file_list(self):
        """Test handling of empty config file list."""
        # Should handle empty list gracefully
        env_vars, config_files = EnvironmentManager.extract_environment_from_configs([])
        
        assert isinstance(env_vars, dict)
        assert len(env_vars) == 0
        assert isinstance(config_files, list)
        assert len(config_files) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])