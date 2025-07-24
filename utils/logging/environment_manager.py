"""
Environment Variable Manager for DiffusionFlow

This module provides shared functionality for parsing and setting environment
variables from config files. It's used by both fresh training (fit) and 
resume commands to ensure consistent environment setup.

Enhanced with manager state persistence for complete environment restoration
during resume operations.
"""

import os
import sys
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class EnvironmentManager:
    """Manages environment variables from config files with state persistence support."""
    
    # Global state instance for checkpoint persistence
    _state_manager: Optional['EnvironmentManagerState'] = None
    
    @classmethod
    def get_state_manager(cls) -> 'EnvironmentManagerState':
        """Get or create the global state manager instance."""
        if cls._state_manager is None:
            from ..checkpoint.manager_state import EnvironmentManagerState
            cls._state_manager = EnvironmentManagerState()
        return cls._state_manager
    
    @classmethod
    def register_for_checkpoint_persistence(cls) -> None:
        """Register this manager for checkpoint state persistence."""
        from ..checkpoint.manager_state import register_manager
        state_manager = cls.get_state_manager()
        register_manager(state_manager)
        logger.info("🔗 EnvironmentManager registered for checkpoint persistence")
    
    @staticmethod
    def extract_environment_from_configs(config_paths: List[Path]) -> Tuple[Dict[str, str], List[Tuple[Path, Path]]]:
        """
        Extract environment variables from config files and create cleaned versions.
        
        Args:
            config_paths: List of config file paths to process
            
        Returns:
            Tuple of:
            - Dictionary of environment variables to set
            - List of (original_path, cleaned_path) tuples for cleanup
        """
        environment_vars = {}
        modified_files = []
        
        for config_path in config_paths:
            if not config_path.exists():
                print(f"[WARNING] Config file not found: {config_path}")
                continue
                
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    
                if config and 'environment' in config:
                    # Extract and merge environment variables
                    environment_vars.update(config['environment'])
                    
                    # Create cleaned config without environment section
                    cleaned_config = {k: v for k, v in config.items() if k != 'environment'}
                    
                    # Create temporary cleaned config file
                    temp_config_path = config_path.with_suffix('.tmp' + config_path.suffix)
                    with open(temp_config_path, 'w') as f:
                        yaml.dump(cleaned_config, f, default_flow_style=False, sort_keys=False)
                    
                    modified_files.append((config_path, temp_config_path))
                    
            except Exception as e:
                print(f"[WARNING] Failed to process config file {config_path}: {e}")
                
        return environment_vars, modified_files
    
    @classmethod
    def set_environment_variables(cls, env_vars: Dict[str, str], config_sources: List[str] = None) -> None:
        """
        Set environment variables with conflict checking, logging, and state tracking.
        
        Args:
            env_vars: Dictionary of environment variable names to values
            config_sources: List of config file paths that provided these variables
        """
        if not env_vars:
            return
        
        print("[INFO] Setting environment variables from config:")
        
        # Get state manager for persistence
        state_manager = cls.get_state_manager()
        
        for var_name, value in env_vars.items():
            value_str = str(value)
            
            # Check for conflicts with existing environment variables
            existing = os.environ.get(var_name)
            if existing and existing != value_str:
                print(f"[WARNING] Overriding environment variable {var_name}")
                print(f"  Existing: {existing}")
                print(f"  New:      {value_str}")
            
            print(f"[INFO] Set {var_name}={value_str}")
        
        # Set variables and track in state manager
        config_source_paths = [str(p) for p in (config_sources or [])]
        state_manager.set_environment_variables(env_vars, config_source_paths)
        
        print(f"[INFO] Successfully configured {len(env_vars)} environment variables")
        print(f"[INFO] Environment state tracked for checkpoint persistence")
    
    @staticmethod
    def cleanup_temp_files(modified_files: List[Tuple[Path, Path]]) -> None:
        """
        Clean up temporary config files.
        
        Args:
            modified_files: List of (original_path, temp_path) tuples
        """
        for original_file, temp_file in modified_files:
            try:
                if Path(temp_file).exists():
                    Path(temp_file).unlink()
            except Exception as e:
                print(f"[WARNING] Failed to cleanup temp config file {temp_file}: {e}")
    
    @staticmethod
    def parse_sys_argv_configs() -> List[Path]:
        """
        Parse --config arguments from sys.argv.
        
        Returns:
            List of config file paths found in command line arguments
        """
        config_files = []
        
        i = 0
        while i < len(sys.argv):
            if sys.argv[i] == '--config' and i + 1 < len(sys.argv):
                config_files.append(Path(sys.argv[i + 1]))
                i += 2
            elif sys.argv[i].startswith('--config='):
                config_files.append(Path(sys.argv[i].split('=', 1)[1]))
                i += 1
            else:
                i += 1
                
        return config_files
    
    @staticmethod
    def update_sys_argv_with_cleaned_configs(modified_files: List[Tuple[Path, Path]]) -> None:
        """
        Update sys.argv to use cleaned config files.
        
        Args:
            modified_files: List of (original_path, temp_path) tuples
        """
        for original_path, temp_path in modified_files:
            for j, arg in enumerate(sys.argv):
                if arg == str(original_path):
                    sys.argv[j] = str(temp_path)
                elif arg == '--config' and j + 1 < len(sys.argv) and sys.argv[j + 1] == str(original_path):
                    sys.argv[j + 1] = str(temp_path)
                elif arg.startswith(f'--config={original_path}'):
                    sys.argv[j] = f'--config={temp_path}'
    
    @classmethod
    def restore_environment_from_checkpoint(cls, checkpoint_state: Dict[str, any]) -> bool:
        """
        Restore environment variables from checkpoint manager state.
        
        Args:
            checkpoint_state: Dictionary containing saved manager states
            
        Returns:
            True if environment was successfully restored, False otherwise
        """
        try:
            # Import here to avoid circular dependency
            from ..checkpoint.manager_state import restore_all_manager_states
            
            # Restore all manager states (including environment)
            results = restore_all_manager_states(checkpoint_state)
            
            # Check if environment manager was successfully restored
            env_success = results.get("environment_manager", False)
            
            if env_success:
                logger.info("✅ Environment variables restored from checkpoint")
            else:
                logger.warning("⚠️ Environment variables not restored from checkpoint")
            
            return env_success
            
        except Exception as e:
            logger.error(f"❌ Failed to restore environment from checkpoint: {e}")
            return False
    
    @classmethod
    def get_environment_summary(cls) -> Dict[str, any]:
        """
        Get a summary of currently managed environment variables.
        
        Returns:
            Dictionary with environment variable summary
        """
        state_manager = cls.get_state_manager()
        captured_vars = state_manager.get_captured_variables()
        
        return {
            "managed_variables": captured_vars,
            "variable_count": len(captured_vars),
            "state_available": len(captured_vars) > 0
        }