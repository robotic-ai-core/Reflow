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
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..checkpoint.manager_state import EnvironmentManagerState

logger = logging.getLogger(__name__)


class EnvironmentManager:
    """Manages environment variables from config files with state persistence support."""
    
    # Global state instance for checkpoint persistence
    _state_manager: Optional['EnvironmentManagerState'] = None
    
    @classmethod
    def get_state_manager(cls) -> 'EnvironmentManagerState':
        """Get or create the global state manager instance."""
        if cls._state_manager is None:
            # Use LightningReflow's own manager state system (preferred for portability)
            from ..checkpoint.manager_state import EnvironmentManagerState
            cls._state_manager = EnvironmentManagerState()
        return cls._state_manager
    
    @classmethod
    def register_for_checkpoint_persistence(cls) -> None:
        """Register this manager for checkpoint state persistence."""
        # Use LightningReflow's own manager state system (preferred for portability)
        from ..checkpoint.manager_state import register_manager
        state_manager = cls.get_state_manager()
        register_manager(state_manager)
        logger.info("🔗 EnvironmentManager registered with LightningReflow manager state system")
    
    @staticmethod
    def extract_environment_from_configs(config_paths: List[Path]) -> Tuple[Dict[str, str], List[str]]:
        """
        Extract environment variables from config files.
        
        Args:
            config_paths: List of config file paths to process
            
        Returns:
            Tuple of (environment variables dict, list of processed config files)
        """
        environment_vars = {}
        processed_files = []
        
        for config_path in config_paths:
            if not config_path.exists():
                print(f"[WARNING] Config file not found: {config_path}")
                continue
                
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Track that we processed this file
                processed_files.append(str(config_path))
                
                # Extract from top-level environment section
                if config and 'environment' in config:
                    environment_vars.update(config['environment'])
                
                # Extract from EnvironmentCallback configurations
                if config and 'trainer' in config and 'callbacks' in config['trainer']:
                    callbacks = config['trainer']['callbacks']
                    
                    for callback in callbacks:
                        if isinstance(callback, dict):
                            class_path = callback.get('class_path', '')
                            if 'EnvironmentCallback' in class_path:
                                # Extract environment variables from this callback
                                init_args = callback.get('init_args', {})
                                if 'env_vars' in init_args:
                                    environment_vars.update(init_args['env_vars'])
                    
            except Exception as e:
                print(f"[WARNING] Failed to process config file {config_path}: {e}")
                
        return environment_vars, processed_files
    
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