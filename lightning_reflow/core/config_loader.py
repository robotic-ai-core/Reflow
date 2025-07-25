"""
Configuration loader with early environment variable setting.

This module provides the ConfigLoader class which acts as a facade for the complex
configuration loading process, ensuring environment variables are set at the right time.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import yaml
from omegaconf import OmegaConf, DictConfig

from ..utils.logging.environment_manager import EnvironmentManager

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Facade for loading and merging configurations from multiple sources.
    
    This class handles the complex process of:
    1. Loading configuration from YAML files
    2. Applying programmatic overrides
    3. Setting environment variables EARLY in the process
    4. Merging everything into a final OmegaConf object
    """
    
    def __init__(self):
        self.config: Optional[DictConfig] = None
        self._env_vars_applied = False
    
    def load_config(
        self, 
        config_files: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        apply_env_vars: bool = True
    ) -> DictConfig:
        """
        Load configuration from files and apply overrides.
        
        Args:
            config_files: Path(s) to YAML config file(s)
            config_overrides: Dictionary of config overrides (e.g., from Ray Tune)
            apply_env_vars: Whether to apply environment variables from config
            
        Returns:
            Merged configuration as OmegaConf DictConfig
        """
        logger.info("Loading configuration...")
        
        # Step 1: Load base configuration from files
        base_config = self._load_config_files(config_files)
        
        # Step 2: Apply programmatic overrides
        if config_overrides:
            logger.info(f"Applying {len(config_overrides)} configuration overrides")
            merged_config = self._apply_overrides(base_config, config_overrides)
        else:
            merged_config = base_config
        
        # Step 3: ⚠️ CRITICAL: Set environment variables EARLY ⚠️
        if apply_env_vars and not self._env_vars_applied:
            self._apply_environment_variables(merged_config)
            self._env_vars_applied = True
        
        self.config = merged_config
        return merged_config
    
    def _load_config_files(
        self, 
        config_files: Optional[Union[str, Path, List[Union[str, Path]]]]
    ) -> DictConfig:
        """Load configuration from YAML files."""
        if not config_files:
            logger.info("No config files specified, using empty base config")
            return OmegaConf.create({})
        
        if isinstance(config_files, (str, Path)):
            config_files = [config_files]
        
        merged_config = OmegaConf.create({})
        
        for config_file in config_files:
            config_path = Path(config_file)
            if not config_path.exists():
                logger.warning(f"Config file not found: {config_path}")
                continue
                
            logger.info(f"Loading config file: {config_path}")
            
            try:
                with open(config_path, 'r') as f:
                    file_config = OmegaConf.create(yaml.safe_load(f))
                merged_config = OmegaConf.merge(merged_config, file_config)
            except Exception as e:
                logger.error(f"Failed to load config file {config_path}: {e}")
                raise
        
        return merged_config
    
    def _apply_overrides(
        self, 
        base_config: DictConfig, 
        overrides: Dict[str, Any]
    ) -> DictConfig:
        """Apply programmatic overrides to the base configuration."""
        merged_config = OmegaConf.create(base_config)
        
        # Convert overrides to nested dict structure
        override_dict = {}
        for key, value in overrides.items():
            if '.' in key:
                # Split nested key and create nested dict
                keys = key.split('.')
                current = override_dict
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = value
            else:
                override_dict[key] = value
        
        # Apply overrides using OmegaConf merge
        try:
            OmegaConf.set_struct(merged_config, False)  # Allow new keys
            override_config = OmegaConf.create(override_dict)
            merged_config = OmegaConf.merge(merged_config, override_config)
            
            logger.debug(f"Applied {len(overrides)} overrides successfully")
        except Exception as e:
            logger.error(f"Failed to apply overrides: {e}")
            raise
        
        return merged_config
    
    def _apply_environment_variables(self, config: DictConfig) -> None:
        """
        Apply environment variables from configuration EARLY in the process.
        
        This is critical - environment variables must be set before PyTorch,
        CUDA, or other system libraries initialize.
        """
        logger.info("🔧 Applying environment variables (EARLY)")
        
        # Check if config has environment variable section
        env_config = OmegaConf.select(config, "environment")
        if not env_config:
            logger.debug("No 'environment' section found in config")
            return
        
        # Convert OmegaConf to regular dict for EnvironmentManager
        env_vars = OmegaConf.to_container(env_config, resolve=True)
        if not isinstance(env_vars, dict):
            logger.warning("Environment config is not a dictionary, skipping")
            return
        
        # Use EnvironmentManager to set variables with proper logging
        try:
            EnvironmentManager.set_environment_variables(
                env_vars, 
                config_sources=["config_loader"]
            )
            logger.info(f"✅ Applied {len(env_vars)} environment variables")
        except Exception as e:
            logger.error(f"Failed to apply environment variables: {e}")
            raise
    
    def get_section(self, section_path: str, default: Any = None) -> Any:
        """
        Get a specific section from the loaded configuration.
        
        Args:
            section_path: Dot-separated path (e.g., "model.learning_rate")
            default: Default value if section not found
            
        Returns:
            Configuration section value
        """
        if self.config is None:
            raise RuntimeError("No configuration loaded. Call load_config() first.")
        
        return OmegaConf.select(self.config, section_path, default=default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the loaded configuration to a regular Python dictionary."""
        if self.config is None:
            raise RuntimeError("No configuration loaded. Call load_config() first.")
        
        return OmegaConf.to_container(self.config, resolve=True) 