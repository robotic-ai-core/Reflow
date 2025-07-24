"""
Config Synthesis Module for DiffusionFlow

This module provides utilities for working with Lightning's auto-generated
configuration files for checkpoint embedding and resume functionality.
"""

import yaml
import warnings
from typing import Optional, Dict, Any

from lightning.pytorch import Trainer
from lightning_reflow.utils.logging.logging_config import get_logger
from pathlib import Path


def synthesize_config(trainer: Trainer, cli: "LightningCLI", verbose: bool = True, cli_config: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Synthesizes the full configuration from a LightningCLI instance.
    This is a convenience wrapper around the ConfigSynthesizer class.
    """
    synthesizer = ConfigSynthesizer(verbose=verbose)
    if cli_config:
        return yaml.dump(cli_config)
    return synthesizer.capture_resolved_config(trainer)


def get_config_raw(config_path: Path) -> Optional[str]:
    """
    Reads a config file and returns the raw YAML string.
    """
    if config_path.exists():
        with open(config_path, 'r') as f:
            return f.read()
    return None


class ConfigSynthesizer:
    """
    Configuration management for checkpoints using Lightning's auto-generated config.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.logger = get_logger(__name__)
    
    def capture_resolved_config(self, trainer: Trainer) -> Optional[str]:
        """
        Capture the fully resolved configuration using Lightning's auto-generated config file.
        """
        try:
            if self.verbose:
                print("🔍 [CONFIG] Reading Lightning's auto-generated config.yaml...")
            
            config_path = Path("config.yaml")
            
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_yaml = f.read()
                
                if self.verbose:
                    self.logger.info(f"✅ Successfully read Lightning config from {config_path}")
                return config_yaml
            else:
                if self.verbose:
                    self.logger.error("❌ Lightning config.yaml not found")
                return None
            
        except Exception as e:
            if self.verbose:
                self.logger.error(f"❌ Exception during config capture: {e}")
            return None