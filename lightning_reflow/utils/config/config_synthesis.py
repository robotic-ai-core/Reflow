"""
Config Synthesis Module for DiffusionFlow

This module provides utilities for working with Lightning's auto-generated
configuration files for checkpoint embedding and resume functionality.
"""

import yaml
import warnings
from typing import Optional, Dict, Any, TYPE_CHECKING

from lightning.pytorch import Trainer
from lightning_reflow.utils.logging.logging_config import get_logger
from pathlib import Path

if TYPE_CHECKING:
    from lightning.pytorch.cli import LightningCLI


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


def convert_config_dict_to_dataclasses(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a configuration dictionary to use proper dataclass instances.
    
    This is needed when loading from embedded YAML configs where everything
    is deserialized as dictionaries, but the model expects dataclass instances.
    
    Args:
        config_dict: Configuration dictionary with nested structure
        
    Returns:
        Modified configuration with dataclass instances where appropriate
    """
    # Import config classes lazily to avoid circular imports
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        from modules.utils.config.config import (
            GenerativeBackboneConfig, UNetConfig, OptimizerConfig, 
            LoggingConfig, FlowMatchingProcessSamplerConfig, 
            DiffusionProcessSamplerConfig, MultiModalEncoderConfig,
            ModalityEncoderConfig, EmbeddingAdaptorConfig, ODESolverConfig
        )
    except ImportError:
        logger = get_logger(__name__)
        logger.warning("Could not import config classes from modules.utils.config")
        return config_dict
    
    converted_config = config_dict.copy()
    
    # Convert backbone config
    if 'backbone' in converted_config and isinstance(converted_config['backbone'], dict):
        backbone_dict = converted_config['backbone']
        backbone_type = backbone_dict.get('type', 'unet')
        init_args = backbone_dict.get('init_args', {})
        
        # Convert init_args to proper config type
        if backbone_type == 'unet' and isinstance(init_args, dict):
            init_args = UNetConfig(**init_args)
        
        converted_config['backbone'] = GenerativeBackboneConfig(
            type=backbone_type,
            init_args=init_args
        )
    
    # Convert optimizer config
    if 'optimizer' in converted_config and isinstance(converted_config['optimizer'], dict):
        try:
            converted_config['optimizer'] = OptimizerConfig(**converted_config['optimizer'])
        except Exception as e:
            logger.warning(f"Failed to convert optimizer config: {e}")
    
    # Convert logging config
    if 'logging' in converted_config and isinstance(converted_config['logging'], dict):
        try:
            converted_config['logging'] = LoggingConfig(**converted_config['logging'])
        except Exception as e:
            logger.warning(f"Failed to convert logging config: {e}")
    
    # Convert backbone config
    if 'backbone' in converted_config and isinstance(converted_config['backbone'], dict):
        try:
            from modules.utils.config.config import GenerativeBackboneConfig, UNetConfig
            backbone_dict = converted_config['backbone']
            backbone_type = backbone_dict.get('type', 'unet')
            
            if backbone_type == 'unet':
                init_args = backbone_dict.get('init_args', {})
                if isinstance(init_args, dict):
                    unet_config = UNetConfig(**init_args)
                else:
                    unet_config = init_args
                
                converted_config['backbone'] = GenerativeBackboneConfig(
                    type=backbone_type,
                    init_args=unet_config
                )
            else:
                # For other backbone types, create with the existing structure
                converted_config['backbone'] = GenerativeBackboneConfig(**backbone_dict)
        except Exception as e:
            logger.warning(f"Failed to convert backbone config: {e}")

    # Convert process_sampler config with proper nested handling
    if 'process_sampler' in converted_config and isinstance(converted_config['process_sampler'], dict):
        try:
            sampler_dict = converted_config['process_sampler']
            sampler_type = sampler_dict.get('type', 'flow_matching')
            
            if sampler_type == 'flow_matching':
                # Handle nested ode_solver properly
                ode_solver_dict = sampler_dict.get('ode_solver', {})
                if isinstance(ode_solver_dict, dict):
                    ode_solver_config = ODESolverConfig(**ode_solver_dict)
                else:
                    ode_solver_config = ode_solver_dict
                
                converted_config['process_sampler'] = FlowMatchingProcessSamplerConfig(
                    type=sampler_dict.get('type', 'flow_matching'),
                    path_type=sampler_dict.get('path_type', 'linear'),
                    ode_solver=ode_solver_config
                )
            elif sampler_type == 'diffusion':
                converted_config['process_sampler'] = DiffusionProcessSamplerConfig(**sampler_dict)
        except Exception as e:
            logger.warning(f"Failed to convert process_sampler config: {e}")
    
    # Convert condition_encoder config
    if 'condition_encoder' in converted_config and isinstance(converted_config['condition_encoder'], dict):
        try:
            encoder_dict = converted_config['condition_encoder']
            encoder_type = encoder_dict.get('type')
            
            if encoder_type == 'multimodal':
                converted_config['condition_encoder'] = MultiModalEncoderConfig(**encoder_dict)
            elif encoder_type == 'pretrained_text':
                # Import TextEncoderConfig for text encoders
                try:
                    from modules.utils.config.config import TextEncoderConfig
                    converted_config['condition_encoder'] = TextEncoderConfig(**encoder_dict)
                except ImportError:
                    # Fallback - just use the dict
                    logger.warning("TextEncoderConfig not available, keeping as dict")
            elif encoder_type:
                # For single modality encoders
                converted_config['condition_encoder'] = ModalityEncoderConfig(**encoder_dict)
        except Exception as e:
            logger.warning(f"Failed to convert condition_encoder config: {e}")
    
    # Convert embedding_adaptor config  
    if 'embedding_adaptor' in converted_config and isinstance(converted_config['embedding_adaptor'], dict):
        try:
            converted_config['embedding_adaptor'] = EmbeddingAdaptorConfig(**converted_config['embedding_adaptor'])
        except Exception as e:
            logger.warning(f"Failed to convert embedding_adaptor config: {e}")
    
    return converted_config