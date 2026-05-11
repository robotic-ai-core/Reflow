"""Shared utilities for checkpoint handling in lightning_reflow."""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, List
import torch
import lightning.pytorch as pl

logger = logging.getLogger(__name__)


def create_checkpoint_metadata(
    trainer: "pl.Trainer",
    reason: str = "manual",
    include_system_info: bool = False,
    extra: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create standardized checkpoint metadata.

    This is a helper function that creates common metadata fields used across
    checkpoint-saving operations. It consolidates the trainer state extraction
    and optional system information collection.

    Args:
        trainer: PyTorch Lightning trainer
        reason: Reason for checkpoint creation (e.g., "pause", "exception", "manual")
        include_system_info: Whether to include system info (torch version, cuda, etc.)
        extra: Additional metadata to merge into the result

    Returns:
        Dictionary with standardized checkpoint metadata
    """
    metadata = {
        'timestamp': time.time(),
        'save_reason': reason,
        'global_step': trainer.global_step,
        'current_epoch': trainer.current_epoch,
        'max_epochs': trainer.max_epochs,
        'max_steps': trainer.max_steps,
    }

    if include_system_info:
        metadata.update({
            'current_working_directory': os.getcwd(),
            'python_executable': sys.executable,
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        })

    if extra:
        metadata.update(extra)

    return metadata


def validate_checkpoint_structure(checkpoint: Dict[str, Any], checkpoint_path: str) -> Dict[str, Any]:
    """
    Comprehensive checkpoint validation with metadata extraction.
    
    Args:
        checkpoint: The loaded checkpoint dictionary
        checkpoint_path: Path to the checkpoint file for error reporting
        
    Returns:
        Dictionary containing extracted metadata
        
    Raises:
        ValueError: If checkpoint structure is invalid
    """
    try:
        # Basic structure validation
        required_keys = ['state_dict', 'epoch', 'global_step']
        missing_keys = [key for key in required_keys if key not in checkpoint]
        if missing_keys:
            raise ValueError(f"Checkpoint missing required keys: {missing_keys}")
        
        # Extract basic metadata
        metadata = {
            'epoch': checkpoint.get('epoch', 'unknown'),
            'global_step': checkpoint.get('global_step', 'unknown'),
            'pytorch_lightning_version': checkpoint.get('pytorch-lightning_version', 'unknown'),
            'enhanced_features': []
        }
        
        logger.info(
            "Checkpoint validation successful: epoch=%s, global_step=%s, pl_version=%s",
            metadata['epoch'], metadata['global_step'], metadata['pytorch_lightning_version'],
        )
        
        # Surface modern PauseCallback metadata if present.
        enhanced_features: list = []
        pause_metadata = checkpoint.get('pause_callback_metadata')
        if isinstance(pause_metadata, dict):
            if pause_metadata.get('wandb_run_id'):
                metadata['wandb_run_id'] = pause_metadata['wandb_run_id']
                enhanced_features.append(f"W&B run ID: {pause_metadata['wandb_run_id']}")
            if pause_metadata.get('pause_timestamp'):
                metadata['checkpoint_type'] = 'pause'
                enhanced_features.append("Pause checkpoint")
        metadata['enhanced_features'] = enhanced_features
        
        if enhanced_features:
            logger.info("Enhanced checkpoint features: %s", ', '.join(enhanced_features))
            
        return metadata
        
    except Exception as e:
        raise ValueError(f"Failed to validate checkpoint structure for '{checkpoint_path}': {e}")


def extract_wandb_run_id(checkpoint: Dict[str, Any]) -> Optional[str]:
    """Return the W&B run id from checkpoint metadata, or None.

    Checks the two locations the current library writes:
    ``self_contained_metadata`` (ConfigEmbeddingMixin) and
    ``pause_callback_metadata`` (PauseCallback). Top-level ``wandb_run_id``
    is deliberately not consulted — only trusted metadata blobs.
    """
    for key in ("self_contained_metadata", "pause_callback_metadata"):
        metadata = checkpoint.get(key)
        if isinstance(metadata, dict):
            run_id = metadata.get("wandb_run_id")
            if isinstance(run_id, str) and run_id.strip():
                return run_id.strip()
    return None


def extract_embedded_config(checkpoint_path: str) -> Optional[str]:
    """
    Extract embedded configuration YAML from a checkpoint.
    
    Searches all known metadata locations for embedded config content.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        YAML configuration string if found, None otherwise
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        logger.debug("Checkpoint keys: %s", list(checkpoint.keys())[:10])

        # The two metadata blobs the current library writes.
        for location in ("self_contained_metadata", "pause_callback_metadata"):
            metadata = checkpoint.get(location)
            if isinstance(metadata, dict):
                config_content = metadata.get('embedded_config_content')
                if config_content:
                    logger.info("Found embedded config in %s", location)
                    return config_content

        # Lightning's own saved config dict, if a ConfigCallback wrote one.
        if 'lightning_config' in checkpoint:
            logger.info("Found lightning_config (clean checkpoint format)")
            import yaml
            return yaml.dump(checkpoint['lightning_config'])

        return None

    except Exception as e:
        logger.warning("Failed to extract embedded config from %s: %s", checkpoint_path, e)
        return None


