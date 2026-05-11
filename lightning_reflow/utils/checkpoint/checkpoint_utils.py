"""
Shared utilities for comprehensive checkpoint handling across the DiffusionFlow codebase.

This module consolidates the sophisticated checkpoint logic from the pause/exit system
for consistent use across all checkpoint operations in the codebase.
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, List
import torch
import lightning.pytorch as pl


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
        
        print(f"[INFO] Checkpoint validation successful:")
        print(f"       Epoch: {metadata['epoch']}, Global step: {metadata['global_step']}")
        print(f"       PyTorch Lightning version: {metadata['pytorch_lightning_version']}")
        
        # Check for enhanced checkpoint metadata from different systems
        enhanced_features = []
        
        # Check for pause/exit system metadata
        if 'pause_exit_callback_state' in checkpoint:
            pause_state = checkpoint['pause_exit_callback_state']
            if 'wandb_run_id' in pause_state:
                enhanced_features.append(f"W&B run ID: {pause_state['wandb_run_id']}")
                metadata['wandb_run_id'] = pause_state['wandb_run_id']
            if 'pause_timestamp' in pause_state:
                enhanced_features.append("Pause/Exit checkpoint")
                metadata['checkpoint_type'] = 'pause_exit'
        
        # Check for WandbArtifactCheckpoint metadata
        if 'wandb_artifact_checkpoint_state' in checkpoint:
            artifact_state = checkpoint['wandb_artifact_checkpoint_state']
            if 'emergency_reason' in artifact_state:
                enhanced_features.append(f"Emergency checkpoint ({artifact_state['emergency_reason']})")
                metadata['checkpoint_type'] = 'emergency'
                metadata['emergency_reason'] = artifact_state['emergency_reason']
            if 'wandb_run_id' in artifact_state:
                enhanced_features.append(f"W&B run ID: {artifact_state['wandb_run_id']}")
                metadata['wandb_run_id'] = artifact_state['wandb_run_id']
        
        # Check for comprehensive metadata
        if 'diffusion_flow_checkpoint_metadata' in checkpoint:
            df_metadata = checkpoint['diffusion_flow_checkpoint_metadata']
            if 'save_reason' in df_metadata:
                enhanced_features.append(f"DiffusionFlow checkpoint ({df_metadata['save_reason']})")
                metadata['checkpoint_type'] = 'comprehensive'
                metadata['save_reason'] = df_metadata['save_reason']
        
        metadata['enhanced_features'] = enhanced_features
        
        if enhanced_features:
            print(f"[INFO] Enhanced checkpoint features: {', '.join(enhanced_features)}")
            
        return metadata
        
    except Exception as e:
        raise ValueError(f"Failed to validate checkpoint structure for '{checkpoint_path}': {e}")


def extract_wandb_run_id(checkpoint: Dict[str, Any]) -> Optional[str]:
    """
    Extract W&B run ID from checkpoint metadata.
    
    Searches all known metadata locations for W&B run ID.
    
    Args:
        checkpoint: The loaded checkpoint dictionary
        
    Returns:
        W&B run ID if found, None otherwise
    """
    # NOTE: We do NOT check root level 'wandb_run_id' for security reasons.
    # Only extract from trusted metadata locations to prevent tampering.
    
    # Check self_contained_metadata (modern format)
    if 'self_contained_metadata' in checkpoint:
        metadata = checkpoint['self_contained_metadata']
        if 'wandb_run_id' in metadata:
            run_id = metadata['wandb_run_id']
            if isinstance(run_id, str) and run_id.strip():
                return run_id.strip()
    
    # Check current PauseCallback metadata format
    if 'pause_callback_metadata' in checkpoint:
        pause_metadata = checkpoint['pause_callback_metadata']
        if 'wandb_run_id' in pause_metadata:
            run_id = pause_metadata['wandb_run_id']
            # Clean whitespace and validate
            if isinstance(run_id, str) and run_id.strip():
                return run_id.strip()
    
    # Check validation boundary pause metadata
    if 'validation_boundary_pause_metadata' in checkpoint:
        vb_pause_state = checkpoint['validation_boundary_pause_metadata']
        if 'wandb_run_id' in vb_pause_state:
            run_id = vb_pause_state['wandb_run_id']
            if isinstance(run_id, str) and run_id.strip():
                return run_id.strip()
    
    # Check pause/exit system metadata (top-level - legacy)
    if 'pause_exit_callback_state' in checkpoint:
        pause_state = checkpoint['pause_exit_callback_state']
        if 'wandb_run_id' in pause_state:
            run_id = pause_state['wandb_run_id']
            if isinstance(run_id, str) and run_id.strip():
                return run_id.strip()
    
    # Check WandbArtifactCheckpoint metadata
    if 'wandb_artifact_checkpoint_state' in checkpoint:
        artifact_state = checkpoint['wandb_artifact_checkpoint_state']
        if 'wandb_run_id' in artifact_state:
            run_id = artifact_state['wandb_run_id']
            if isinstance(run_id, str) and run_id.strip():
                return run_id.strip()
    
    # Check comprehensive metadata
    if 'diffusion_flow_checkpoint_metadata' in checkpoint:
        df_metadata = checkpoint['diffusion_flow_checkpoint_metadata']
        if 'wandb_run_id' in df_metadata:
            run_id = df_metadata['wandb_run_id']
            if isinstance(run_id, str) and run_id.strip():
                return run_id.strip()
        
        # Check pause/exit system metadata nested in comprehensive metadata
        if 'pause_exit_callback_state' in df_metadata:
            pause_state = df_metadata['pause_exit_callback_state']
            if 'wandb_run_id' in pause_state:
                run_id = pause_state['wandb_run_id']
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
        print(f"[DEBUG] Checkpoint keys: {list(checkpoint.keys())[:10]}")  # Show first 10 keys
        
        # Check all known metadata locations
        metadata_locations = [
            'self_contained_metadata',  # ConfigEmbeddingMixin standard location
            'pause_callback_metadata',  # PauseCallback metadata
            'wandb_artifact_checkpoint_metadata',  # WandbArtifactCheckpoint metadata
            'checkpoint_metadata',  # Generic metadata
            'model_checkpoint_metadata',  # ModelCheckpoint metadata
        ]
        
        for location in metadata_locations:
            if location in checkpoint:
                metadata = checkpoint[location]
                if isinstance(metadata, dict) and 'embedded_config_content' in metadata:
                    config_content = metadata['embedded_config_content']
                    if config_content:
                        print(f"[INFO] Found embedded config in {location}")
                        return config_content
        
        # Check top-level for legacy format
        if 'embedded_config_content' in checkpoint:
            print("[INFO] Found embedded config at top level (legacy format)")
            return checkpoint['embedded_config_content']
        
        # Check for clean checkpoint format (lightning_config)
        if 'lightning_config' in checkpoint:
            print("[INFO] Found lightning_config (clean checkpoint format)")
            import yaml
            # Convert the config dict to YAML string for consistency
            return yaml.dump(checkpoint['lightning_config'])
        
        return None
        
    except Exception as e:
        print(f"[WARNING] Failed to extract embedded config from {checkpoint_path}: {e}")
        return None


