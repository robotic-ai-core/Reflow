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


def save_comprehensive_checkpoint(
    trainer: "pl.Trainer", 
    pl_module: "pl.LightningModule", 
    checkpoint_path: str,
    reason: str = "manual",
    extra_metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save comprehensive checkpoint with enhanced state and metadata.
    
    This function provides a standardized way to save checkpoints with enhanced
    metadata across all DiffusionFlow components.
    
    Args:
        trainer: PyTorch Lightning trainer
        pl_module: PyTorch Lightning module
        checkpoint_path: Path where to save the checkpoint
        reason: Reason for checkpoint creation (e.g., "pause", "exception", "manual")
        extra_metadata: Additional metadata to include in the checkpoint
    """
    # Use trainer's save_checkpoint to get standard Lightning state
    trainer.save_checkpoint(checkpoint_path)
    
    # Load the checkpoint to add our additional state
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Add comprehensive state information
    checkpoint_metadata = {
        'save_timestamp': time.time(),
        'save_reason': reason,
        'current_working_directory': os.getcwd(),
        'python_executable': sys.executable,
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'trainer_state': {
            'current_epoch': trainer.current_epoch,
            'global_step': trainer.global_step,
            'max_epochs': trainer.max_epochs,
            'max_steps': trainer.max_steps,
        }
    }
    
    # Add extra metadata if provided
    if extra_metadata:
        checkpoint_metadata.update(extra_metadata)
    
    # Store under a standard key
    checkpoint['diffusion_flow_checkpoint_metadata'] = checkpoint_metadata
    
    # Note: fix_epoch_progress_state was removed as it's no longer needed with modern PyTorch Lightning
    # Modern Lightning handles epoch progress correctly during checkpoint resumption
    
    # Save enhanced checkpoint
    torch.save(checkpoint, checkpoint_path)




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
    # Check root level first (simplest format, used in some tests)
    if 'wandb_run_id' in checkpoint:
        run_id = checkpoint['wandb_run_id']
        if isinstance(run_id, str) and run_id.strip():
            return run_id.strip()
    
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


def load_and_validate_checkpoint(checkpoint_path: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load and validate a checkpoint with comprehensive error handling.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        Tuple of (checkpoint_dict, metadata_dict)
        
    Raises:
        ValueError: If checkpoint is invalid
        FileNotFoundError: If checkpoint file doesn't exist
    """
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        metadata = validate_checkpoint_structure(checkpoint, checkpoint_path)
        return checkpoint, metadata
    except Exception as e:
        raise ValueError(f"Failed to load checkpoint from {checkpoint_path}: {e}")


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


def standardize_checkpoint_directory_structure(
    base_dir: str = "checkpoints",
    create_subdirs: bool = True
) -> Dict[str, Path]:
    """
    Create and return standardized checkpoint directory structure.
    
    Args:
        base_dir: Base directory for all checkpoints
        create_subdirs: Whether to create subdirectories
        
    Returns:
        Dictionary mapping checkpoint types to their directories
    """
    base_path = Path(base_dir)
    
    structure = {
        'lightning': base_path / 'lightning',      # Standard Lightning checkpoints
        'pause': base_path / 'pause',              # Pause/exit checkpoints
        'emergency': base_path / 'emergency',      # Emergency checkpoints
        'manual': base_path / 'manual',            # Manual checkpoints
        'wandb_downloads': base_path / 'wandb_downloads',  # Downloaded W&B checkpoints
    }
    
    if create_subdirs:
        for checkpoint_type, directory in structure.items():
            directory.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] Created checkpoint directory: {directory}")
    
    return structure