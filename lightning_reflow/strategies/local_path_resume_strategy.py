"""
Local path resume strategy.

This strategy handles resuming from checkpoint files stored on the local filesystem.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

from .resume_strategy import ResumeStrategy

logger = logging.getLogger(__name__)


class LocalPathResumeStrategy(ResumeStrategy):
    """
    Resume strategy for local filesystem paths.
    
    This strategy handles resuming from checkpoint files that are stored
    locally on the filesystem.
    """
    
    def validate_source(self, resume_source: str) -> bool:
        """
        Validate that the source is a local path.
        
        Args:
            resume_source: Path to validate
            
        Returns:
            True if it's a valid local path, False otherwise
        """
        try:
            # Reject obvious W&B artifact patterns
            if ':' in resume_source and '/' in resume_source:
                # Pattern like "entity/project/artifact:version"
                return False
            
            path = Path(resume_source)
            # Consider it a local path if it exists or looks like a filesystem path
            return path.exists() or path.suffix in ['.ckpt', '.pt', '.pth', '.pkl']
        except Exception:
            return False
    
    def prepare_resume(
        self, 
        resume_source: str, 
        **kwargs
    ) -> Tuple[Optional[Path], Optional[str]]:
        """
        Prepare for resuming from a local checkpoint path.
        
        Args:
            resume_source: Local filesystem path to the checkpoint
            **kwargs: Additional arguments (ignored for local paths)
            
        Returns:
            Tuple of (checkpoint_path, embedded_config_yaml)
            
        Raises:
            FileNotFoundError: If the checkpoint file doesn't exist
        """
        logger.info(f"Preparing to resume from local path: {resume_source}")
        
        checkpoint_path = Path(resume_source)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {resume_source}")
        
        if not checkpoint_path.is_file():
            raise ValueError(f"Checkpoint path is not a file: {resume_source}")
        
        # Extract embedded config from checkpoint as raw YAML string
        from lightning_reflow.utils.checkpoint.checkpoint_utils import extract_embedded_config
        embedded_config_yaml = extract_embedded_config(str(checkpoint_path))
        
        if embedded_config_yaml:
            logger.info(f"📄 Found embedded configuration in checkpoint ({len(embedded_config_yaml)} chars)")
        else:
            logger.info("📄 No embedded configuration found in checkpoint")
        
        logger.info(f"✅ Local checkpoint prepared for resumption: {checkpoint_path}")
        
        return checkpoint_path, embedded_config_yaml
    
    def _is_valid_checkpoint(self, checkpoint_path: Path) -> bool:
        """
        Perform basic validation that the file is a checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            True if the file appears to be a valid checkpoint
        """
        try:
            # Check file extension
            valid_extensions = {'.ckpt', '.pt', '.pth', '.pkl', '.pickle'}
            if checkpoint_path.suffix.lower() not in valid_extensions:
                logger.warning(f"Checkpoint has unusual extension: {checkpoint_path.suffix}")
            
            # Check file size (should be non-empty)
            if checkpoint_path.stat().st_size == 0:
                logger.error("Checkpoint file is empty")
                return False
            
            # Try to peek at the file to see if it's a valid PyTorch file
            # We don't load the full checkpoint here to avoid memory issues
            import torch
            try:
                # Just check if torch can read the file header
                torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                return True
            except Exception as e:
                logger.error(f"Checkpoint file appears corrupted: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Error validating checkpoint: {e}")
            return False
    
    def cleanup(self) -> None:
        """
        Clean up resources (no-op for local paths).
        
        Local paths don't require cleanup since we don't create temporary files.
        """
        pass 