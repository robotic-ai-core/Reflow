"""
Local path resume strategy.

This strategy handles resuming from checkpoint files stored on the local filesystem.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

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
            path = Path(resume_source)
            # Consider it a local path if it exists or if it's not a URL-like string
            return path.exists() or not ('://' in resume_source and not resume_source.startswith('file://'))
        except Exception:
            return False
    
    def prepare_resume(
        self, 
        resume_source: str, 
        **kwargs
    ) -> Tuple[Optional[Path], Optional[Dict[str, Any]]]:
        """
        Prepare for resuming from a local checkpoint path.
        
        Args:
            resume_source: Local filesystem path to the checkpoint
            **kwargs: Additional arguments (unused for local paths)
            
        Returns:
            Tuple of (checkpoint_path, None) - no additional config for local paths
            
        Raises:
            ValueError: If the path doesn't exist or isn't a file
            RuntimeError: If the checkpoint file is corrupted or unreadable
        """
        logger.info(f"Preparing to resume from local path: {resume_source}")
        
        # Handle file:// URLs
        if resume_source.startswith('file://'):
            resume_source = resume_source[7:]  # Remove 'file://' prefix
        
        checkpoint_path = Path(resume_source)
        
        # Validate the path exists
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
        
        if not checkpoint_path.is_file():
            raise ValueError(f"Checkpoint path is not a file: {checkpoint_path}")
        
        # Basic validation that it's a checkpoint file
        if not self._is_valid_checkpoint(checkpoint_path):
            raise RuntimeError(f"File does not appear to be a valid checkpoint: {checkpoint_path}")
        
        logger.info(f"✅ Local checkpoint validated: {checkpoint_path}")
        
        # For local paths, we don't extract additional config
        return checkpoint_path, None
    
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