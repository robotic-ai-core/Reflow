"""
Abstract base class for resume strategies.

This module defines the interface that all resume strategies must implement.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
from pathlib import Path


class ResumeStrategy(ABC):
    """
    Abstract base class for resume strategies.
    
    This interface allows different ways of resuming training (from local paths,
    W&B artifacts, S3 URLs, etc.) to be implemented as pluggable strategies.
    """
    
    @abstractmethod
    def prepare_resume(
        self, 
        resume_source: str, 
        **kwargs
    ) -> Tuple[Optional[Path], Optional[Dict[str, Any]]]:
        """
        Prepare for resuming training from the given source.
        
        Args:
            resume_source: Source identifier (e.g., path, artifact name, URL)
            **kwargs: Strategy-specific additional arguments
            
        Returns:
            Tuple of (checkpoint_path, additional_config)
            - checkpoint_path: Local path to the checkpoint file to resume from
            - additional_config: Optional configuration overrides from the checkpoint
            
        Raises:
            ValueError: If the resume source is invalid
            RuntimeError: If the resume operation fails
        """
        pass
    
    @abstractmethod
    def validate_source(self, resume_source: str) -> bool:
        """
        Validate that the resume source is compatible with this strategy.
        
        Args:
            resume_source: Source identifier to validate
            
        Returns:
            True if this strategy can handle the source, False otherwise
        """
        pass
    
    def cleanup(self) -> None:
        """
        Clean up any temporary resources created during resume preparation.
        
        This method is called after training completes or fails, allowing
        strategies to clean up temporary files, close connections, etc.
        """
        pass 