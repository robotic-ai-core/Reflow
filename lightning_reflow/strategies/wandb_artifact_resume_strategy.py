"""
W&B artifact resume strategy.

This strategy handles resuming from checkpoint files stored as W&B artifacts.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from .resume_strategy import ResumeStrategy
from ..services.wandb_service import WandbService

logger = logging.getLogger(__name__)


class WandbArtifactResumeStrategy(ResumeStrategy):
    """
    Resume strategy for W&B artifacts.
    
    This strategy handles resuming from checkpoint files that are stored
    as Weights & Biases artifacts.
    """
    
    def __init__(self):
        self.wandb_service = WandbService()
    
    def validate_source(self, resume_source: str) -> bool:
        """
        Validate that the source is a W&B artifact.
        
        Args:
            resume_source: Artifact identifier to validate
            
        Returns:
            True if it looks like a W&B artifact, False otherwise
        """
        try:
            # W&B artifacts typically have formats like:
            # - "entity/project/artifact:version"
            # - "artifact:version" (if in context)
            # - "run-id:latest"
            
            # Check for basic artifact patterns
            if ':' in resume_source:
                return True  # Has version specifier
            
            # Check for entity/project/artifact pattern
            if resume_source.count('/') >= 2:
                return True
            
            # If it contains no filesystem indicators, assume it might be W&B
            return not ('/' in resume_source and not resume_source.count('/') >= 2)
            
        except Exception:
            return False
    
    def prepare_resume(
        self, 
        resume_source: str, 
        entity: Optional[str] = None,
        project: Optional[str] = None,
        use_wandb_config: bool = False,
        **kwargs
    ) -> Tuple[Optional[Path], Optional[str]]:
        """
        Prepare for resuming from a W&B artifact.
        
        Args:
            resume_source: W&B artifact identifier
            entity: W&B entity (optional, can be inferred)
            project: W&B project (optional, can be inferred)  
            use_wandb_config: Whether to use config from W&B run
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (checkpoint_path, embedded_config_yaml)
            
        Raises:
            ValueError: If the artifact identifier is invalid
            RuntimeError: If the artifact download fails
        """
        logger.info(f"Preparing to resume from W&B artifact: {resume_source}")
        
        try:
            # Download the artifact
            download_path, artifact_metadata = self.wandb_service.download_artifact(
                artifact_name=resume_source,
                entity=entity,
                project=project
            )
            
            # Find the checkpoint file in the downloaded artifact
            checkpoint_path = self._find_checkpoint_in_artifact(download_path)
            
            # Extract embedded config from checkpoint - use Lightning's merged config
            from lightning_reflow.utils.checkpoint.checkpoint_utils import extract_embedded_config
            embedded_config_yaml = extract_embedded_config(str(checkpoint_path))
            
            if embedded_config_yaml:
                logger.info(f"📄 Found embedded configuration in checkpoint ({len(embedded_config_yaml)} chars)")
                logger.info("🎯 Using Lightning's original merged config for resume")
            else:
                logger.info("📄 No embedded configuration found in checkpoint")
            
            # Optionally override with config from W&B run (this path less common with embedded configs)
            if use_wandb_config and not embedded_config_yaml:
                import yaml
                wandb_config = self._get_wandb_config(artifact_metadata)
                if wandb_config:
                    embedded_config_yaml = yaml.dump(wandb_config, default_flow_style=False, sort_keys=False)
            
            logger.info(f"✅ W&B artifact prepared for resumption")
            logger.info(f"   Checkpoint: {checkpoint_path}")
            logger.info(f"   Artifact: {artifact_metadata['name']}:{artifact_metadata['version']}")
            
            return checkpoint_path, embedded_config_yaml
            
        except Exception as e:
            logger.error(f"Failed to prepare W&B artifact resume: {e}")
            raise RuntimeError(f"W&B artifact resume preparation failed: {e}")
    
    def _find_checkpoint_in_artifact(self, artifact_path: Path) -> Path:
        """
        Find the checkpoint file within the downloaded artifact.
        
        Args:
            artifact_path: Path to the downloaded artifact directory
            
        Returns:
            Path to the checkpoint file
            
        Raises:
            ValueError: If no checkpoint file is found
        """
        logger.debug(f"Searching for checkpoint in: {artifact_path}")
        
        # Common checkpoint file patterns
        checkpoint_patterns = [
            "*.ckpt",           # Lightning checkpoints
            "*.pt", "*.pth",    # PyTorch checkpoints
            "*.pkl", "*.pickle", # Pickle files
            "model.pt",         # Common naming convention
            "checkpoint.ckpt",  # Lightning default
            "last.ckpt",        # Lightning last checkpoint
            "best.ckpt"         # Lightning best checkpoint
        ]
        
        # Search for checkpoint files
        found_checkpoints = []
        for pattern in checkpoint_patterns:
            found_checkpoints.extend(artifact_path.glob(f"**/{pattern}"))
        
        if not found_checkpoints:
            # List all files for debugging
            all_files = list(artifact_path.glob("**/*"))
            logger.error(f"No checkpoint files found in artifact")
            logger.error(f"Available files: {[f.name for f in all_files if f.is_file()]}")
            raise ValueError(f"No checkpoint file found in W&B artifact at {artifact_path}")
        
        if len(found_checkpoints) == 1:
            checkpoint_path = found_checkpoints[0]
        else:
            # Multiple checkpoints found, try to pick the best one
            logger.warning(f"Multiple checkpoint files found: {[c.name for c in found_checkpoints]}")
            checkpoint_path = self._select_best_checkpoint(found_checkpoints)
        
        logger.info(f"Selected checkpoint: {checkpoint_path.name}")
        return checkpoint_path
    
    def _select_best_checkpoint(self, checkpoints: list) -> Path:
        """
        Select the best checkpoint from multiple options.
        
        Args:
            checkpoints: List of checkpoint paths
            
        Returns:
            Selected checkpoint path
        """
        # Priority order for checkpoint selection
        priority_names = ['best.ckpt', 'checkpoint.ckpt', 'last.ckpt', 'model.pt']
        
        # Try to find by priority name
        for priority_name in priority_names:
            for checkpoint in checkpoints:
                if checkpoint.name == priority_name:
                    logger.info(f"Selected checkpoint by priority: {checkpoint.name}")
                    return checkpoint
        
        # If no priority match, take the largest file (likely most complete)
        largest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_size)
        logger.info(f"Selected largest checkpoint: {largest_checkpoint.name}")
        return largest_checkpoint
    
    def _get_wandb_config(self, artifact_metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Get configuration from the W&B run that created this artifact.
        
        Args:
            artifact_metadata: Metadata from the downloaded artifact
            
        Returns:
            W&B run configuration or None if not available
        """
        try:
            entity = artifact_metadata['entity']
            project = artifact_metadata['project']
            
            # Try to extract run ID from artifact metadata
            # This is a heuristic and may need adjustment based on how artifacts are created
            run_id = None
            if 'metadata' in artifact_metadata and artifact_metadata['metadata']:
                run_id = artifact_metadata['metadata'].get('run_id')
            
            if not run_id:
                logger.warning("No run ID found in artifact metadata, cannot retrieve W&B config")
                return None
            
            logger.info(f"Retrieving config from W&B run: {entity}/{project}/{run_id}")
            return self.wandb_service.get_run_config(entity, project, run_id)
            
        except Exception as e:
            logger.warning(f"Failed to retrieve W&B config: {e}")
            return None
    
    def cleanup(self) -> None:
        """
        Clean up temporary files created during artifact download.
        """
        try:
            self.wandb_service.cleanup()
        except Exception as e:
            logger.warning(f"Error during W&B service cleanup: {e}") 