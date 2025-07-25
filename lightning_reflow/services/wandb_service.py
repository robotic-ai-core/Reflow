"""
Weights & Biases service for centralized API interactions.

This service encapsulates all direct interactions with the W&B API, making it
easier to mock for testing and centralize API logic.
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import warnings

logger = logging.getLogger(__name__)


class WandbService:
    """
    Service class for interacting with Weights & Biases API.
    
    This class centralizes all W&B API calls, making it easier to:
    - Mock for testing
    - Handle API errors consistently
    - Add retry logic if needed
    - Cache API responses
    """
    
    def __init__(self):
        self._api = None
        self._temp_dirs: List[Path] = []
    
    @property
    def api(self):
        """Lazy initialization of W&B API."""
        if self._api is None:
            try:
                import wandb
                self._api = wandb.Api()
            except ImportError:
                raise ImportError("wandb is required for W&B artifact resumption. Install with: pip install wandb")
        return self._api
    
    def download_artifact(
        self, 
        artifact_name: str, 
        entity: Optional[str] = None,
        project: Optional[str] = None
    ) -> Tuple[Path, Dict[str, Any]]:
        """
        Download a W&B artifact to a temporary directory.
        
        Args:
            artifact_name: Name of the artifact (e.g., "run-id:latest" or "model-name:v1")
            entity: W&B entity (username/team), optional
            project: W&B project name, optional
            
        Returns:
            Tuple of (download_path, artifact_metadata)
            
        Raises:
            ValueError: If artifact_name is invalid
            RuntimeError: If download fails
        """
        logger.info(f"Downloading W&B artifact: {artifact_name}")
        
        try:
            # Parse the artifact name to handle different formats
            full_artifact_name = self._build_full_artifact_name(artifact_name, entity, project)
            
            # Get the artifact
            artifact = self.api.artifact(full_artifact_name)
            
            # Create temporary directory for download
            temp_dir = Path(tempfile.mkdtemp(prefix="wandb_artifact_"))
            self._temp_dirs.append(temp_dir)
            
            # Download the artifact
            logger.info(f"Downloading to: {temp_dir}")
            download_path = artifact.download(root=str(temp_dir))
            
            # Extract metadata
            metadata = {
                'entity': artifact.entity,
                'project': artifact.project,
                'name': artifact.name,
                'version': artifact.version,
                'size': artifact.size,
                'created_at': artifact.created_at,
                'description': artifact.description,
                'aliases': artifact.aliases,
                'metadata': artifact.metadata
            }
            
            logger.info(f"✅ Downloaded artifact {artifact.name}:{artifact.version} ({artifact.size} bytes)")
            
            return Path(download_path), metadata
            
        except Exception as e:
            logger.error(f"Failed to download W&B artifact '{artifact_name}': {e}")
            raise RuntimeError(f"W&B artifact download failed: {e}")
    
    def get_run_config(self, entity: str, project: str, run_id: str) -> Dict[str, Any]:
        """
        Get the configuration from a W&B run.
        
        Args:
            entity: W&B entity (username/team)
            project: W&B project name
            run_id: W&B run ID
            
        Returns:
            Run configuration as dictionary
            
        Raises:
            RuntimeError: If run retrieval fails
        """
        logger.info(f"Retrieving config for run: {entity}/{project}/{run_id}")
        
        try:
            run = self.api.run(f"{entity}/{project}/{run_id}")
            config = dict(run.config)
            logger.info(f"✅ Retrieved config with {len(config)} entries")
            return config
            
        except Exception as e:
            logger.error(f"Failed to get run config: {e}")
            raise RuntimeError(f"Failed to retrieve W&B run config: {e}")
    
    def list_artifacts(
        self, 
        entity: str, 
        project: str, 
        artifact_type: str = "model"
    ) -> List[Dict[str, Any]]:
        """
        List artifacts in a W&B project.
        
        Args:
            entity: W&B entity (username/team)
            project: W&B project name
            artifact_type: Type of artifacts to list
            
        Returns:
            List of artifact metadata dictionaries
        """
        logger.info(f"Listing {artifact_type} artifacts in {entity}/{project}")
        
        try:
            artifacts = self.api.artifacts(
                entity=entity,
                project=project,
                type_name=artifact_type
            )
            
            artifact_list = []
            for artifact in artifacts:
                artifact_list.append({
                    'name': artifact.name,
                    'version': artifact.version,
                    'size': artifact.size,
                    'created_at': artifact.created_at,
                    'aliases': artifact.aliases
                })
            
            logger.info(f"✅ Found {len(artifact_list)} artifacts")
            return artifact_list
            
        except Exception as e:
            logger.error(f"Failed to list artifacts: {e}")
            raise RuntimeError(f"Failed to list W&B artifacts: {e}")
    
    def _build_full_artifact_name(
        self, 
        artifact_name: str, 
        entity: Optional[str], 
        project: Optional[str]
    ) -> str:
        """
        Build the full artifact name from components.
        
        Args:
            artifact_name: Base artifact name
            entity: W&B entity (optional)
            project: W&B project (optional)
            
        Returns:
            Full artifact name in format "entity/project/name:version"
        """
        # If already fully qualified, return as-is
        if '/' in artifact_name and ':' in artifact_name:
            return artifact_name
        
        # If missing entity/project, try to get from current W&B context
        if not entity or not project:
            try:
                import wandb
                if wandb.run is not None:
                    entity = entity or wandb.run.entity
                    project = project or wandb.run.project
            except ImportError:
                pass
        
        # Build the full name
        if entity and project:
            if '/' not in artifact_name:
                return f"{entity}/{project}/{artifact_name}"
            else:
                return artifact_name
        else:
            # Return as-is and let W&B API handle it
            return artifact_name
    
    def cleanup(self) -> None:
        """
        Clean up temporary directories created during artifact downloads.
        """
        for temp_dir in self._temp_dirs:
            try:
                if temp_dir.exists():
                    import shutil
                    shutil.rmtree(temp_dir)
                    logger.debug(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary directory {temp_dir}: {e}")
        
        self._temp_dirs.clear()
    
    def __del__(self):
        """Cleanup on object destruction."""
        try:
            self.cleanup()
        except Exception:
            pass  # Don't raise exceptions in __del__ 