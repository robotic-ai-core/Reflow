"""
WandB Artifact Checkpoint state manager for checkpoint persistence.
"""

from typing import Dict, Any
from .manager_state import ManagerState


class WandbArtifactState(ManagerState):
    """
    Manager state for WandbArtifactCheckpoint callback.

    Handles persistence of upload tracking state across checkpoint save/load cycles.
    """

    def __init__(self, callback):
        """
        Initialize with reference to the WandbArtifactCheckpoint callback.

        Args:
            callback: WandbArtifactCheckpoint instance
        """
        self.callback = callback

    @property
    def manager_name(self) -> str:
        """Return unique identifier for this manager."""
        return "wandb_artifact_checkpoint"

    def capture_state(self) -> Dict[str, Any]:
        """
        Capture current state for checkpoint persistence.

        Returns:
            Dictionary containing callback state
        """
        return {
            'config': self.callback.config.__dict__,
            'state': self.callback.state.__dict__,
            'version': '2.0'
        }

    def restore_state(self, state: Dict[str, Any]) -> bool:
        """
        Restore state from checkpoint.

        Args:
            state: Previously captured state dictionary

        Returns:
            True if restoration successful
        """
        if 'state' in state:
            for key, value in state['state'].items():
                if hasattr(self.callback.state, key):
                    setattr(self.callback.state, key, value)
        return True

    def validate_state(self, state: Dict[str, Any]) -> bool:
        """
        Validate that state dictionary is properly formatted.

        Args:
            state: State dictionary to validate

        Returns:
            True if state is valid
        """
        return isinstance(state, dict) and 'version' in state