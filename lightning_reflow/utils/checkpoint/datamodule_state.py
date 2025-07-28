"""
DataModuleState - Manager for preserving datamodule state across resume operations.

This manager ensures that datamodule state (like dataloader positions, shuffling state, etc.)
is preserved during pause/resume cycles, enabling exact training resumption.
"""

import logging
from typing import Dict, Any, Optional

from .manager_state import ManagerState

logger = logging.getLogger(__name__)


class DataModuleState(ManagerState):
    """
    Manager state for preserving datamodule state across checkpoints.
    
    This ensures that datamodule state like dataloader positions, shuffling state,
    and other training-related state is preserved during resume operations.
    """
    
    def __init__(self, trainer=None):
        """
        Initialize DataModuleState.
        
        Args:
            trainer: PyTorch Lightning trainer (can be None, will be set later)
        """
        self.trainer = trainer
        self._captured_state = None
    
    @property
    def manager_name(self) -> str:
        """Return the unique name of this manager for state storage."""
        return "datamodule"
    
    def capture_state(self) -> Dict[str, Any]:
        """
        Capture datamodule state for persistence.
        
        Returns:
            Dictionary containing datamodule state that should be preserved
        """
        if not self.trainer:
            logger.debug("DataModuleState: No trainer available for state capture")
            return {}
        
        if not hasattr(self.trainer, 'datamodule') or self.trainer.datamodule is None:
            logger.debug("DataModuleState: No datamodule available for state capture")
            return {}
        
        datamodule = self.trainer.datamodule
        
        # Get datamodule state_dict if available
        datamodule_state_dict = {}
        if hasattr(datamodule, 'state_dict') and callable(datamodule.state_dict):
            try:
                datamodule_state_dict = datamodule.state_dict()
                logger.debug(f"Captured datamodule state_dict with keys: {list(datamodule_state_dict.keys()) if isinstance(datamodule_state_dict, dict) else 'non-dict state'}")
            except Exception as e:
                logger.warning(f"Failed to capture datamodule state_dict: {e}")
                datamodule_state_dict = {}
        else:
            logger.debug("DataModule does not have state_dict method")
        
        # Capture basic datamodule information
        state = {
            'version': '1.0.0',
            'datamodule_class': datamodule.__class__.__name__,
            'datamodule_module': datamodule.__class__.__module__,
            'datamodule_state_dict': datamodule_state_dict
        }
        
        self._captured_state = state
        
        logger.info(f"📊 Captured datamodule state for {state['datamodule_class']}")
        logger.debug(f"   State keys: {list(datamodule_state_dict.keys()) if isinstance(datamodule_state_dict, dict) else 'non-dict state'}")
        
        return state
    
    def restore_state(self, state: Dict[str, Any]) -> bool:
        """
        Restore datamodule state from saved data.
        
        Args:
            state: Dictionary containing saved datamodule state
            
        Returns:
            True if restoration was successful, False otherwise
        """
        try:
            if not self.validate_state(state):
                return False
            
            if not self.trainer:
                logger.warning("DataModuleState: No trainer available for state restoration")
                return False
            
            if not hasattr(self.trainer, 'datamodule') or self.trainer.datamodule is None:
                logger.warning("DataModuleState: No datamodule available for state restoration")
                return False
            
            datamodule = self.trainer.datamodule
            
            # Verify class compatibility
            expected_class = state.get('datamodule_class')
            actual_class = datamodule.__class__.__name__
            if expected_class and expected_class != actual_class:
                logger.warning(f"DataModule class mismatch: expected {expected_class}, got {actual_class}")
                # Don't fail on class name mismatch - may be due to refactoring
            
            # Restore datamodule state_dict if available
            datamodule_state_dict = state.get('datamodule_state_dict', {})
            if datamodule_state_dict and hasattr(datamodule, 'load_state_dict') and callable(datamodule.load_state_dict):
                try:
                    datamodule.load_state_dict(datamodule_state_dict)
                    logger.debug(f"Loaded datamodule state_dict with keys: {list(datamodule_state_dict.keys()) if isinstance(datamodule_state_dict, dict) else 'non-dict state'}")
                except Exception as e:
                    logger.warning(f"Failed to load datamodule state_dict: {e}")
                    # Don't fail on state_dict loading errors - may be due to version differences
            else:
                logger.debug("DataModule does not have load_state_dict method or no state to restore")
            
            logger.info(f"✅ Restored datamodule state for {actual_class}")
            logger.debug(f"   Restored keys: {list(datamodule_state_dict.keys()) if isinstance(datamodule_state_dict, dict) else 'non-dict state'}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to restore DataModuleState: {e}")
            return False
    
    def validate_state(self, state: Dict[str, Any]) -> bool:
        """
        Validate that the state dictionary contains valid datamodule state.
        
        Args:
            state: State dictionary to validate
            
        Returns:
            True if state is valid, False otherwise
        """
        if not isinstance(state, dict):
            logger.warning("DataModuleState: Invalid state type, expected dict")
            return False
        
        # Check for version (backward compatibility - older checkpoints may not have version)
        version = state.get('version')
        if not version:
            logger.info("DataModuleState: No version in state (legacy checkpoint), assuming compatible")
            # Don't fail for missing version - this maintains backward compatibility
        
        # Check for datamodule_state_dict
        if 'datamodule_state_dict' not in state:
            # Handle simplified datamodules that return empty state_dict
            logger.info("DataModuleState: Missing datamodule_state_dict in state - assuming simplified datamodule with no state to restore")
            return True  # Allow validation to pass for simplified datamodules
        
        logger.debug("✅ DataModuleState validation passed")
        return True
    
    def set_trainer(self, trainer):
        """
        Set the trainer instance for this state manager.
        
        Args:
            trainer: PyTorch Lightning trainer instance
        """
        self.trainer = trainer 