"""
TrainerConfigState - Manager for preserving trainer configuration across resume operations.

This manager ensures that essential trainer configuration (max_epochs, max_steps, etc.)
is preserved during pause/resume cycles, maintaining consistent behavior for progress
bars and training duration calculations.
"""

import logging
from typing import Dict, Any

from .manager_state import ManagerState

logger = logging.getLogger(__name__)


class TrainerConfigState(ManagerState):
    """
    Manager state for preserving trainer configuration across checkpoints.
    
    This ensures that trainer configuration like max_epochs, max_steps is preserved
    during resume operations, preventing inconsistencies in progress bar calculations
    and training duration.
    """
    
    def __init__(self, trainer=None):
        """
        Initialize TrainerConfigState.
        
        Args:
            trainer: PyTorch Lightning trainer (can be None, will be set later)
        """
        self.trainer = trainer
        self._captured_config = None
    
    @property
    def manager_name(self) -> str:
        """Return the unique name of this manager for state storage."""
        return "trainer_config"
    
    def capture_state(self) -> Dict[str, Any]:
        """
        Capture essential trainer configuration for persistence.
        
        Returns:
            Dictionary containing trainer configuration that should be preserved
        """
        if not self.trainer:
            logger.warning("TrainerConfigState: No trainer available for state capture")
            return {}
        
        # Capture essential training configuration
        config = {
            'version': '1.0.0',
            'training_duration': {
                'max_epochs': getattr(self.trainer, 'max_epochs', None),
                'max_steps': getattr(self.trainer, 'max_steps', -1),
                'min_epochs': getattr(self.trainer, 'min_epochs', None), 
                'min_steps': getattr(self.trainer, 'min_steps', None)
            },
            'validation_config': {
                'val_check_interval': getattr(self.trainer, 'val_check_interval', None),
                'check_val_every_n_epoch': getattr(self.trainer, 'check_val_every_n_epoch', 1),
                'num_sanity_val_steps': getattr(self.trainer, 'num_sanity_val_steps', None)
            },
            'training_control': {
                'limit_train_batches': getattr(self.trainer, 'limit_train_batches', None),
                'limit_val_batches': getattr(self.trainer, 'limit_val_batches', None),
                'accumulate_grad_batches': getattr(self.trainer, 'accumulate_grad_batches', 1)
            }
        }
        
        self._captured_config = config
        
        logger.info(f"📋 Captured trainer config - max_epochs: {config['training_duration']['max_epochs']}, "
                   f"max_steps: {config['training_duration']['max_steps']}")
        
        return config
    
    def restore_state(self, state: Dict[str, Any]) -> bool:
        """
        Restore trainer configuration from saved data.
        
        Args:
            state: Dictionary containing saved trainer configuration
            
        Returns:
            True if restoration was successful, False otherwise
        """
        try:
            if not self.validate_state(state):
                return False
            
            if not self.trainer:
                logger.warning("TrainerConfigState: No trainer available for state restoration")
                return False
            
            # Restore training duration settings
            training_duration = state.get('training_duration', {})
            for key, value in training_duration.items():
                if hasattr(self.trainer, key) and value is not None:
                    self._safe_setattr(key, value)
            
            # Restore validation configuration
            validation_config = state.get('validation_config', {})
            for key, value in validation_config.items():
                if hasattr(self.trainer, key) and value is not None:
                    self._safe_setattr(key, value)
            
            # Restore training control settings
            training_control = state.get('training_control', {})
            for key, value in training_control.items():
                if hasattr(self.trainer, key) and value is not None:
                    self._safe_setattr(key, value)
            
            logger.info("✅ TrainerConfigState restoration completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to restore TrainerConfigState: {e}")
            return False
    
    def _safe_setattr(self, attr_name: str, value: Any) -> None:
        """
        Safely set trainer attribute, handling potential read-only attributes.
        
        Args:
            attr_name: Name of the attribute to set
            value: Value to set
        """
        try:
            setattr(self.trainer, attr_name, value)
            logger.debug(f"Set trainer.{attr_name} = {value}")
        except (AttributeError, TypeError) as e:
            logger.warning(f"Could not restore trainer.{attr_name}: {e}")
    
    def validate_state(self, state: Dict[str, Any]) -> bool:
        """
        Validate that the state dictionary contains valid trainer configuration.
        
        Args:
            state: State dictionary to validate
            
        Returns:
            True if state is valid, False otherwise
        """
        if not isinstance(state, dict):
            logger.warning("TrainerConfigState: Invalid state type, expected dict")
            return False
        
        # Check for version
        version = state.get('version')
        if not version:
            logger.warning("TrainerConfigState: No version in state")
            return False
        
        # Check for required sections
        required_sections = ['training_duration', 'validation_config', 'training_control']
        for section in required_sections:
            if section not in state:
                logger.warning(f"TrainerConfigState: Missing required section: {section}")
                return False
        
        logger.debug("✅ TrainerConfigState validation passed")
        return True
    
    def set_trainer(self, trainer):
        """
        Set the trainer instance for this state manager.
        
        Args:
            trainer: PyTorch Lightning trainer instance
        """
        self.trainer = trainer 