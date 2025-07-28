"""
Shared configuration for Lightning Reflow components.

This module centralizes common configuration that is shared between 
LightningReflow (core) and LightningReflowCLI classes to eliminate duplication.
"""

from typing import Dict, Any, List
import logging
from lightning.pytorch.callbacks import Callback

logger = logging.getLogger(__name__)

# Shared trainer defaults for both Core and CLI
SHARED_TRAINER_DEFAULTS = {
    "enable_progress_bar": False,  # Disable Lightning's default to prevent conflicts with FlowProgressBarCallback
}

# Default callback configurations
DEFAULT_PAUSE_CALLBACK_CONFIG = {
    'checkpoint_dir': 'pause_checkpoints',
    'enable_pause': True,
    'pause_key': 'p',
    'upload_key': 'w',
    'debounce_interval': 0.3,
    'refresh_rate': 1,
    'bar_colour': '#fcac17',
    'global_bar_metrics': ['*lr*'],
    'interval_bar_metrics': ['loss', 'train/loss', 'train_loss'],
    'logging_interval': 'step',
}

DEFAULT_STEP_LOGGER_CONFIG = {
    'train_prog_bar_metrics': ['loss', 'train/loss'],
    'val_prog_bar_metrics': ['val_loss', 'val/val_loss']
}


def get_trainer_defaults(user_defaults: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Get trainer configuration with shared defaults.
    
    Args:
        user_defaults: User-provided trainer defaults that can override shared defaults
        
    Returns:
        Merged trainer configuration with shared defaults + user overrides
    """
    trainer_config = SHARED_TRAINER_DEFAULTS.copy()
    
    if user_defaults:
        trainer_config.update(user_defaults)
    
    return trainer_config


def ensure_essential_callbacks(callbacks: List[Callback], trainer=None) -> List[Callback]:
    """
    Ensure essential Lightning Reflow callbacks are present.
    
    Args:
        callbacks: Existing list of callbacks
        trainer: Trainer instance (optional, for CLI compatibility)
        
    Returns:
        List of callbacks with essential callbacks ensured
    """
    callbacks = list(callbacks) if callbacks else []
    
    # Ensure PauseCallback
    _ensure_pause_callback(callbacks)
    
    # Ensure StepOutputLoggerCallback  
    _ensure_step_output_logger(callbacks)
    
    return callbacks


def _ensure_pause_callback(callbacks: List[Callback]) -> None:
    """Ensure PauseCallback is present in callbacks list."""
    try:
        from ..callbacks.pause import PauseCallback
        
        # Check if PauseCallback is already present
        has_pause_callback = any(isinstance(cb, PauseCallback) for cb in callbacks)
        
        if not has_pause_callback:
            pause_callback = PauseCallback(**DEFAULT_PAUSE_CALLBACK_CONFIG)
            callbacks.append(pause_callback)
            logger.info("✅ Automatically added PauseCallback for progress bar functionality")
            
    except ImportError as e:
        logger.warning(f"Could not import PauseCallback: {e}")
    except Exception as e:
        logger.warning(f"Failed to ensure PauseCallback: {e}")


def _ensure_step_output_logger(callbacks: List[Callback]) -> None:
    """Ensure StepOutputLoggerCallback is present in callbacks list."""
    try:
        from ..callbacks.logging import StepOutputLoggerCallback
        
        # Check if StepOutputLoggerCallback is already present
        has_step_logger = any(isinstance(cb, StepOutputLoggerCallback) for cb in callbacks)
        
        if not has_step_logger:
            step_logger = StepOutputLoggerCallback(**DEFAULT_STEP_LOGGER_CONFIG)
            callbacks.append(step_logger)
            logger.info("✅ Automatically added StepOutputLoggerCallback for metrics logging")
            
    except ImportError as e:
        logger.warning(f"Could not import StepOutputLoggerCallback: {e}")
    except Exception as e:
        logger.warning(f"Failed to ensure StepOutputLoggerCallback: {e}") 