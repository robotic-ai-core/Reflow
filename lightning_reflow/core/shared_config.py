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


def get_trainer_defaults(user_defaults: Dict[str, Any] = None, disable_pause_callback: bool = False) -> Dict[str, Any]:
    """
    Get trainer configuration with shared defaults.
    
    Args:
        user_defaults: User-provided trainer defaults that can override shared defaults
        disable_pause_callback: If True, allow Lightning's default progress bar
        
    Returns:
        Merged trainer configuration with shared defaults + user overrides
    """
    trainer_config = SHARED_TRAINER_DEFAULTS.copy()
    
    # If pause callback is disabled, we can use Lightning's default progress bar
    if disable_pause_callback:
        # Allow Lightning's default progress bar when PauseCallback is disabled
        trainer_config['enable_progress_bar'] = True
        logger.info("📊 PauseCallback disabled - using Lightning's default progress bar")
    
    if user_defaults:
        # Check for progress bar override that could cause conflicts (only if PauseCallback is enabled)
        if not disable_pause_callback and 'enable_progress_bar' in user_defaults and user_defaults['enable_progress_bar'] is True:
            logger.warning("⚠️ User is trying to enable Lightning's default progress bar!")
            logger.warning("⚠️ This will conflict with Reflow's custom progress bar (PauseCallback).")
            logger.warning("⚠️ Keeping enable_progress_bar=False to prevent UI conflicts.")
            logger.warning("⚠️ If you need to disable progress bars entirely, set enable_pause=False in PauseCallback config.")
            
            # Apply all user defaults except the conflicting progress bar setting
            user_defaults_safe = user_defaults.copy()
            user_defaults_safe.pop('enable_progress_bar')
            trainer_config.update(user_defaults_safe)
        else:
            trainer_config.update(user_defaults)
    
    return trainer_config


def ensure_essential_callbacks(callbacks: List[Callback], trainer=None, disable_pause_callback: bool = False) -> List[Callback]:
    """
    Ensure essential Lightning Reflow callbacks are present.
    
    Args:
        callbacks: Existing list of callbacks
        trainer: Trainer instance (optional, for CLI compatibility)
        disable_pause_callback: If True, skip adding PauseCallback
        
    Returns:
        List of callbacks with essential callbacks ensured
    """
    callbacks = list(callbacks) if callbacks else []
    
    # Ensure PauseCallback (unless disabled)
    if not disable_pause_callback:
        _ensure_pause_callback(callbacks)
    
    # Ensure StepOutputLoggerCallback  
    _ensure_step_output_logger(callbacks)
    
    # Ensure WandbArtifactCheckpoint if W&B logger is present
    if trainer:
        _ensure_wandb_artifact_checkpoint(callbacks, trainer)
    
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


def _ensure_wandb_artifact_checkpoint(callbacks: List[Callback], trainer) -> None:
    """
    Ensure WandbArtifactCheckpoint is present when W&B logger is active.
    
    This provides automatic checkpoint artifact uploads to W&B when:
    1. W&B logger is present
    2. No WandbArtifactCheckpoint is already configured
    3. Environment suggests W&B integration is desired
    """
    try:
        # Check if W&B logger is present
        has_wandb_logger = False
        if hasattr(trainer, 'logger'):
            from lightning.pytorch.loggers import WandbLogger
            if isinstance(trainer.logger, WandbLogger):
                has_wandb_logger = True
            elif hasattr(trainer.logger, 'experiment_loggers'):
                # Check for W&B in multi-logger setup
                for exp_logger in trainer.logger.experiment_loggers:
                    if isinstance(exp_logger, WandbLogger):
                        has_wandb_logger = True
                        break
        
        if not has_wandb_logger:
            return  # No W&B logger, no need for artifact checkpoint
        
        from ..callbacks.wandb import WandbArtifactCheckpoint
        
        # Check if WandbArtifactCheckpoint is already present
        has_wandb_checkpoint = any(isinstance(cb, WandbArtifactCheckpoint) for cb in callbacks)
        
        if not has_wandb_checkpoint:
            # Create with smart defaults for LightningReflow
            wandb_checkpoint = WandbArtifactCheckpoint(
                upload_best_model=True,
                upload_last_model=True,
                upload_periodic_checkpoints=False,  # Avoid artifact spam by default
                upload_every_n_epoch=5,  # Upload every 5 epochs as a reasonable default
                monitor_pause_checkpoints=True,  # Integrate with PauseCallback
                create_emergency_checkpoints=True,  # Safety for crashes
                min_training_minutes=5.0,  # Don't upload for very short test runs
                min_training_minutes_for_exceptions=0.0,  # Always upload on crash
                use_compression=True,  # Save W&B storage
                upload_best_last_only_at_end=True,  # Storage optimization
                periodic_upload_pattern="timestamped",  # Most storage efficient
                wandb_verbose=False  # Less verbose by default
            )
            callbacks.append(wandb_checkpoint)
            logger.info("✅ Automatically added WandbArtifactCheckpoint for W&B artifact uploads")
            logger.info("   - Uploads: best & last models at end, periodic every 5 epochs")
            logger.info("   - Pause integration: enabled (uploads pause checkpoints on request)")
            logger.info("   - Emergency checkpoints: enabled (crash recovery)")
            
    except ImportError as e:
        logger.debug(f"Could not import W&B components: {e}")
    except Exception as e:
        logger.warning(f"Failed to ensure WandbArtifactCheckpoint: {e}") 