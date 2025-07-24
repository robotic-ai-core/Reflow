"""Configuration objects for pause functionality."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
from .unified_keyboard_handler import KeyboardHandlerStrategy


@dataclass
class PauseConfig:
    """Configuration for pause functionality."""
    
    # Core pause settings
    enable_pause: bool = True
    pause_key: str = 'p'
    upload_key: str = 'w'
    
    # Keyboard handling
    keyboard_strategy: KeyboardHandlerStrategy = KeyboardHandlerStrategy.IMPROVED_MODE
    debounce_interval: float = 0.3
    keyboard_poll_frequency: int = 10  # Check every N batches (overhead reduction)
    max_time_between_checks: float = 0.5  # Never wait more than this (responsiveness protection)
    use_responsive_mode: bool = False  # If True, check keyboard at all hook points (max responsiveness)
    
    # Checkpoint settings
    checkpoint_dir: Path = field(default_factory=lambda: Path("pause_checkpoints"))
    overwrite_pause_checkpoints: bool = True
    
    # Smart timing
    enable_smart_checkpoint_timing: bool = True
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.debounce_interval < 0.1:
            raise ValueError("debounce_interval must be at least 0.1 seconds")
        
        if self.keyboard_poll_frequency < 1:
            raise ValueError("keyboard_poll_frequency must be at least 1")
        
        # Ensure checkpoint_dir is a Path object
        if isinstance(self.checkpoint_dir, str):
            self.checkpoint_dir = Path(self.checkpoint_dir)


@dataclass
class UploadConfig:
    """Configuration for W&B uploads."""
    
    # Upload triggers
    upload_best_model: bool = True
    upload_last_model: bool = True
    upload_all_checkpoints: bool = False
    
    # Upload requirements
    config_upload_required: bool = True
    checkpoint_upload_required: bool = False
    
    # Artifact settings
    artifact_type: str = "model"
    model_checkpoint_monitor_metric: Optional[str] = None
    
    # Upload timing
    min_training_minutes: float = 5.0
    min_training_minutes_for_exceptions: Optional[float] = None
    upload_on_exception: bool = True
    upload_on_teardown: bool = True
    
    # Emergency checkpoints
    create_emergency_checkpoints: bool = True
    cleanup_emergency_checkpoints: bool = True
    
    # Verbose logging
    wandb_verbose: bool = True
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.min_training_minutes < 0:
            raise ValueError("min_training_minutes must be non-negative")
        
        if (self.min_training_minutes_for_exceptions is not None and 
            self.min_training_minutes_for_exceptions < 0):
            raise ValueError("min_training_minutes_for_exceptions must be non-negative")


@dataclass
class ProgressConfig:
    """Configuration for progress bar display."""
    
    # Progress bar settings
    refresh_rate: int = 1
    process_position: int = 0
    bar_colour: Optional[str] = None
    
    # Metrics to display
    global_bar_metrics: Optional[List[str]] = field(default_factory=lambda: ['*lr*'])
    interval_bar_metrics: Optional[List[str]] = field(default_factory=lambda: ['loss'])
    
    # Logging settings
    logging_interval: str = "step"
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.refresh_rate < 1:
            raise ValueError("refresh_rate must be at least 1")
        
        if self.process_position < 0:
            raise ValueError("process_position must be non-negative")
        
        if self.logging_interval not in ["step", "epoch"]:
            raise ValueError("logging_interval must be 'step' or 'epoch'")
        
        # Ensure default values if None
        if self.global_bar_metrics is None:
            self.global_bar_metrics = ['*lr*']
        if self.interval_bar_metrics is None:
            self.interval_bar_metrics = ['loss']


@dataclass
class CallbackConfig:
    """Master configuration combining all callback settings."""
    
    pause_config: PauseConfig = field(default_factory=PauseConfig)
    upload_config: UploadConfig = field(default_factory=UploadConfig)
    progress_config: ProgressConfig = field(default_factory=ProgressConfig)
    
    @classmethod
    def create_minimal(cls, enable_pause: bool = True) -> 'CallbackConfig':
        """Create minimal configuration for basic usage."""
        return cls(
            pause_config=PauseConfig(enable_pause=enable_pause),
            upload_config=UploadConfig(),
            progress_config=ProgressConfig()
        )
    
    @classmethod
    def create_for_development(cls) -> 'CallbackConfig':
        """Create configuration optimized for development."""
        return cls(
            pause_config=PauseConfig(
                enable_pause=True,
                keyboard_poll_frequency=5,  # More responsive
                debounce_interval=0.2,      # Faster response
                max_time_between_checks=0.3,  # Very responsive (300ms max)
                use_responsive_mode=True,   # Maximum responsiveness for development
            ),
            upload_config=UploadConfig(
                upload_best_model=True,
                upload_last_model=False,    # Reduce uploads during dev
                min_training_minutes=1.0,   # Upload sooner
            ),
            progress_config=ProgressConfig(
                refresh_rate=2,             # More frequent updates
            )
        )
    
    @classmethod
    def create_for_production(cls) -> 'CallbackConfig':
        """Create configuration optimized for production."""
        return cls(
            pause_config=PauseConfig(
                enable_pause=False,         # Disable interactive pause
                keyboard_poll_frequency=20, # Less overhead
            ),
            upload_config=UploadConfig(
                upload_best_model=True,
                upload_last_model=True,
                min_training_minutes=10.0,  # Require substantial training
            ),
            progress_config=ProgressConfig(
                refresh_rate=1,             # Standard refresh rate
            )
        )
    
    def validate(self) -> None:
        """Validate the complete configuration."""
        # Individual configs validate themselves in __post_init__
        # Add any cross-config validation here
        
        if (self.pause_config.enable_pause and 
            not self.upload_config.upload_best_model and 
            not self.upload_config.upload_last_model):
            # Warning: pause enabled but no uploads configured
            import warnings
            warnings.warn(
                "Pause is enabled but no model uploads are configured. "
                "Consider enabling upload_best_model or upload_last_model."
            )
    
    def get_pause_keys_description(self) -> str:
        """Get description of pause key bindings."""
        if not self.pause_config.enable_pause:
            return "Pause functionality disabled"
        
        return (
            f"Pause: '{self.pause_config.pause_key}' key, "
            f"Upload toggle: '{self.pause_config.upload_key}' key"
        )


# Convenience functions for common configurations
def create_development_config() -> CallbackConfig:
    """Create development-optimized configuration."""
    return CallbackConfig.create_for_development()


def create_production_config() -> CallbackConfig:
    """Create production-optimized configuration."""
    return CallbackConfig.create_for_production()


def create_minimal_config(enable_pause: bool = True) -> CallbackConfig:
    """Create minimal configuration."""
    return CallbackConfig.create_minimal(enable_pause)