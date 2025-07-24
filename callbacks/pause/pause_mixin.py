"""Mixin providing pause functionality to Lightning callbacks."""

import time
import sys
from typing import Any, Optional
import lightning.pytorch as pl
from lightning.pytorch.core import LightningModule

from .pause_state_machine import PauseStateMachine, PauseState, PauseAction, PauseStatusMessageFactory
from .unified_keyboard_handler import create_keyboard_handler, KeyboardHandler
from .pause_config import PauseConfig
import logging

logger = logging.getLogger(__name__)


class PauseMixin:
    """Mixin providing pause functionality to Lightning callbacks."""
    
    def __init__(self, pause_config: Optional[PauseConfig] = None, **kwargs):
        """
        Initialize pause mixin.
        
        Args:
            pause_config: Configuration for pause functionality
            **kwargs: Additional arguments passed to parent classes
        """
        super().__init__(**kwargs)
        
        # Configuration
        self._pause_config = pause_config or PauseConfig()
        
        # State management
        self._pause_state_machine = PauseStateMachine()
        self._message_factory = PauseStatusMessageFactory(
            self._pause_config.pause_key,
            self._pause_config.upload_key
        )
        
        # Keyboard handling
        self._keyboard_handler: Optional[KeyboardHandler] = None
        self._last_key_time = 0
        
        # Polling optimization
        self._batch_counter = 0
        self._keyboard_poll_frequency = self._pause_config.keyboard_poll_frequency
        
        # Original argv for resume command generation
        self._original_argv = sys.argv.copy()
        
        # Initialize keyboard handler if pause is enabled
        if self._pause_config.enable_pause:
            self._keyboard_handler = create_keyboard_handler(
                self._pause_config.keyboard_strategy,
                self._pause_config.debounce_interval
            )
            
            if self._keyboard_handler and self._keyboard_handler.is_available():
                logger.info("🎹 Pause functionality enabled - keyboard monitoring available")
                logger.info(f"   Controls: '{self._pause_config.pause_key}' to pause, '{self._pause_config.upload_key}' to toggle upload")
            else:
                logger.info("⚠️  Pause functionality not available (no TTY or termios)")
                self._pause_config.enable_pause = False
    
    # Pause-specific properties
    @property
    def pause_state(self) -> PauseState:
        """Get current pause state."""
        return self._pause_state_machine.state
    
    @property
    def is_pause_requested(self) -> bool:
        """Check if pause is requested."""
        return self._pause_state_machine.is_pause_scheduled()
    
    @property
    def is_upload_requested(self) -> bool:
        """Check if upload is requested with pause."""
        return self._pause_state_machine.is_upload_requested()
    
    # Keyboard polling optimization with responsiveness protection
    def _should_check_keyboard(self, force_responsive: bool = False) -> bool:
        """
        Centralized logic for when to check keyboard input.
        
        Args:
            force_responsive: If True, always check (for responsive hook points)
        
        Uses hybrid approach:
        - Responsive mode: Always check (matches commit 0aca683 behavior)  
        - Hybrid mode: Time-based + batch-based checking (efficiency optimization)
        """
        if not self._pause_config.enable_pause:
            return False
        
        # Responsive mode: Always check (for maximum responsiveness)
        if force_responsive:
            return True
        
        current_time = time.time()
        
        # Time-based check: Always check if it's been too long (responsiveness protection)
        if not hasattr(self, '_last_keyboard_check_time'):
            self._last_keyboard_check_time = current_time
            return True  # Always check on first call
        
        time_since_last_check = current_time - self._last_keyboard_check_time
        max_time_between_checks = self._pause_config.max_time_between_checks
        
        # Batch-based check: Every N batches (overhead reduction)
        self._batch_counter += 1
        batch_check_due = self._batch_counter % self._keyboard_poll_frequency == 0
        
        # Check keyboard if either condition is met
        should_check = time_since_last_check > max_time_between_checks or batch_check_due
        
        if should_check:
            self._last_keyboard_check_time = current_time
        
        return should_check
    
    def _process_keyboard_input(self) -> None:
        """Process keyboard input and handle pause/upload requests."""
        if not self._keyboard_handler:
            return
        
        key = self._keyboard_handler.get_key()
        if not key:
            return
        
        current_time = time.time()
        if current_time - self._last_key_time < self._pause_config.debounce_interval:
            return  # Debounce
        
        self._last_key_time = current_time
        
        if key == self._pause_config.pause_key:
            self._handle_pause_key_press()
        elif key == self._pause_config.upload_key:
            self._handle_upload_key_press()
    
    def _handle_pause_key_press(self) -> None:
        """Handle pause key press with state machine."""
        old_state = self._pause_state_machine.state
        
        if self._pause_state_machine.toggle_pause():
            new_state = self._pause_state_machine.state
            
            if new_state == PauseState.RUNNING:
                # Unpause
                print(self._message_factory.get_console_message("pause_cancelled"))
            else:
                # Pause scheduled
                print(self._message_factory.get_console_message("pause_scheduled"))
    
    def _handle_upload_key_press(self) -> None:
        """Handle upload toggle key press."""
        if not self._pause_state_machine.is_pause_scheduled():
            return  # Upload toggle only works when pause is scheduled
        
        if self._pause_state_machine.toggle_upload():
            if self._pause_state_machine.is_upload_requested():
                print(self._message_factory.get_console_message("upload_enabled"))
            else:
                print(self._message_factory.get_console_message("upload_disabled"))
    
    def _handle_pause_logic(self, trainer: pl.Trainer, pl_module: LightningModule) -> None:
        """
        Encapsulated pause logic that can be called from lifecycle methods.
        
        This is the main entry point for pause functionality from callback methods.
        """
        # Check keyboard input with hybrid approach
        if self._should_check_keyboard():
            self._process_keyboard_input()
        
        # Execute pause if scheduled and we're at a validation boundary
        if (self._pause_state_machine.should_execute_pause() and 
            hasattr(self, '_at_validation_boundary') and 
            self._at_validation_boundary):
            self._execute_pause_at_validation_boundary(trainer, pl_module)
    
    def _handle_responsive_pause_logic(self, trainer: pl.Trainer, pl_module: LightningModule) -> None:
        """
        Highly responsive pause logic for frequent checking points.
        
        This should be called from multiple hook points for maximum responsiveness
        as established in commit 0aca683.
        """
        # Check if responsive mode is enabled
        if self._pause_config.use_responsive_mode:
            # Always check keyboard for maximum responsiveness (matches original behavior)
            if self._should_check_keyboard(force_responsive=True):
                self._process_keyboard_input()
        
        # Note: Don't execute pause here - only at validation boundaries
    
    def _execute_pause_at_validation_boundary(self, trainer: pl.Trainer, pl_module: LightningModule) -> None:
        """Execute pause at validation boundary."""
        print(self._message_factory.get_console_message("executing_pause"))
        
        # Create pause checkpoint
        checkpoint_path = self._create_pause_checkpoint(trainer, pl_module)
        
        # Handle upload if requested
        upload_successful = False
        if self._pause_state_machine.is_upload_requested():
            upload_successful = self._upload_pause_checkpoint(trainer, pl_module, checkpoint_path)
        
        # Generate resume commands
        self._print_resume_commands(trainer, checkpoint_path, upload_successful)
        
        # Reset state machine and stop training
        self._pause_state_machine.execute_pause()  # This resets to RUNNING
        trainer.should_stop = True
    
    def _create_pause_checkpoint(self, trainer: pl.Trainer, pl_module: LightningModule) -> str:
        """Create pause checkpoint and return its path."""
        # This method should be implemented by the concrete callback
        # that inherits from this mixin
        raise NotImplementedError("Subclass must implement _create_pause_checkpoint")
    
    def _upload_pause_checkpoint(self, trainer: pl.Trainer, pl_module: LightningModule, checkpoint_path: str) -> bool:
        """
        Upload pause checkpoint if W&B is available.
        
        Returns:
            True if upload was successful, False otherwise
        """
        # This method should be implemented by the concrete callback
        # that inherits from this mixin and has W&B functionality
        logger.debug("Upload requested but not implemented in mixin")
        return False
    
    def _print_resume_commands(self, trainer: pl.Trainer, checkpoint_path: str, upload_successful: bool) -> None:
        """Print resume commands based on what's available."""
        print(f"\n🔄 Training paused. Resume options:")
        print(f"📁 Local resume:  python {' '.join(self._original_argv)} --ckpt_path {checkpoint_path}")
        
        if upload_successful:
            print(f"☁️  W&B resume:   Use the artifact resume command from W&B logs above")
    
    # Progress bar message integration
    def _get_pause_status_suffix_for_interval_bar(self) -> str:
        """Get pause status suffix for interval progress bar."""
        if not self._pause_config.enable_pause:
            return ""
        return self._message_factory.get_interval_bar_message(self._pause_state_machine.state)
    
    def _get_pause_status_suffix_for_global_bar(self) -> str:
        """Get pause status suffix for global progress bar."""
        if not self._pause_config.enable_pause:
            return ""
        return self._message_factory.get_global_bar_message(self._pause_state_machine.state)
    
    # Lifecycle management
    def _start_pause_monitoring(self) -> None:
        """Start keyboard monitoring when training begins."""
        if self._keyboard_handler and self._pause_config.enable_pause:
            self._keyboard_handler.start_monitoring()
    
    def _stop_pause_monitoring(self) -> None:
        """Stop keyboard monitoring when training ends."""
        if self._keyboard_handler:
            self._keyboard_handler.stop_monitoring()
    
    def _reset_pause_state_on_resume(self) -> None:
        """Reset pause state when resuming from checkpoint."""
        # Reset state machine to running
        self._pause_state_machine.reset()
        
        # Reset trainer.should_stop if it was set by pause
        # This should be called from on_load_checkpoint
    
    # Utility methods for subclasses
    def _extract_config_path_from_argv(self) -> Optional[str]:
        """Extract config path from command line arguments."""
        try:
            for i, arg in enumerate(self._original_argv):
                if arg == "--config" and i + 1 < len(self._original_argv):
                    return self._original_argv[i + 1]
            return None
        except Exception as e:
            logger.warning(f"Failed to extract config path from argv: {e}")
            return None
    
    def _mark_validation_boundary(self, is_boundary: bool = True) -> None:
        """Mark whether we're at a validation boundary (for subclasses to use)."""
        self._at_validation_boundary = is_boundary