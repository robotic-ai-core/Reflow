"""Dedicated pause state machine for better control and validation."""

from enum import Enum, auto
from typing import Dict, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class PauseState(Enum):
    """Enhanced pause state machine for better UX control."""
    RUNNING = "running"
    PAUSE_SCHEDULED_NO_UPLOAD = "pause_scheduled_no_upload"
    PAUSE_SCHEDULED_WITH_UPLOAD = "pause_scheduled_with_upload"
    PAUSE_SCHEDULED_UPLOAD_ALL = "pause_scheduled_upload_all"


class PauseAction(Enum):
    """Actions that can trigger state transitions."""
    TOGGLE_PAUSE = auto()
    TOGGLE_UPLOAD = auto()
    RESET = auto()
    EXECUTE_PAUSE = auto()


class PauseStateMachine:
    """Dedicated state machine for pause functionality with validation."""
    
    def __init__(self, initial_state: PauseState = PauseState.RUNNING):
        """
        Initialize the pause state machine.
        
        Args:
            initial_state: Starting state for the machine
        """
        self._state = initial_state
        self._previous_state = initial_state
        
        # Define valid state transitions
        self._transitions: Dict[PauseState, Dict[PauseAction, PauseState]] = {
            PauseState.RUNNING: {
                PauseAction.TOGGLE_PAUSE: PauseState.PAUSE_SCHEDULED_NO_UPLOAD,
                PauseAction.RESET: PauseState.RUNNING,
            },
            PauseState.PAUSE_SCHEDULED_NO_UPLOAD: {
                PauseAction.TOGGLE_PAUSE: PauseState.RUNNING,
                PauseAction.TOGGLE_UPLOAD: PauseState.PAUSE_SCHEDULED_WITH_UPLOAD,
                PauseAction.RESET: PauseState.RUNNING,
                PauseAction.EXECUTE_PAUSE: PauseState.RUNNING,  # Reset after execution
            },
            PauseState.PAUSE_SCHEDULED_WITH_UPLOAD: {
                PauseAction.TOGGLE_PAUSE: PauseState.RUNNING,
                PauseAction.TOGGLE_UPLOAD: PauseState.PAUSE_SCHEDULED_UPLOAD_ALL,
                PauseAction.RESET: PauseState.RUNNING,
                PauseAction.EXECUTE_PAUSE: PauseState.RUNNING,  # Reset after execution
            },
            PauseState.PAUSE_SCHEDULED_UPLOAD_ALL: {
                PauseAction.TOGGLE_PAUSE: PauseState.RUNNING,
                PauseAction.TOGGLE_UPLOAD: PauseState.PAUSE_SCHEDULED_NO_UPLOAD,  # Cycle back to no upload
                PauseAction.RESET: PauseState.RUNNING,
                PauseAction.EXECUTE_PAUSE: PauseState.RUNNING,  # Reset after execution
            }
        }
        
        # Optional state change callbacks
        self._state_change_callbacks: Dict[PauseState, Optional[Callable]] = {}
    
    @property
    def state(self) -> PauseState:
        """Get current state."""
        return self._state
    
    @property
    def previous_state(self) -> PauseState:
        """Get previous state."""
        return self._previous_state
    
    def is_running(self) -> bool:
        """Check if in running state."""
        return self._state == PauseState.RUNNING
    
    def is_pause_scheduled(self) -> bool:
        """Check if pause is scheduled (any pause state)."""
        return self._state in [
            PauseState.PAUSE_SCHEDULED_NO_UPLOAD,
            PauseState.PAUSE_SCHEDULED_WITH_UPLOAD,
            PauseState.PAUSE_SCHEDULED_UPLOAD_ALL
        ]
    
    def is_upload_requested(self) -> bool:
        """Check if upload is requested with the pause."""
        return self._state in [PauseState.PAUSE_SCHEDULED_WITH_UPLOAD, PauseState.PAUSE_SCHEDULED_UPLOAD_ALL]
    
    def is_upload_all_requested(self) -> bool:
        """Check if upload all checkpoints is requested."""
        return self._state == PauseState.PAUSE_SCHEDULED_UPLOAD_ALL
    
    def should_execute_pause(self) -> bool:
        """Check if pause should be executed (alias for is_pause_scheduled)."""
        return self.is_pause_scheduled()
    
    def transition(self, action: PauseAction) -> bool:
        """
        Execute state transition.
        
        Args:
            action: Action to execute
            
        Returns:
            True if state changed, False otherwise
            
        Raises:
            ValueError: If action is invalid type
            RuntimeError: If current state is invalid
        """
        if not isinstance(action, PauseAction):
            raise ValueError(f"Invalid action type: {type(action)}")
        
        if self._state not in self._transitions:
            raise RuntimeError(f"Invalid state: {self._state}")
        
        if action not in self._transitions[self._state]:
            # Log but don't error - invalid transitions are ignored
            logger.debug(f"Invalid transition: {action} from state {self._state}")
            return False
        
        # Execute transition
        old_state = self._state
        self._previous_state = old_state
        self._state = self._transitions[self._state][action]
        
        # Call state change callback if registered
        if self._state in self._state_change_callbacks:
            callback = self._state_change_callbacks[self._state]
            if callback:
                try:
                    callback(old_state, self._state)
                except Exception as e:
                    logger.error(f"Error in state change callback: {e}")
        
        state_changed = old_state != self._state
        if state_changed:
            logger.debug(f"State transition: {old_state} -> {self._state} (action: {action})")
        
        return state_changed
    
    def reset(self) -> bool:
        """Reset to running state."""
        return self.transition(PauseAction.RESET)
    
    def toggle_pause(self) -> bool:
        """Toggle pause state."""
        return self.transition(PauseAction.TOGGLE_PAUSE)
    
    def toggle_upload(self) -> bool:
        """Toggle upload preference (only valid in pause scheduled states)."""
        return self.transition(PauseAction.TOGGLE_UPLOAD)
    
    def execute_pause(self) -> bool:
        """Execute pause and reset to running state."""
        return self.transition(PauseAction.EXECUTE_PAUSE)
    
    def register_state_change_callback(self, state: PauseState, callback: Callable) -> None:
        """
        Register a callback to be called when entering a specific state.
        
        Args:
            state: State to watch for
            callback: Function to call when entering this state (old_state, new_state)
        """
        self._state_change_callbacks[state] = callback
    
    def get_valid_actions(self) -> list[PauseAction]:
        """Get list of valid actions from current state."""
        return list(self._transitions.get(self._state, {}).keys())
    
    def can_transition(self, action: PauseAction) -> bool:
        """Check if a transition is valid from current state."""
        return action in self._transitions.get(self._state, {})
    
    def get_state_description(self) -> str:
        """Get human-readable description of current state."""
        descriptions = {
            PauseState.RUNNING: "Training is running normally",
            PauseState.PAUSE_SCHEDULED_NO_UPLOAD: "Pause scheduled, no upload",
            PauseState.PAUSE_SCHEDULED_WITH_UPLOAD: "Pause scheduled with upload",
            PauseState.PAUSE_SCHEDULED_UPLOAD_ALL: "Pause scheduled with upload all checkpoints"
        }
        return descriptions.get(self._state, f"Unknown state: {self._state}")
    
    def __repr__(self) -> str:
        return f"PauseStateMachine(state={self._state}, previous={self._previous_state})"
    
    def __str__(self) -> str:
        return f"State: {self._state.value}"


class PauseStatusMessageFactory:
    """Centralized factory for generating pause status messages."""
    
    def __init__(self, pause_key: str = 'p', upload_key: str = 'w'):
        """
        Initialize message factory.
        
        Args:
            pause_key: Key used for pause/unpause
            upload_key: Key used for upload toggle
        """
        self.pause_key = pause_key
        self.upload_key = upload_key
        
        # Centralized message templates
        self._interval_messages = {
            PauseState.RUNNING: f" - Press '{pause_key}' to pause",
            PauseState.PAUSE_SCHEDULED_NO_UPLOAD: f" - {pause_key}: to unpause, {upload_key}: upload pause ckpt",
            PauseState.PAUSE_SCHEDULED_WITH_UPLOAD: f" - {pause_key}: to unpause, {upload_key}: upload all (✓)",
            PauseState.PAUSE_SCHEDULED_UPLOAD_ALL: f" - {pause_key}: to unpause, {upload_key}: cancel upload (✓✓)"
        }
        
        self._console_messages = {
            "pause_scheduled": f"\n🔄 PAUSE scheduled - will pause at next validation boundary",
            "pause_cancelled": f"\n❌ PAUSE cancelled - training will continue",
            "upload_pause_enabled": f"\n📤 Upload ENABLED - pause checkpoint will be uploaded to W&B",
            "upload_all_enabled": f"\n📤📤 Upload ALL ENABLED - all checkpoints will be uploaded to W&B",
            "upload_disabled": f"\n📤 Upload DISABLED - checkpoint will not be uploaded to W&B",
            "executing_pause": f"🔄 Executing pause at validation boundary...",
        }
    
    def get_interval_bar_message(self, state: PauseState) -> str:
        """Get status message for interval progress bar."""
        return self._interval_messages.get(state, f" - Press '{self.pause_key}' to pause")
    
    def get_global_bar_message(self, state: PauseState) -> str:
        """Get status message for global progress bar (keep it clean)."""
        return ""  # Keep global bar clean
    
    def get_console_message(self, action: str) -> str:
        """Get console message for specific actions."""
        return self._console_messages.get(action, f"Unknown action: {action}")
    
    def update_keys(self, pause_key: str, upload_key: str) -> None:
        """Update key bindings and regenerate messages."""
        self.pause_key = pause_key
        self.upload_key = upload_key
        
        # Regenerate messages with new keys
        self._interval_messages = {
            PauseState.RUNNING: f" - Press '{pause_key}' to pause",
            PauseState.PAUSE_SCHEDULED_NO_UPLOAD: f" - {pause_key}: to unpause, {upload_key}: upload pause ckpt",
            PauseState.PAUSE_SCHEDULED_WITH_UPLOAD: f" - {pause_key}: to unpause, {upload_key}: upload all (✓)",
            PauseState.PAUSE_SCHEDULED_UPLOAD_ALL: f" - {pause_key}: to unpause, {upload_key}: cancel upload (✓✓)"
        }