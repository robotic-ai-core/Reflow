"""
Flow Progress Bar state manager for checkpoint persistence.
"""

from typing import Dict, Any, Optional
from .manager_state import ManagerState


class FlowProgressBarState(ManagerState):
    """
    Manager state for FlowProgressBarCallback.

    Handles persistence of progress tracking metrics across checkpoint save/load cycles.
    """

    def __init__(self, callback):
        """
        Initialize with reference to the FlowProgressBarCallback.

        Args:
            callback: FlowProgressBarCallback instance
        """
        self.callback = callback

    @property
    def manager_name(self) -> str:
        """Return unique identifier for this manager."""
        return "flow_progress_bar"

    def capture_state(self) -> Dict[str, Any]:
        """
        Capture current progress state for checkpoint persistence.

        Returns:
            Dictionary containing progress metrics and state
        """
        import time
        return {
            'version': '1.0.0',
            'validation_count': self.callback._validation_count if hasattr(self.callback, '_validation_count') else 0,
            'last_validation_batch': self.callback._last_validation_batch if hasattr(self.callback, '_last_validation_batch') else 0,
            'configuration': {
                'refresh_rate': self.callback._refresh_rate if hasattr(self.callback, '_refresh_rate') else 1,
                'global_bar_metrics': self.callback.global_bar_metrics if hasattr(self.callback, 'global_bar_metrics') else None,
                'interval_bar_metrics': self.callback.interval_bar_metrics if hasattr(self.callback, 'interval_bar_metrics') else None,
                'bar_colour': self.callback._bar_colour if hasattr(self.callback, '_bar_colour') else None
            },
            'timestamp': time.time()
        }

    def restore_state(self, state: Dict[str, Any]) -> bool:
        """
        Restore progress state from checkpoint.

        Args:
            state: Previously captured state dictionary

        Returns:
            True if restoration successful
        """
        try:
            if not self.validate_state(state):
                return False

            # Restore critical tracking state
            self.callback._validation_count = state.get('validation_count', 0)
            self.callback._last_validation_batch = state.get('last_validation_batch', state.get('last_validation_step', 0))

            # Clear metric caches so they get rebuilt with correct state
            if hasattr(self.callback, '_global_metric_keys_cache'):
                self.callback._global_metric_keys_cache = None
            if hasattr(self.callback, '_interval_metric_keys_cache'):
                self.callback._interval_metric_keys_cache = None
            if hasattr(self.callback, '_available_metric_keys_cache'):
                self.callback._available_metric_keys_cache = None

            return True

        except Exception as e:
            print(f"Failed to restore FlowProgressBarCallback state: {e}")
            return False

    def validate_state(self, state: Dict[str, Any]) -> bool:
        """
        Validate that state dictionary is properly formatted.

        Args:
            state: State dictionary to validate

        Returns:
            True if state is valid
        """
        if not isinstance(state, dict):
            return False

        version = state.get('version')
        if version != '1.0.0':
            print(f"FlowProgressBarCallback: Incompatible state version {version}, expected 1.0.0")
            return False

        # Validate required fields (with backward compatibility)
        if 'validation_count' not in state:
            print(f"FlowProgressBarCallback: Missing required field in state: validation_count")
            return False

        # Check for either new or old field name (backward compatibility)
        if 'last_validation_batch' not in state and 'last_validation_step' not in state:
            print(f"FlowProgressBarCallback: Missing required field in state: last_validation_batch (or legacy last_validation_step)")
            return False

        return True