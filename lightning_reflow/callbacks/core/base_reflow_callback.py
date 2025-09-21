"""
Base callback for LightningReflow with common patterns.

This base class provides:
- Automatic manager state registration
- Common initialization patterns
- Shared utility methods
"""

import lightning.pytorch as pl
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
from ...utils.checkpoint.manager_state import ManagerState, register_manager
from ...utils.logging.logging_config import get_logger


class BaseReflowCallback(pl.Callback, ABC):
    """
    Base callback class for LightningReflow callbacks.

    Provides common functionality:
    - Automatic manager state registration
    - Logging setup
    - Common utilities
    """

    def __init__(self, enable_state_management: bool = True, verbose: bool = True):
        """
        Initialize base callback.

        Args:
            enable_state_management: Whether to enable automatic state management
            verbose: Whether to enable verbose logging
        """
        super().__init__()
        self.logger = get_logger(self.__class__.__name__)
        self.verbose = verbose
        self._state_manager: Optional[ManagerState] = None

        # Register manager state if enabled and provided by subclass
        if enable_state_management:
            manager = self.create_manager_state()
            if manager:
                self._state_manager = manager
                register_manager(manager)
                if self.verbose:
                    self.logger.info(f"🔗 {self.__class__.__name__} registered for state persistence")

    def create_manager_state(self) -> Optional[ManagerState]:
        """
        Create and return a manager state for this callback.

        Subclasses can override this to provide their own manager state.

        Returns:
            ManagerState instance or None if no state management needed
        """
        return None

    def log_verbose(self, message: str) -> None:
        """
        Log a message if verbose mode is enabled.

        Args:
            message: Message to log
        """
        if self.verbose:
            self.logger.info(message)

    def log_debug(self, message: str) -> None:
        """
        Log a debug message.

        Args:
            message: Debug message to log
        """
        self.logger.debug(message)

    def log_warning(self, message: str) -> None:
        """
        Log a warning message.

        Args:
            message: Warning message to log
        """
        self.logger.warning(message)

    def log_error(self, message: str, exc_info: bool = False) -> None:
        """
        Log an error message.

        Args:
            message: Error message to log
            exc_info: Whether to include exception info
        """
        self.logger.error(message, exc_info=exc_info)