"""Unified keyboard handler consolidating all keyboard input strategies."""

import sys
import threading
import time
from abc import ABC, abstractmethod
from queue import Queue, Empty
from typing import Optional, Protocol
from enum import Enum
import logging

try:
    import termios
    import tty
    import select
    HAS_TERMIOS = True
except ImportError:
    HAS_TERMIOS = False

logger = logging.getLogger(__name__)


class KeyboardHandlerStrategy(Enum):
    """Different strategies for keyboard input handling."""
    PERSISTENT_MODE = "persistent"   # Keep terminal in cbreak mode (most responsive)
    TRANSIENT_MODE = "transient"    # Switch terminal mode per read (safest)
    IMPROVED_MODE = "improved"      # Enhanced buffering with longer timeouts


class KeyboardHandler(Protocol):
    """Protocol defining the keyboard handler interface."""
    
    def start_monitoring(self) -> None:
        """Start monitoring for keyboard input."""
        ...
    
    def stop_monitoring(self) -> None:
        """Stop monitoring for keyboard input."""
        ...
    
    def get_key(self) -> Optional[str]:
        """Get the next key press if available, None otherwise."""
        ...
    
    def is_available(self) -> bool:
        """Check if keyboard handling is available on this system."""
        ...


class UnifiedKeyboardHandler:
    """Unified keyboard handler with configurable strategies."""
    
    def __init__(self, 
                 strategy: KeyboardHandlerStrategy = KeyboardHandlerStrategy.IMPROVED_MODE,
                 debounce_interval: float = 0.2):
        """
        Initialize unified keyboard handler.
        
        Args:
            strategy: Keyboard handling strategy to use
            debounce_interval: Minimum time between key presses to prevent duplicates
        """
        self._strategy = strategy
        self._debounce_interval = debounce_interval
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._key_queue: Queue = Queue()
        self._last_key_time = 0
        self._original_settings = None
        self._terminal_mode_set = False
        self._retry_count = 0
        self._max_retries = 3
    
    def is_available(self) -> bool:
        """Check if keyboard handling is available."""
        return HAS_TERMIOS and sys.stdin.isatty()
    
    def start_monitoring(self) -> None:
        """Start keyboard monitoring with the configured strategy."""
        if not self.is_available() or self._monitoring:
            return
        
        try:
            # Save original terminal settings
            self._original_settings = termios.tcgetattr(sys.stdin.fileno())
            
            # Set terminal mode based on strategy
            if self._strategy == KeyboardHandlerStrategy.PERSISTENT_MODE:
                tty.setcbreak(sys.stdin.fileno())
                self._terminal_mode_set = True
            
            self._monitoring = True
            self._monitor_thread = threading.Thread(target=self._monitor_keyboard, daemon=True)
            self._monitor_thread.start()
            
        except (termios.error, OSError) as e:
            logger.warning(f"Failed to initialize keyboard monitoring: {e}")
    
    def stop_monitoring(self) -> None:
        """Stop keyboard monitoring and restore terminal."""
        self._monitoring = False
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=1.0)
        
        # Restore original terminal settings
        if self._terminal_mode_set and self._original_settings:
            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._original_settings)
                self._terminal_mode_set = False
            except (termios.error, OSError):
                pass  # Ignore restoration errors
    
    def get_key(self) -> Optional[str]:
        """Get the next key press if available."""
        try:
            return self._key_queue.get_nowait()
        except Empty:
            return None
    
    def _monitor_keyboard(self) -> None:
        """Monitor keyboard input based on the configured strategy."""
        if self._strategy == KeyboardHandlerStrategy.PERSISTENT_MODE:
            self._monitor_persistent_mode()
        elif self._strategy == KeyboardHandlerStrategy.TRANSIENT_MODE:
            self._monitor_transient_mode()
        else:  # IMPROVED_MODE
            self._monitor_improved_mode()
    
    def _monitor_persistent_mode(self) -> None:
        """Monitor with persistent terminal mode (most responsive)."""
        while self._monitoring and self._retry_count < self._max_retries:
            try:
                # Check for input with short timeout
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    char = sys.stdin.read(1)
                    self._process_key_input(char)
                
                time.sleep(0.02)  # Short sleep for responsiveness
                self._retry_count = 0  # Reset on success
                
            except (termios.error, OSError) as e:
                self._handle_monitoring_error(e)
            except KeyboardInterrupt:
                logger.info("Keyboard monitoring interrupted by user")
                break
    
    def _monitor_transient_mode(self) -> None:
        """Monitor with transient terminal mode (safest)."""
        while self._monitoring and self._retry_count < self._max_retries:
            try:
                # Save current terminal settings
                old_settings = termios.tcgetattr(sys.stdin.fileno())
                
                try:
                    # Temporarily set terminal to cbreak mode
                    tty.setcbreak(sys.stdin.fileno())
                    
                    # Check for input with very short timeout
                    if select.select([sys.stdin], [], [], 0.01)[0]:
                        char = sys.stdin.read(1)
                        self._process_key_input(char)
                
                finally:
                    # Always restore terminal settings immediately
                    termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_settings)
                
                time.sleep(0.05)  # Moderate sleep
                self._retry_count = 0  # Reset on success
                
            except (termios.error, OSError) as e:
                self._handle_monitoring_error(e)
            except KeyboardInterrupt:
                logger.info("Keyboard monitoring interrupted by user")
                break
    
    def _monitor_improved_mode(self) -> None:
        """Monitor with improved buffering and longer timeouts."""
        # Set terminal mode once if not already set
        if not self._terminal_mode_set:
            try:
                tty.setcbreak(sys.stdin.fileno())
                self._terminal_mode_set = True
            except (termios.error, OSError) as e:
                logger.warning(f"Failed to set terminal mode: {e}")
                return
        
        while self._monitoring and self._retry_count < self._max_retries:
            try:
                # Longer timeout for better key capture (200ms)
                if select.select([sys.stdin], [], [], 0.2)[0]:
                    # Read available characters (might be multiple)
                    chars = []
                    while True:
                        # Check if more input is immediately available
                        if select.select([sys.stdin], [], [], 0)[0]:
                            char = sys.stdin.read(1)
                            if char:
                                chars.append(char)
                        else:
                            break
                    
                    # Process the most recent character (ignore key repeat/buffering)
                    if chars:
                        char = chars[-1]  # Take the last character
                        self._process_key_input(char)
                
                time.sleep(0.02)  # Short sleep for responsiveness
                self._retry_count = 0  # Reset on success
                
            except (termios.error, OSError) as e:
                self._handle_monitoring_error(e)
            except KeyboardInterrupt:
                logger.info("Keyboard monitoring interrupted by user")
                break
    
    def _process_key_input(self, char: str) -> None:
        """Process a key input with debouncing."""
        current_time = time.time()
        if current_time - self._last_key_time > self._debounce_interval:
            self._key_queue.put(char)
            self._last_key_time = current_time
    
    def _handle_monitoring_error(self, error: Exception) -> None:
        """Handle errors during keyboard monitoring with retry logic."""
        self._retry_count += 1
        if self._retry_count >= self._max_retries:
            logger.error(f"Terminal error after {self._max_retries} retries: {error}")
        else:
            logger.debug(f"Terminal error (retry {self._retry_count}/{self._max_retries}): {error}")
            time.sleep(0.1)  # Brief pause before retry
    
    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()


class NoOpKeyboardHandler:
    """No-op handler for systems without termios support."""
    
    def is_available(self) -> bool:
        return False
    
    def start_monitoring(self) -> None:
        pass
    
    def stop_monitoring(self) -> None:
        pass
    
    def get_key(self) -> Optional[str]:
        return None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def create_keyboard_handler(
    strategy: KeyboardHandlerStrategy = KeyboardHandlerStrategy.IMPROVED_MODE,
    debounce_interval: float = 0.2
) -> KeyboardHandler:
    """
    Factory function to create the appropriate keyboard handler.
    
    Args:
        strategy: Keyboard handling strategy to use
        debounce_interval: Minimum time between key presses
        
    Returns:
        Configured keyboard handler instance
    """
    handler = UnifiedKeyboardHandler(strategy, debounce_interval)
    if handler.is_available():
        return handler
    else:
        return NoOpKeyboardHandler()


# Backward compatibility aliases
def create_non_blocking_keyboard_handler():
    """Backward compatibility for non_blocking_keyboard_handler."""
    return create_keyboard_handler(KeyboardHandlerStrategy.TRANSIENT_MODE)


def create_improved_keyboard_handler(debounce_interval: float = 0.2):
    """Backward compatibility for improved_keyboard_handler."""
    return create_keyboard_handler(KeyboardHandlerStrategy.IMPROVED_MODE, debounce_interval)