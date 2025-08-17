"""Improved non-blocking keyboard handler with simple bulk input rejection."""

import sys
import threading
import time
from queue import Queue, Empty
from typing import Optional

try:
    import termios
    import tty
    import select
    HAS_TERMIOS = True
except ImportError:
    HAS_TERMIOS = False


class ImprovedKeyboardHandler:
    """Keyboard handler with time-window detection for automated input."""
    
    def __init__(self, debounce_interval: float = 0.2, startup_grace_period: float = 2.0):
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._key_queue: Queue = Queue()
        self._original_settings = None
        self._terminal_mode_set = False
        self._char_window = 0.25  # 250ms window to check for following characters
    
    def is_available(self) -> bool:
        """Check if keyboard handling is available."""
        return HAS_TERMIOS and sys.stdin.isatty()
    
    def start_monitoring(self) -> None:
        """Start keyboard monitoring with persistent terminal mode."""
        if not self.is_available() or self._monitoring:
            return
            
        try:
            # Save original terminal settings ONCE
            self._original_settings = termios.tcgetattr(sys.stdin.fileno())
            
            # Set terminal to cbreak mode ONCE and keep it
            tty.setcbreak(sys.stdin.fileno())
            self._terminal_mode_set = True
            
            self._monitoring = True
            self._monitor_thread = threading.Thread(target=self._monitor_keyboard, daemon=True)
            self._monitor_thread.start()
            
            print(f"⌨️  Keyboard monitoring started")
            
        except (termios.error, OSError) as e:
            print(f"⚠️  Failed to initialize keyboard monitoring: {e}")
    
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
        """Monitor keyboard input with time-window detection for automated input."""
        while self._monitoring:
            try:
                # Check for input with timeout
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    # Read the first character
                    first_char = sys.stdin.read(1)
                    
                    if first_char:
                        # Wait briefly to see if more characters follow
                        time.sleep(self._char_window)
                        
                        # Collect any additional characters that arrived during the window
                        additional_chars = []
                        while select.select([sys.stdin], [], [], 0)[0]:
                            char = sys.stdin.read(1)
                            if char:
                                additional_chars.append(char)
                            else:
                                break
                        
                        # If more characters arrived, it's likely automated input
                        if additional_chars:
                            all_chars = first_char + ''.join(additional_chars)
                            print(f"🛡️ Ignored automated input: {repr(all_chars)}")
                        else:
                            # Single character with no followers - accept it
                            self._key_queue.put(first_char)
                
                # Small sleep to prevent CPU spinning
                time.sleep(0.01)
                
            except (termios.error, OSError, KeyboardInterrupt):
                break
    
    def __enter__(self):
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_monitoring()


class NoOpKeyboardHandler:
    """No-op handler for systems without termios."""
    
    def is_available(self) -> bool:
        return False
    
    def start_monitoring(self) -> None:
        pass
    
    def stop_monitoring(self) -> None:
        pass
    
    def get_key(self) -> Optional[str]:
        return None


def create_improved_keyboard_handler(debounce_interval: float = 0.2, startup_grace_period: float = 2.0):
    """Create improved keyboard handler with bulk input rejection.
    
    Args:
        debounce_interval: Kept for backward compatibility but not used
        startup_grace_period: Kept for backward compatibility but not used
    """
    handler = ImprovedKeyboardHandler(debounce_interval, startup_grace_period)
    if handler.is_available():
        return handler
    else:
        return NoOpKeyboardHandler()