"""Improved non-blocking keyboard handler with single-character validation."""

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
    """Keyboard handler that only accepts single-character input (no bulk/automated input)."""
    
    def __init__(self, single_char_window: float = 0.05):
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._key_queue: Queue = Queue()
        self._original_settings = None
        self._terminal_mode_set = False
        
        # Single character validation
        self._single_char_window = single_char_window  # 50ms window to check for trailing characters
    
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
            
            print(f"⌨️  Keyboard monitoring started (single-character input only)")
            
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
        """Monitor keyboard input - only accept single characters with no trailing input."""
        while self._monitoring:
            try:
                # Check if input is available
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    # Read the first character
                    char = sys.stdin.read(1)
                    
                    if char:
                        # Wait to see if more characters follow
                        time.sleep(self._single_char_window)  # Wait 50ms
                        
                        # Check if more input arrived during the wait
                        if select.select([sys.stdin], [], [], 0)[0]:
                            # More input arrived - this is bulk/automated input
                            # Consume and discard all the buffered input
                            consumed_chars = [char]
                            while select.select([sys.stdin], [], [], 0)[0]:
                                next_char = sys.stdin.read(1)
                                if next_char:
                                    consumed_chars.append(next_char)
                                else:
                                    break
                            
                            # Log what we ignored
                            ignored_input = ''.join(consumed_chars)
                            if len(ignored_input) > 20:
                                print(f"🛡️ Ignored bulk input: {repr(ignored_input[:20])}... ({len(ignored_input)} chars)")
                            else:
                                print(f"🛡️ Ignored bulk input: {repr(ignored_input)}")
                        else:
                            # No trailing characters - this is genuine single-character input
                            self._key_queue.put(char)
                
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


def create_improved_keyboard_handler(single_char_window: float = 0.05):
    """Create improved keyboard handler with single-character validation.
    
    Args:
        single_char_window: Time window (in seconds) to wait for trailing characters.
                          Only accepts input if no additional characters arrive within this window.
    """
    handler = ImprovedKeyboardHandler(single_char_window)
    if handler.is_available():
        return handler
    else:
        return NoOpKeyboardHandler()