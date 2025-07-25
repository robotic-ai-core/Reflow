"""Non-blocking keyboard handler that doesn't interfere with terminal output."""

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


class NonBlockingKeyboardHandler:
    """Keyboard handler that preserves terminal state for output."""
    
    def __init__(self):
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._key_queue: Queue = Queue()
        self._last_key_time = 0
        self._debounce_interval = 0.1
    
    def is_available(self) -> bool:
        """Check if keyboard handling is available."""
        return HAS_TERMIOS and sys.stdin.isatty()
    
    def start_monitoring(self) -> None:
        """Start keyboard monitoring."""
        if not self.is_available() or self._monitoring:
            return
            
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_keyboard, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop keyboard monitoring."""
        self._monitoring = False
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=1.0)
    
    def get_key(self) -> Optional[str]:
        """Get the next key press if available."""
        try:
            return self._key_queue.get_nowait()
        except Empty:
            return None
    
    def _monitor_keyboard(self) -> None:
        """Monitor keyboard input without permanently changing terminal mode."""
        while self._monitoring:
            try:
                # Save current terminal settings
                old_settings = termios.tcgetattr(sys.stdin.fileno())
                
                try:
                    # Temporarily set terminal to cbreak mode
                    tty.setcbreak(sys.stdin.fileno())
                    
                    # Check for input with very short timeout
                    if select.select([sys.stdin], [], [], 0.01)[0]:
                        char = sys.stdin.read(1)
                        
                        # Debouncing
                        current_time = time.time()
                        if current_time - self._last_key_time > self._debounce_interval:
                            self._key_queue.put(char)
                            self._last_key_time = current_time
                
                finally:
                    # ALWAYS restore terminal settings immediately
                    termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_settings)
                
                # Sleep a bit to avoid busy-waiting
                time.sleep(0.05)
                
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


def create_non_blocking_keyboard_handler():
    """Create appropriate keyboard handler for the system."""
    handler = NonBlockingKeyboardHandler()
    if handler.is_available():
        return handler
    else:
        return NoOpKeyboardHandler()