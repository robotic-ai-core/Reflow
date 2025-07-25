"""Improved non-blocking keyboard handler with better responsiveness."""

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
    """Much more responsive keyboard handler with better buffering."""
    
    def __init__(self, debounce_interval: float = 0.2):
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._key_queue: Queue = Queue()
        self._last_key_time = 0
        self._debounce_interval = debounce_interval
        self._original_settings = None
        self._terminal_mode_set = False
    
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
        """Monitor keyboard input with better responsiveness."""
        while self._monitoring:
            try:
                # Much longer timeout for better key capture (200ms)
                # This means even brief key presses are more likely to be caught
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
                        
                        # Smart debouncing - only debounce the same character
                        current_time = time.time()
                        if (current_time - self._last_key_time > self._debounce_interval):
                            self._key_queue.put(char)
                            self._last_key_time = current_time
                
                # Shorter sleep for more responsive monitoring
                time.sleep(0.02)  # 20ms instead of 50ms
                
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


def create_improved_keyboard_handler(debounce_interval: float = 0.2):
    """Create improved keyboard handler with configurable debouncing."""
    handler = ImprovedKeyboardHandler(debounce_interval)
    if handler.is_available():
        return handler
    else:
        return NoOpKeyboardHandler()