"""Improved non-blocking keyboard handler with better responsiveness and pyenv protection."""

import sys
import threading
import time
from queue import Queue, Empty
from typing import Optional, List
from collections import deque

try:
    import termios
    import tty
    import select
    HAS_TERMIOS = True
except ImportError:
    HAS_TERMIOS = False


class ImprovedKeyboardHandler:
    """Much more responsive keyboard handler with better buffering and automated input protection."""
    
    def __init__(self, debounce_interval: float = 0.2, startup_grace_period: float = 2.0):
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._key_queue: Queue = Queue()
        self._last_key_time = 0
        self._debounce_interval = debounce_interval
        self._original_settings = None
        self._terminal_mode_set = False
        
        # New: Protection against automated input
        self._startup_time = 0  # Time when monitoring started
        self._startup_grace_period = startup_grace_period  # Ignore input for this long after startup
        self._recent_chars = deque(maxlen=10)  # Track recent characters for pattern detection
        self._char_timestamps = deque(maxlen=10)  # Track timestamps of recent characters
        
        # Patterns to ignore (common automated inputs from IDEs)
        self._ignore_patterns = [
            'pyenv',  # VSCode/Cursor pyenv activation
            'source',  # Shell sourcing commands
            'export',  # Environment variable exports
            'conda',   # Conda activation
            'eval',    # Shell eval commands
        ]
    
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
            
            # Record startup time for grace period
            self._startup_time = time.time()
            
            self._monitoring = True
            self._monitor_thread = threading.Thread(target=self._monitor_keyboard, daemon=True)
            self._monitor_thread.start()
            
            print(f"🛡️ Keyboard monitoring started with {self._startup_grace_period}s grace period to prevent automated input")
            
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
    
    def _is_automated_input(self, char: str) -> bool:
        """Check if the current character is part of automated input."""
        current_time = time.time()
        
        # Check 1: Startup grace period
        if current_time - self._startup_time < self._startup_grace_period:
            return True  # Ignore all input during grace period
        
        # Update recent character tracking
        self._recent_chars.append(char)
        self._char_timestamps.append(current_time)
        
        # Check 2: Rapid multi-character input detection
        if len(self._char_timestamps) >= 5:
            # If 5+ characters arrived within 100ms, it's likely automated
            time_span = self._char_timestamps[-1] - self._char_timestamps[-5]
            if time_span < 0.1:  # 100ms for 5 characters
                return True
        
        # Check 3: Pattern matching
        if len(self._recent_chars) >= 3:
            # Build recent string from last few characters
            recent_string = ''.join(self._recent_chars)
            
            # Check if any ignore pattern is forming
            for pattern in self._ignore_patterns:
                if pattern.startswith(recent_string.lower()) or recent_string.lower() in pattern:
                    # Pattern detected or forming
                    return True
        
        # Check 4: Special case for 'p' after very rapid input
        if char == 'p' and len(self._char_timestamps) >= 2:
            # If previous character was very recent (< 50ms), might be part of 'pyenv'
            if current_time - self._char_timestamps[-2] < 0.05:
                return True
        
        return False
    
    def _monitor_keyboard(self) -> None:
        """Monitor keyboard input with better responsiveness and automated input protection."""
        while self._monitoring:
            try:
                # Much longer timeout for better key capture (200ms)
                # This means even brief key presses are more likely to be caught
                if select.select([sys.stdin], [], [], 0.2)[0]:
                    # Read available characters (might be multiple)
                    chars = []
                    char_times = []
                    read_time = time.time()
                    
                    while True:
                        # Check if more input is immediately available
                        if select.select([sys.stdin], [], [], 0)[0]:
                            char = sys.stdin.read(1)
                            if char:
                                chars.append(char)
                                char_times.append(time.time())
                        else:
                            break
                    
                    # Process characters with automated input detection
                    if chars:
                        # If multiple characters arrived at once, it's likely automated
                        is_bulk_input = len(chars) > 1
                        
                        # Process the most recent character (ignore key repeat/buffering)
                        char = chars[-1]  # Take the last character
                        
                        # Check for automated input patterns
                        if is_bulk_input or self._is_automated_input(char):
                            # Automated input detected - ignore it
                            if is_bulk_input:
                                print(f"🛡️ Ignored bulk input: {repr(''.join(chars))}")
                            elif time.time() - self._startup_time < self._startup_grace_period:
                                # During grace period - silent ignore
                                pass
                            else:
                                print(f"🛡️ Ignored automated pattern containing: {repr(char)}")
                        else:
                            # Regular debouncing for manual input
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


def create_improved_keyboard_handler(debounce_interval: float = 0.2, startup_grace_period: float = 2.0):
    """Create improved keyboard handler with configurable debouncing and startup grace period."""
    handler = ImprovedKeyboardHandler(debounce_interval, startup_grace_period)
    if handler.is_available():
        return handler
    else:
        return NoOpKeyboardHandler()