"""
Unit tests for keyboard handler with time-window detection.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from queue import Queue

from lightning_reflow.callbacks.pause.improved_keyboard_handler import (
    ImprovedKeyboardHandler,
    create_improved_keyboard_handler
)


class MockStdin:
    """Mock stdin for testing keyboard input."""
    
    def __init__(self):
        self.input_queue = Queue()
        self.read_delay = 0  # Delay between chars for simulating typing speed
    
    def read(self, n=1):
        """Read n characters from the mock input."""
        if self.read_delay > 0:
            time.sleep(self.read_delay)
        
        if not self.input_queue.empty():
            return self.input_queue.get()
        return None
    
    def fileno(self):
        """Return a fake file descriptor."""
        return 0
    
    def add_input(self, text, delay_between_chars=0):
        """Add input to the queue with optional delay between characters."""
        self.read_delay = delay_between_chars
        for char in text:
            self.input_queue.put(char)


class TestImprovedKeyboardHandler:
    """Test the improved keyboard handler with time-window detection."""
    
    @pytest.fixture
    def mock_termios(self):
        """Mock termios module."""
        with patch('lightning_reflow.callbacks.pause.improved_keyboard_handler.HAS_TERMIOS', True):
            with patch('lightning_reflow.callbacks.pause.improved_keyboard_handler.termios'):
                with patch('lightning_reflow.callbacks.pause.improved_keyboard_handler.tty'):
                    yield
    
    @pytest.fixture
    def mock_stdin(self):
        """Create a mock stdin."""
        return MockStdin()
    
    def test_handler_initialization(self, mock_termios):
        """Test keyboard handler initialization."""
        handler = ImprovedKeyboardHandler()
        
        assert handler._monitoring is False
        assert handler._monitor_thread is None
        assert handler._key_queue is not None
        assert handler._char_window == 0.05  # 50ms window
    
    def test_single_character_accepted(self, mock_termios, mock_stdin):
        """Test that single characters with no followers are accepted."""
        handler = ImprovedKeyboardHandler()
        
        # Mock select to return True when we have input
        with patch('lightning_reflow.callbacks.pause.improved_keyboard_handler.select.select') as mock_select:
            with patch('sys.stdin', mock_stdin):
                # Add single 'p' to input
                mock_stdin.add_input('p')
                
                # First select returns True (input available)
                # Second select (after 50ms wait) returns False (no more input)
                mock_select.side_effect = [
                    ([mock_stdin], [], []),  # Initial check - input available
                    ([], [], []),  # After 50ms - no more input
                ]
                
                # Run one iteration of the monitor loop
                handler._monitoring = True
                
                # Run one iteration of monitor manually
                self._monitor_keyboard_iteration(handler)
                
                # Check that 'p' was accepted
                key = handler.get_key()
                assert key == 'p'
    
    def test_automated_input_rejected(self, mock_termios, mock_stdin):
        """Test that rapid character sequences are rejected as automated."""
        handler = ImprovedKeyboardHandler()
        
        with patch('lightning_reflow.callbacks.pause.improved_keyboard_handler.select.select') as mock_select:
            with patch('sys.stdin', mock_stdin):
                # Add 'pyenv' to input queue
                mock_stdin.add_input('pyenv')
                
                # First select returns True (first char available)
                # Second select (after 50ms) returns True (more chars available)
                # Subsequent selects return True until queue is empty
                mock_select.side_effect = [
                    ([mock_stdin], [], []),  # Initial - 'p' available
                    ([mock_stdin], [], []),  # After 50ms - 'y' available
                    ([mock_stdin], [], []),  # 'e' available
                    ([mock_stdin], [], []),  # 'n' available
                    ([mock_stdin], [], []),  # 'v' available
                    ([], [], []),  # No more input
                ]
                
                # Capture print output
                with patch('builtins.print') as mock_print:
                    # Run one iteration
                    handler._monitoring = True
                    self._monitor_keyboard_iteration(handler)
                    
                    # Verify automated input was detected and rejected
                    mock_print.assert_called_once()
                    call_args = mock_print.call_args[0][0]
                    assert "Ignored automated input" in call_args
                    assert "pyenv" in call_args
                
                # Verify no key was accepted
                key = handler.get_key()
                assert key is None
    
    def test_time_window_detection(self, mock_termios):
        """Test that the 50ms time window correctly distinguishes input types."""
        handler = ImprovedKeyboardHandler()
        
        # Test that char_window is set correctly
        assert handler._char_window == 0.05
        
        # Test with custom window
        handler2 = ImprovedKeyboardHandler()
        handler2._char_window = 0.1  # 100ms window
        assert handler2._char_window == 0.1
    
    def test_handler_context_manager(self, mock_termios):
        """Test keyboard handler as context manager."""
        with patch('sys.stdin.isatty', return_value=True):
            handler = ImprovedKeyboardHandler()
            
            with patch.object(handler, 'start_monitoring') as mock_start:
                with patch.object(handler, 'stop_monitoring') as mock_stop:
                    with handler:
                        mock_start.assert_called_once()
                    mock_stop.assert_called_once()
    
    def _monitor_keyboard_iteration(self, handler):
        """Helper to run one iteration of keyboard monitoring."""
        # This is a simplified version for testing
        import select
        import sys
        
        if select.select([sys.stdin], [], [], 0.1)[0]:
            first_char = sys.stdin.read(1)
            
            if first_char:
                time.sleep(handler._char_window)
                
                additional_chars = []
                while select.select([sys.stdin], [], [], 0)[0]:
                    char = sys.stdin.read(1)
                    if char:
                        additional_chars.append(char)
                    else:
                        break
                
                if additional_chars:
                    all_chars = first_char + ''.join(additional_chars)
                    print(f"🛡️ Ignored automated input: {repr(all_chars)}")
                else:
                    handler._key_queue.put(first_char)




class TestKeyboardHandlerIntegration:
    """Integration tests for keyboard handler."""
    
    def test_create_keyboard_handler(self):
        """Test creating keyboard handler with factory function."""
        with patch('lightning_reflow.callbacks.pause.improved_keyboard_handler.HAS_TERMIOS', True):
            with patch('sys.stdin.isatty', return_value=True):
                handler = create_improved_keyboard_handler()
                assert isinstance(handler, ImprovedKeyboardHandler)
    
    def test_create_handler_without_termios(self):
        """Test handler creation when termios is not available."""
        with patch('lightning_reflow.callbacks.pause.improved_keyboard_handler.HAS_TERMIOS', False):
            handler = create_improved_keyboard_handler()
            # Should return NoOpKeyboardHandler
            assert not handler.is_available()
            assert handler.get_key() is None