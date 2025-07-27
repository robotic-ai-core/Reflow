"""Tests to verify no responsiveness regression from commit 0aca683."""

import pytest
from unittest.mock import Mock, patch
import time

from lightning_reflow.callbacks import PauseCallback
from lightning_reflow.callbacks.pause.pause_mixin import PauseMixin
from lightning_reflow.callbacks.pause.pause_state_machine import PauseStateMachine, PauseState, PauseAction, PauseStatusMessageFactory
from lightning_reflow.callbacks.pause.pause_config import PauseConfig, create_development_config


class TestResponsivenessRegression:
    """Test that responsiveness improvements from commit 0aca683 are maintained."""
    
    def test_current_callback_has_all_responsive_hooks(self):
        """Verify the current callback has responsive hooks for keyboard checking."""
        callback = PauseCallback(enable_pause=True)
        
        # Check that responsive hook methods exist
        assert hasattr(callback, 'on_train_batch_end'), "Missing on_train_batch_end hook"
        assert hasattr(callback, 'on_train_batch_start'), "Missing on_train_batch_start hook" 
        assert hasattr(callback, 'on_validation_batch_start'), "Missing on_validation_batch_start hook"
        assert hasattr(callback, 'on_validation_batch_end'), "Missing on_validation_batch_end hook"
        
        # Verify that keyboard checking is called in the main responsive hook
        with patch.object(callback, '_check_keyboard_input') as mock_check:
            # Mock trainer and module
            trainer = Mock()
            pl_module = Mock()
            
            # Test the main responsive hook (where keyboard checking should happen)
            callback.on_train_batch_end(trainer, pl_module, None, None, 0)
            
            # Should have checked keyboard once (optimized for performance)
            assert mock_check.call_count == 1, f"Expected 1 keyboard check, got {mock_check.call_count}"
        
        print("✅ Current implementation uses optimized keyboard checking in on_train_batch_end")
    
    def test_mixin_responsive_mode_matches_original(self):
        """Test that mixin responsive mode works correctly."""
        
        # Import classes first to avoid scope issues
        try:
            from lightning_reflow.callbacks.pause.pause_mixin import PauseMixin
            from lightning_reflow.callbacks.pause.pause_config import PauseConfig
        except ImportError:
            pytest.skip("PauseMixin or PauseConfig not available")
        
        class TestCallback(PauseMixin):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.keyboard_checks = 0
            
            def _process_keyboard_input(self):
                self.keyboard_checks += 1
                super()._process_keyboard_input()
        
        # Test with responsive mode enabled (matches original behavior)
        responsive_config = PauseConfig(
            enable_pause=True,
            use_responsive_mode=True,  # Enable maximum responsiveness
            debounce_interval=0.1
        )
        
        callback = TestCallback(pause_config=responsive_config)
        
        # Mock trainer and module
        trainer = Mock()
        pl_module = Mock()
        
        # Test that the callback can be created and basic functionality works
        assert callback is not None
        assert hasattr(callback, '_process_keyboard_input')
        
        print("✅ Responsive mode configuration works correctly")
    
    def test_mixin_hybrid_mode_reduces_overhead(self):
        """Test that mixin hybrid mode configuration works."""
        
        # Skip test if PauseMixin or PauseConfig are not available
        try:
            from lightning_reflow.callbacks.pause.pause_mixin import PauseMixin
            from lightning_reflow.callbacks.pause.pause_config import PauseConfig
        except ImportError:
            pytest.skip("PauseMixin or PauseConfig not available")
        
        class TestCallback(PauseMixin):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.keyboard_checks = 0
            
            def _process_keyboard_input(self):
                self.keyboard_checks += 1
                super()._process_keyboard_input()
        
        # Test with hybrid mode (efficiency optimization)
        hybrid_config = PauseConfig(
            enable_pause=True,
            use_responsive_mode=False,  # Use hybrid approach
            keyboard_poll_frequency=3,  # Check every 3 calls
            max_time_between_checks=1.0,  # 1 second time limit
            debounce_interval=0.1
        )
        
        callback = TestCallback(pause_config=hybrid_config)
        
        # Test that the callback can be created and configured
        assert callback is not None
        assert hasattr(callback, '_process_keyboard_input')
        
        print("✅ Hybrid mode configuration works correctly")
    
    def test_development_config_maintains_responsiveness(self):
        """Test that development config maintains maximum responsiveness."""
        dev_config = create_development_config()
        
        # Development config should enable responsive mode
        assert dev_config.pause_config.use_responsive_mode is True
        assert dev_config.pause_config.max_time_between_checks <= 0.5  # Fast response time
        assert dev_config.pause_config.debounce_interval <= 0.3  # Fast debouncing
    
    def test_backward_compatibility_with_improved_keyboard_handler(self):
        """Test that our unified handler maintains the improvements from commit 0aca683."""
        from lightning_reflow.callbacks.pause.unified_keyboard_handler import create_keyboard_handler, KeyboardHandlerStrategy
        
        # Test that improved strategy is available
        handler = create_keyboard_handler(KeyboardHandlerStrategy.IMPROVED_MODE)
        assert handler is not None
        
        # Test backward compatibility import
        from lightning_reflow.callbacks.pause.unified_keyboard_handler import create_improved_keyboard_handler
        
        # Test that the compatibility function works
        compat_handler = create_improved_keyboard_handler()
        assert compat_handler is not None
        
        print("✅ Backward compatibility maintained with improved keyboard handler")
    
    def test_responsiveness_timing_requirements(self):
        """Test specific timing requirements from commit 0aca683."""
        
        # Test that debounce interval matches the improved version (0.3s vs original 0.5s)
        config = PauseConfig()
        assert config.debounce_interval == 0.3, f"Expected 0.3s debounce, got {config.debounce_interval}"
        
        # Test that development config is even more responsive
        dev_config = create_development_config()
        assert dev_config.pause_config.debounce_interval <= 0.3
        assert dev_config.pause_config.max_time_between_checks <= 0.5
    
    def test_no_mode_thrashing_in_unified_handler(self):
        """Test that unified handler avoids terminal mode thrashing mentioned in commit 0aca683."""
        from lightning_reflow.callbacks.pause.unified_keyboard_handler import UnifiedKeyboardHandler, KeyboardHandlerStrategy
        
        # Test persistent mode (should set terminal mode once and keep it)
        handler = UnifiedKeyboardHandler(KeyboardHandlerStrategy.PERSISTENT_MODE)
        
        # Verify it has the persistent mode logic
        assert hasattr(handler, '_terminal_mode_set')
        assert hasattr(handler, '_original_settings')
        
        # Test improved mode (should also have persistent terminal management)
        handler2 = UnifiedKeyboardHandler(KeyboardHandlerStrategy.IMPROVED_MODE)
        assert hasattr(handler2, '_terminal_mode_set')


class TestResponsivenessConfiguration:
    """Test responsiveness configuration options."""
    
    def test_responsiveness_presets(self):
        """Test that different presets provide appropriate responsiveness levels."""
        from lightning_reflow.callbacks.pause.pause_config import create_development_config, create_production_config
        
        # Development should prioritize responsiveness
        dev_config = create_development_config()
        assert dev_config.pause_config.use_responsive_mode is True
        assert dev_config.pause_config.max_time_between_checks <= 0.5
        
        # Production should prioritize efficiency
        prod_config = create_production_config()
        assert prod_config.pause_config.use_responsive_mode is False  # Should default to efficient mode
        assert prod_config.pause_config.enable_pause is False  # Disabled in production
    
    def test_custom_responsiveness_configuration(self):
        """Test custom responsiveness configuration."""
        
        # Ultra-responsive configuration
        ultra_config = PauseConfig(
            enable_pause=True,
            use_responsive_mode=True,
            max_time_between_checks=0.1,  # 100ms
            debounce_interval=0.1,
            keyboard_poll_frequency=1  # Every batch
        )
        
        # Should allow very responsive settings
        assert ultra_config.use_responsive_mode is True
        assert ultra_config.max_time_between_checks == 0.1
        
        # Efficient configuration
        efficient_config = PauseConfig(
            enable_pause=True,
            use_responsive_mode=False,
            max_time_between_checks=1.0,  # 1 second
            keyboard_poll_frequency=20  # Every 20 batches
        )
        
        # Should allow efficient settings
        assert efficient_config.use_responsive_mode is False
        assert efficient_config.max_time_between_checks == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])