"""
Test suite for LightningReflow progress bar management.

This test suite verifies that LightningReflow properly manages progress bars
to prevent conflicts between Lightning's default progress bar and the custom
FlowProgressBarCallback system.
"""

import pytest
import pytorch_lightning as pl
from lightning_reflow.core import LightningReflow


class TestProgressBarManagement:
    """Test progress bar conflict prevention in LightningReflow."""
    
    def test_default_progress_bar_disabled_automatically(self):
        """
        Test that LightningReflow automatically disables Lightning's default progress bar.
        
        This prevents UI conflicts between Lightning's default TQDMProgressBar and 
        LightningReflow's custom FlowProgressBarCallback system.
        """
        # Create LightningReflow instance with no explicit progress bar config
        reflow = LightningReflow()
        
        # Create trainer using LightningReflow's trainer creation method
        trainer = reflow._create_trainer()
        
        # Verify Lightning's default progress bar is disabled
        assert trainer.progress_bar_callback is None, (
            "Lightning's default progress bar should be disabled to prevent "
            "conflicts with FlowProgressBarCallback"
        )
    
    def test_user_can_override_progress_bar_setting(self):
        """
        Test that users can still override the progress bar setting if desired.
        
        Some users might want to use Lightning's default progress bar instead
        of the custom FlowProgressBarCallback.
        """
        # User explicitly enables Lightning's progress bar
        reflow = LightningReflow(
            trainer_defaults={"enable_progress_bar": True}
        )
        
        trainer = reflow._create_trainer()
        
        # Verify user's setting is respected
        assert trainer.progress_bar_callback is not None, (
            "User's explicit enable_progress_bar=True should be respected"
        )
    
    def test_config_file_can_override_progress_bar_setting(self):
        """
        Test that config files can override the default progress bar setting.
        """
        import tempfile
        import yaml
        from pathlib import Path
        
        # Create a temporary config file that enables progress bar
        config_content = {
            'trainer': {
                'enable_progress_bar': True,
                'max_epochs': 1
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_content, f)
            config_path = f.name
        
        try:
            # Create LightningReflow with config file
            reflow = LightningReflow(config_files=config_path)
            trainer = reflow._create_trainer()
            
            # Verify config file setting is respected
            assert trainer.progress_bar_callback is not None, (
                "Config file enable_progress_bar=true should override default"
            )
            
        finally:
            # Clean up temporary file
            Path(config_path).unlink()
    
    def test_flow_progress_bar_callback_auto_added(self):
        """
        Test that FlowProgressBarCallback (via PauseCallback) is automatically added.
        
        This ensures users get the enhanced progress bar functionality even
        when Lightning's default is disabled.
        """
        reflow = LightningReflow()
        trainer = reflow._create_trainer()
        
        # Check for PauseCallback (which inherits from FlowProgressBarCallback)
        from lightning_reflow.callbacks.pause import PauseCallback
        
        pause_callbacks = [
            cb for cb in trainer.callbacks 
            if isinstance(cb, PauseCallback)
        ]
        
        assert len(pause_callbacks) > 0, (
            "PauseCallback (FlowProgressBarCallback) should be automatically added "
            "to provide enhanced progress bar functionality"
        )
    
    def test_no_double_progress_bars_by_default(self):
        """
        Test that the default configuration prevents double progress bars.
        
        This is the critical test that ensures users don't see both Lightning's
        progress bar and LightningReflow's custom progress bars simultaneously.
        """
        reflow = LightningReflow()
        trainer = reflow._create_trainer()
        
        # Lightning's default progress bar should be disabled
        assert trainer.progress_bar_callback is None, (
            "Lightning's progress bar should be disabled"
        )
        
        # LightningReflow's custom progress bar should be present
        from lightning_reflow.callbacks.pause import PauseCallback
        has_custom_progress_bar = any(
            isinstance(cb, PauseCallback) for cb in trainer.callbacks
        )
        
        assert has_custom_progress_bar, (
            "LightningReflow's custom progress bar should be present"
        )
        
        print("✅ No double progress bars: Lightning disabled, FlowProgressBar enabled") 