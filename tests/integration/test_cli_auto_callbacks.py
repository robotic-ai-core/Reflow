#!/usr/bin/env python3
"""
Test CLI auto-callback functionality.

Tests that the Lightning Reflow CLI properly adds callbacks automatically
and maintains compatibility with existing configuration systems.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lightning_reflow.cli.lightning_cli import LightningReflowCLI


class TestAutoCallbackAddition:
    """Test automatic callback addition functionality."""

    def test_pause_callback_added_automatically(self):
        """Test that PauseCallback is added automatically when not present."""
        # Create CLI instance
        cli = LightningReflowCLI.__new__(LightningReflowCLI)  # Don't call __init__
        
        # Mock trainer with no existing PauseCallback
        mock_trainer = Mock()
        mock_trainer.callbacks = []
        cli.trainer = mock_trainer
        
        # Call the method
        cli._add_pause_callback()
        
        # Verify PauseCallback was added
        assert len(cli.trainer.callbacks) == 1
        from lightning_reflow.callbacks.pause import PauseCallback
        assert isinstance(cli.trainer.callbacks[0], PauseCallback)

    def test_pause_callback_not_duplicated(self):
        """Test that PauseCallback is not duplicated if already present."""
        # Create CLI instance
        cli = LightningReflowCLI.__new__(LightningReflowCLI)
        
        # Mock trainer with existing PauseCallback
        from lightning_reflow.callbacks.pause import PauseCallback
        existing_pause_callback = PauseCallback()
        
        mock_trainer = Mock()
        mock_trainer.callbacks = [existing_pause_callback]
        cli.trainer = mock_trainer
        
        # Call the method
        cli._add_pause_callback()
        
        # Verify no additional PauseCallback was added
        assert len(cli.trainer.callbacks) == 1
        assert cli.trainer.callbacks[0] is existing_pause_callback

    def test_pause_callback_with_no_trainer(self):
        """Test that pause callback handling works when no trainer is present."""
        # Create CLI instance with no trainer
        cli = LightningReflowCLI.__new__(LightningReflowCLI)
        cli.trainer = None
        
        # Should not raise exception
        cli._add_pause_callback()

    def test_before_fit_hook(self):
        """Test that before_fit calls _add_pause_callback."""
        # Create CLI instance
        cli = LightningReflowCLI.__new__(LightningReflowCLI)
        
        # Mock the _add_pause_callback method
        cli._add_pause_callback = Mock()
        cli._add_step_output_logger = Mock()
        cli.trainer = Mock()
        
        # Call the method that should trigger callback addition
        cli._add_essential_callbacks()
        
        # Verify _add_pause_callback was called
        cli._add_pause_callback.assert_called_once()


class TestTrainerCLIReference:
    """Test CLI reference storage in trainer."""

    @patch('lightning_reflow.cli.lightning_cli.logger')
    def test_instantiate_trainer_sets_cli_reference(self, mock_logger):
        """Test that instantiate_trainer sets CLI reference in trainer."""
        with patch.object(LightningReflowCLI, '__init__', lambda x: None):
            cli = LightningReflowCLI()
            
            # Mock trainer
            mock_trainer = Mock()
            
            # Mock the parent class method
            with patch('lightning.pytorch.cli.LightningCLI.instantiate_trainer') as mock_super:
                mock_super.return_value = mock_trainer
                
                # Mock the state manager registration
                with patch.object(cli, '_register_trainer_config_state'):
                    # Call the method
                    result = cli.instantiate_trainer(max_epochs=10)
                    
                    # Verify parent method was called with kwargs
                    mock_super.assert_called_once_with(max_epochs=10)
                    
                    # Verify cli reference was set
                    assert mock_trainer.cli == cli
                    
                    # Verify the first log message (about CLI reference)
                    mock_logger.info.assert_any_call(
                        "✅ Stored CLI reference in trainer for checkpoint compatibility"
                    )
                    
                    # Verify trainer is returned
                    assert result == mock_trainer

    def test_config_embedding_mixin_can_embed_config(self):
        """Test that ConfigEmbeddingMixin can check CLI context."""
        from lightning_reflow.callbacks.core.config_embedding_mixin import ConfigEmbeddingMixin
        
        # Create mixin instance
        mixin = ConfigEmbeddingMixin()
        
        # Mock trainer with CLI reference
        mock_trainer = Mock()
        mock_trainer.cli = Mock()
        mock_trainer.cli.save_config_kwargs = True
        
        # Should be able to embed config
        assert mixin._can_embed_config(mock_trainer) is True

    def test_config_embedding_mixin_validation_fails_without_cli(self):
        """Test that ConfigEmbeddingMixin fails gracefully without CLI."""
        from lightning_reflow.callbacks.core.config_embedding_mixin import ConfigEmbeddingMixin
        
        # Create mixin instance
        mixin = ConfigEmbeddingMixin()
        
        # Mock trainer without CLI reference
        mock_trainer = Mock()
        mock_trainer.cli = None
        
        # Should not be able to embed config
        assert mixin._can_embed_config(mock_trainer) is False


class TestPauseCallbackDefaultConfig:
    """Test PauseCallback default configuration."""

    def test_pause_callback_default_values(self):
        """Test that PauseCallback is created with proper default values."""
        # Create CLI instance
        cli = LightningReflowCLI.__new__(LightningReflowCLI)
        
        # Mock trainer
        mock_trainer = Mock()
        mock_trainer.callbacks = []
        cli.trainer = mock_trainer
        
        # Add pause callback
        cli._add_pause_callback()
        
        # Verify callback was added with default config
        assert len(cli.trainer.callbacks) == 1
        pause_callback = cli.trainer.callbacks[0]
        
        # Check some default values exist
        assert hasattr(pause_callback, 'enable_pause')
        assert hasattr(pause_callback, 'pause_key')


class TestCLIConfigHandling:
    """Test CLI config handling with callbacks."""

    def test_handles_trainer_with_other_callbacks(self):
        """Test that CLI works with trainer that has other callbacks."""
        # Create CLI instance
        cli = LightningReflowCLI.__new__(LightningReflowCLI)
        
        # Mock trainer with existing callback
        other_callback = Mock()
        mock_trainer = Mock()
        mock_trainer.callbacks = [other_callback]
        cli.trainer = mock_trainer
        
        # Should work without issues
        cli._add_pause_callback()
        
        # Should have both callbacks now
        assert len(cli.trainer.callbacks) == 2

    def test_handles_trainer_none(self):
        """Test that CLI handles None trainer gracefully."""
        # Create CLI instance
        cli = LightningReflowCLI.__new__(LightningReflowCLI)
        cli.trainer = None
        
        # Should not raise exception
        cli._add_pause_callback()


class TestEndToEndCLI:
    """Test end-to-end CLI functionality."""

    def test_cli_before_fit_flow(self):
        """Test that CLI properly calls essential callbacks during fit flow."""
        with patch.object(LightningReflowCLI, '__init__', lambda x: None):
            cli = LightningReflowCLI()
            
            # Mock trainer
            mock_trainer = Mock()
            mock_trainer.callbacks = []
            cli.trainer = mock_trainer
            
            # Call essential callbacks method
            cli._add_essential_callbacks()
            
            # Should have added callbacks
            assert len(cli.trainer.callbacks) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 