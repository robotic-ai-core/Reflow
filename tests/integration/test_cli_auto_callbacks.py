"""
Test automatic callback addition and trainer CLI reference in LightningReflowCLI.

These tests verify that:
1. PauseCallback is automatically added to trainer callbacks
2. The trainer.cli reference is properly set for ConfigEmbeddingMixin
3. The CLI correctly handles various callback configurations
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, PropertyMock
import lightning.pytorch as pl

from lightning_reflow.cli.lightning_cli import LightningReflowCLI
from lightning_reflow.callbacks.pause import PauseCallback


class TestAutoCallbackAddition:
    """Test automatic PauseCallback addition functionality."""

    def test_pause_callback_added_automatically(self):
        """Test that PauseCallback is automatically added when not present."""
        # Setup mock CLI
        cli = LightningReflowCLI.__new__(LightningReflowCLI)
        
        # Create mock trainer with empty callbacks
        mock_trainer = Mock(spec=pl.Trainer)
        mock_trainer.callbacks = []
        cli.trainer = mock_trainer
        
        # Mock logger
        with patch('lightning_reflow.cli.lightning_cli.logger') as mock_logger:
            # Call the method
            cli._ensure_pause_callback()
            
            # Verify PauseCallback was added
            assert len(mock_trainer.callbacks) == 1
            assert isinstance(mock_trainer.callbacks[0], PauseCallback)
            
            # Verify logging
            mock_logger.info.assert_called_with(
                "✅ Automatically added PauseCallback for progress bar functionality"
            )

    def test_pause_callback_not_duplicated(self):
        """Test that PauseCallback is not added if already present."""
        # Setup mock CLI
        cli = LightningReflowCLI.__new__(LightningReflowCLI)
        
        # Create mock trainer with existing PauseCallback
        mock_trainer = Mock(spec=pl.Trainer)
        existing_pause_callback = PauseCallback()
        mock_trainer.callbacks = [existing_pause_callback]
        cli.trainer = mock_trainer
        
        # Mock logger
        with patch('lightning_reflow.cli.lightning_cli.logger') as mock_logger:
            # Call the method
            cli._ensure_pause_callback()
            
            # Verify no additional callback was added
            assert len(mock_trainer.callbacks) == 1
            assert mock_trainer.callbacks[0] is existing_pause_callback
            
            # Verify no logging about adding callback
            assert not any(
                'Automatically added PauseCallback' in str(call) 
                for call in mock_logger.info.call_args_list
            )

    def test_pause_callback_with_no_trainer(self):
        """Test that method returns early when trainer doesn't exist."""
        # Setup mock CLI without trainer
        cli = LightningReflowCLI.__new__(LightningReflowCLI)
        # No trainer attribute
        
        # Mock logger
        with patch('lightning_reflow.cli.lightning_cli.logger') as mock_logger:
            # Call the method - should return early
            cli._ensure_pause_callback()
            
            # Verify no logging occurred (method returned early)
            mock_logger.info.assert_not_called()

    def test_before_fit_hook(self):
        """Test that before_fit calls _ensure_pause_callback."""
        with patch('lightning_reflow.cli.lightning_cli.LightningCLI.__init__', return_value=None):
            cli = LightningReflowCLI.__new__(LightningReflowCLI)
            
            # Mock the methods
            cli._ensure_pause_callback = Mock()
            
            # Call the hook
            cli.before_fit()
            
            # Verify _ensure_pause_callback was called
            cli._ensure_pause_callback.assert_called_once()


class TestTrainerCLIReference:
    """Test that trainer.cli reference is properly set."""

    def test_instantiate_trainer_sets_cli_reference(self):
        """Test that instantiate_trainer sets trainer.cli reference."""
        with patch('lightning_reflow.cli.lightning_cli.LightningCLI.__init__', return_value=None):
            cli = LightningReflowCLI.__new__(LightningReflowCLI)
            
            # Mock parent's instantiate_trainer
            mock_trainer = Mock(spec=pl.Trainer)
            with patch('lightning_reflow.cli.lightning_cli.super') as mock_super:
                mock_super_instance = Mock()
                mock_super.return_value = mock_super_instance
                mock_super_instance.instantiate_trainer.return_value = mock_trainer
                
                # Mock logger
                with patch('lightning_reflow.cli.lightning_cli.logger') as mock_logger:
                    # Call the method
                    result = cli.instantiate_trainer(max_epochs=10)
                    
                    # Verify parent method was called with kwargs
                    mock_super_instance.instantiate_trainer.assert_called_once_with(max_epochs=10)
                    
                    # Verify cli reference was set
                    assert mock_trainer.cli == cli
                    
                    # Verify logging
                    mock_logger.info.assert_called_with(
                        "✅ Stored CLI reference in trainer for checkpoint compatibility"
                    )
                    
                    # Verify trainer is returned
                    assert result == mock_trainer

    def test_config_embedding_mixin_validation_passes(self):
        """Test that ConfigEmbeddingMixin validation passes with proper CLI setup."""
        from lightning_reflow.callbacks.core.config_embedding_mixin import ConfigEmbeddingMixin
        
        # Create a mock trainer with CLI reference
        mock_trainer = Mock(spec=pl.Trainer)
        mock_cli = Mock()
        mock_cli.save_config_kwargs = None  # Default Lightning behavior
        mock_trainer.cli = mock_cli
        
        # Create mixin instance
        mixin = ConfigEmbeddingMixin()
        
        # This should not raise an error
        mixin._validate_cli_configuration(mock_trainer)
        
        # Verify validation flag is set
        assert mixin._cli_config_validated is True

    def test_config_embedding_mixin_validation_fails_without_cli(self):
        """Test that ConfigEmbeddingMixin validation fails without CLI reference."""
        from lightning_reflow.callbacks.core.config_embedding_mixin import ConfigEmbeddingMixin
        
        # Create a mock trainer without CLI reference
        mock_trainer = Mock(spec=pl.Trainer)
        mock_trainer.cli = None
        
        # Create mixin instance
        mixin = ConfigEmbeddingMixin()
        
        # This should raise a RuntimeError
        with pytest.raises(RuntimeError, match="ConfigEmbeddingMixin requires LightningReflowCLI context"):
            mixin._validate_cli_configuration(mock_trainer)


class TestPauseCallbackDefaultConfig:
    """Test the default configuration values for automatically added PauseCallback."""

    def test_pause_callback_default_values(self):
        """Test that PauseCallback is added with correct default values."""
        # Setup mock CLI
        cli = LightningReflowCLI.__new__(LightningReflowCLI)
        
        # Create mock trainer with empty callbacks
        mock_trainer = Mock(spec=pl.Trainer)
        mock_trainer.callbacks = []
        cli.trainer = mock_trainer
        
        # Call the method
        cli._ensure_pause_callback()
        
        # Get the added callback
        added_callback = mock_trainer.callbacks[0]
        
        # Verify it's a PauseCallback
        assert isinstance(added_callback, PauseCallback)
        
        # Verify default values - only check essential public attributes
        assert str(added_callback.checkpoint_dir) == 'pause_checkpoints'
        assert added_callback.enable_pause is True
        assert added_callback.pause_key == 'p'
        assert added_callback.upload_key == 'w'
        # Note: Other attributes like bar_colour, metrics, etc. are internal
        # implementation details and may have underscores or different names


class TestCLIConfigHandling:
    """Test various configuration scenarios."""

    def test_handles_trainer_with_other_callbacks(self):
        """Test that PauseCallback is added alongside existing callbacks."""
        cli = LightningReflowCLI.__new__(LightningReflowCLI)
        
        # Create mock trainer with other callbacks
        mock_trainer = Mock(spec=pl.Trainer)
        other_callback = Mock()
        other_callback.__class__.__name__ = 'SomeOtherCallback'
        mock_trainer.callbacks = [other_callback]
        cli.trainer = mock_trainer
        
        # Call the method
        cli._ensure_pause_callback()
        
        # Should have both callbacks
        assert len(mock_trainer.callbacks) == 2
        assert mock_trainer.callbacks[0] is other_callback
        assert isinstance(mock_trainer.callbacks[1], PauseCallback)

    def test_handles_trainer_none(self):
        """Test handling when trainer attribute is None."""
        cli = LightningReflowCLI.__new__(LightningReflowCLI)
        cli.trainer = None
        
        # Mock logger
        with patch('lightning_reflow.cli.lightning_cli.logger') as mock_logger:
            # Call the method - should return early
            cli._ensure_pause_callback()
            
            # Verify no logging occurred
            mock_logger.info.assert_not_called()


@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary config file."""
    config = {
        'trainer': {
            'max_epochs': 10,
            'callbacks': []
        },
        'model': {
            'class_path': 'some.model.Class'
        }
    }
    
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return config_path


class TestEndToEndCLI:
    """End-to-end tests with real CLI instantiation."""

    def test_cli_before_fit_flow(self):
        """Test CLI before_fit flow with automatic callback addition."""
        # Create CLI instance
        cli = LightningReflowCLI.__new__(LightningReflowCLI)
        
        # Create mock trainer
        mock_trainer = Mock(spec=pl.Trainer)
        mock_trainer.callbacks = []
        cli.trainer = mock_trainer
        
        # Call before_fit
        cli.before_fit()
        
        # Verify both callbacks were added
        assert len(mock_trainer.callbacks) == 2
        # Check that we have both callback types (order may vary)
        callback_types = {type(cb) for cb in mock_trainer.callbacks}
        assert PauseCallback in callback_types
        from lightning_reflow.callbacks.logging import StepOutputLoggerCallback
        assert StepOutputLoggerCallback in callback_types 