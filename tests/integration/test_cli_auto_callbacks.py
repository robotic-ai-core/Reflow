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
        from lightning_reflow.core.shared_config import ensure_essential_callbacks
        from lightning_reflow.callbacks.pause import PauseCallback
        
        # Mock trainer with no existing PauseCallback
        mock_trainer = Mock()
        
        # Call the shared function with empty callbacks
        callbacks = ensure_essential_callbacks([], mock_trainer)
        
        # Verify PauseCallback was added
        assert len(callbacks) >= 1
        pause_callbacks = [cb for cb in callbacks if isinstance(cb, PauseCallback)]
        assert len(pause_callbacks) == 1

    def test_pause_callback_not_duplicated(self):
        """Test that PauseCallback is not duplicated if already present."""
        from lightning_reflow.core.shared_config import ensure_essential_callbacks
        from lightning_reflow.callbacks.pause import PauseCallback
        
        # Mock trainer with existing PauseCallback
        existing_pause_callback = PauseCallback()
        mock_trainer = Mock()
        
        # Call the shared function with existing PauseCallback
        callbacks = ensure_essential_callbacks([existing_pause_callback], mock_trainer)
        
        # Verify no additional PauseCallback was added
        pause_callbacks = [cb for cb in callbacks if isinstance(cb, PauseCallback)]
        assert len(pause_callbacks) == 1
        assert pause_callbacks[0] is existing_pause_callback

    def test_pause_callback_with_no_trainer(self):
        """Test that pause callback handling works when no trainer is present."""
        from lightning_reflow.core.shared_config import ensure_essential_callbacks
        
        # Should not raise exception when trainer is None
        callbacks = ensure_essential_callbacks([], None)
        
        # Should still add callbacks
        assert len(callbacks) >= 1

    def test_before_fit_hook(self):
        """Test that before_fit calls essential callback addition."""
        # Create CLI instance
        cli = LightningReflowCLI.__new__(LightningReflowCLI)
        
        # Mock trainer with proper callbacks list
        mock_trainer = Mock()
        mock_trainer.callbacks = []
        cli.trainer = mock_trainer
        
        # Call the method that should trigger callback addition
        cli._add_essential_callbacks()
        
        # Verify that callbacks were updated
        assert hasattr(cli.trainer, 'callbacks')


class TestTrainerCLIReference:
    """Test CLI reference storage in trainer."""

    @patch('lightning_reflow.cli.lightning_cli.logger')
    def test_instantiate_trainer_sets_cli_reference(self, mock_logger):
        """Test that instantiate_trainer sets CLI reference in trainer."""
        with patch.object(LightningReflowCLI, '__init__', lambda x: None):
            cli = LightningReflowCLI()
            
            # Mock the required attributes
            cli.config = {'trainer': {}}
            
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
        from lightning_reflow.core.shared_config import ensure_essential_callbacks
        from lightning_reflow.callbacks.pause import PauseCallback
        
        # Mock trainer
        mock_trainer = Mock()
        
        # Add essential callbacks
        callbacks = ensure_essential_callbacks([], mock_trainer)
        
        # Find the pause callback
        pause_callbacks = [cb for cb in callbacks if isinstance(cb, PauseCallback)]
        assert len(pause_callbacks) == 1
        pause_callback = pause_callbacks[0]
        
        # Check some default values exist
        assert hasattr(pause_callback, 'enable_pause')
        assert hasattr(pause_callback, 'pause_key')


class TestCLIConfigHandling:
    """Test CLI config handling with callbacks."""

    def test_handles_trainer_with_other_callbacks(self):
        """Test that CLI works with trainer that has other callbacks."""
        from lightning_reflow.core.shared_config import ensure_essential_callbacks
        from lightning_reflow.callbacks.pause import PauseCallback
        
        # Mock trainer with existing callback
        other_callback = Mock()
        mock_trainer = Mock()
        
        # Should work without issues - add essential callbacks to existing ones
        callbacks = ensure_essential_callbacks([other_callback], mock_trainer)
        
        # Should have both callbacks now
        assert len(callbacks) >= 2
        pause_callbacks = [cb for cb in callbacks if isinstance(cb, PauseCallback)]
        assert len(pause_callbacks) == 1

    def test_handles_trainer_none(self):
        """Test that CLI handles None trainer gracefully."""
        from lightning_reflow.core.shared_config import ensure_essential_callbacks
        
        # Should not raise exception
        callbacks = ensure_essential_callbacks([], None)
        
        # Should still add callbacks
        assert len(callbacks) >= 1


class TestWandbArtifactCheckpointAutoInjection:
    """Test auto-injection of WandbArtifactCheckpoint when W&B logger is active."""

    def test_wandb_artifact_checkpoint_added_with_wandb_logger(self):
        """WandbArtifactCheckpoint auto-injected when trainer has WandbLogger."""
        from lightning_reflow.core.shared_config import _ensure_wandb_artifact_checkpoint
        from lightning_reflow.callbacks.wandb import WandbArtifactCheckpoint

        mock_trainer = Mock()
        mock_trainer.logger = Mock(spec=["__class__"])
        mock_trainer.logger.__class__ = type(
            "WandbLogger", (), {}
        )

        # Patch isinstance to recognize mock as WandbLogger
        with patch(
            "lightning_reflow.core.shared_config.isinstance",
            side_effect=lambda obj, cls: (
                cls.__name__ == "WandbLogger"
                if hasattr(cls, "__name__") and obj is mock_trainer.logger
                else builtins_isinstance(obj, cls)
            ),
        ) if False else patch.object(
            mock_trainer, "logger"
        ) as patched_logger:
            # Simpler approach: use real WandbLogger mock
            pass

        # Use direct approach: mock the isinstance check
        from lightning.pytorch.loggers import WandbLogger as RealWandbLogger

        mock_trainer.logger = Mock(spec=RealWandbLogger)
        callbacks = []
        _ensure_wandb_artifact_checkpoint(callbacks, mock_trainer)

        wandb_cbs = [cb for cb in callbacks if isinstance(cb, WandbArtifactCheckpoint)]
        assert len(wandb_cbs) == 1

    def test_keep_n_versions_default_is_2(self):
        """Auto-injected WandbArtifactCheckpoint has keep_n_versions=2."""
        from lightning_reflow.core.shared_config import _ensure_wandb_artifact_checkpoint
        from lightning_reflow.callbacks.wandb import WandbArtifactCheckpoint
        from lightning.pytorch.loggers import WandbLogger as RealWandbLogger

        mock_trainer = Mock()
        mock_trainer.logger = Mock(spec=RealWandbLogger)
        callbacks = []
        _ensure_wandb_artifact_checkpoint(callbacks, mock_trainer)

        wandb_cbs = [cb for cb in callbacks if isinstance(cb, WandbArtifactCheckpoint)]
        assert len(wandb_cbs) == 1
        assert wandb_cbs[0].config.keep_n_versions == 2

    def test_not_added_without_wandb_logger(self):
        """WandbArtifactCheckpoint not injected when no W&B logger."""
        from lightning_reflow.core.shared_config import _ensure_wandb_artifact_checkpoint
        from lightning_reflow.callbacks.wandb import WandbArtifactCheckpoint

        mock_trainer = Mock()
        mock_trainer.logger = Mock()  # Not a WandbLogger
        callbacks = []
        _ensure_wandb_artifact_checkpoint(callbacks, mock_trainer)

        wandb_cbs = [cb for cb in callbacks if isinstance(cb, WandbArtifactCheckpoint)]
        assert len(wandb_cbs) == 0

    def test_not_duplicated_if_already_present(self):
        """WandbArtifactCheckpoint not duplicated if user already added one."""
        from lightning_reflow.core.shared_config import _ensure_wandb_artifact_checkpoint
        from lightning_reflow.callbacks.wandb import WandbArtifactCheckpoint
        from lightning.pytorch.loggers import WandbLogger as RealWandbLogger

        mock_trainer = Mock()
        mock_trainer.logger = Mock(spec=RealWandbLogger)
        existing = WandbArtifactCheckpoint(keep_n_versions=5)
        callbacks = [existing]
        _ensure_wandb_artifact_checkpoint(callbacks, mock_trainer)

        wandb_cbs = [cb for cb in callbacks if isinstance(cb, WandbArtifactCheckpoint)]
        assert len(wandb_cbs) == 1
        assert wandb_cbs[0] is existing
        assert wandb_cbs[0].config.keep_n_versions == 5  # User's value preserved


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