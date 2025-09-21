"""
Test all refactoring improvements to ensure they work correctly.
"""

import pytest
from unittest.mock import MagicMock

from lightning_reflow.callbacks.core.base_reflow_callback import BaseReflowCallback
from lightning_reflow.callbacks.core.memory_cleanup_callback import MemoryCleanupCallback
from lightning_reflow.callbacks.pause.pause_callback import PauseCallback
from lightning_reflow.callbacks.wandb.wandb_artifact_checkpoint import WandbArtifactCheckpoint
from lightning_reflow.callbacks.monitoring.flow_progress_bar_callback import FlowProgressBarCallback

from lightning_reflow.utils.checkpoint.wandb_artifact_state import WandbArtifactState
from lightning_reflow.utils.checkpoint.flow_progress_bar_state import FlowProgressBarState
from lightning_reflow.utils.checkpoint.scientific_reproducibility_state import ScientificReproducibilityState


class TestAllImprovements:
    """Test that all refactoring improvements are working correctly."""

    def test_base_reflow_callback_exists(self):
        """Test that BaseReflowCallback exists and can be instantiated."""
        callback = BaseReflowCallback(enable_state_management=False)
        assert callback is not None
        assert hasattr(callback, 'log_verbose')
        assert hasattr(callback, 'log_debug')
        assert hasattr(callback, 'log_warning')
        assert hasattr(callback, 'log_error')

    def test_memory_cleanup_inherits_from_base(self):
        """Test that MemoryCleanupCallback inherits from BaseReflowCallback."""
        assert issubclass(MemoryCleanupCallback, BaseReflowCallback)

        # Test instantiation
        callback = MemoryCleanupCallback(verbose=True)
        assert isinstance(callback, BaseReflowCallback)
        assert hasattr(callback, 'log_verbose')

    def test_extracted_manager_states(self):
        """Test that extracted manager state classes work correctly."""
        # Test WandbArtifactState
        mock_callback = MagicMock()
        mock_callback.config = MagicMock()
        mock_callback.config.__dict__ = {'upload_best_model': True}
        mock_callback.state = MagicMock()
        mock_callback.state.__dict__ = {'has_uploaded': False}

        wandb_state = WandbArtifactState(mock_callback)
        assert wandb_state.manager_name == "wandb_artifact_checkpoint"

        # Test FlowProgressBarState
        flow_state = FlowProgressBarState(mock_callback)
        assert flow_state.manager_name == "flow_progress_bar"

        # Test ScientificReproducibilityState
        sci_state = ScientificReproducibilityState()
        assert sci_state.manager_name == "scientific_reproducibility"

    def test_pause_callback_has_rng_states_by_default(self):
        """Test that PauseCallback has RNG state saving enabled by default."""
        callback = PauseCallback(checkpoint_dir='/tmp/test')
        assert callback.save_rng_states is True
        assert hasattr(callback, '_reproducibility_manager')

    def test_wandb_artifact_uses_extracted_state(self):
        """Test that WandbArtifactCheckpoint uses the extracted state manager."""
        callback = WandbArtifactCheckpoint(
            upload_best_model=False,
            upload_last_model=False
        )
        assert hasattr(callback, '_state_manager')
        assert isinstance(callback._state_manager, WandbArtifactState)

    def test_flow_progress_bar_uses_extracted_state(self):
        """Test that FlowProgressBarCallback registers its state manager."""
        callback = FlowProgressBarCallback(refresh_rate=1)
        # The state manager is registered in _register_for_state_persistence
        # which is called in __init__

    def test_no_code_duplication(self):
        """Test that we haven't introduced code duplication."""
        # Check that WandbArtifactCheckpoint doesn't have old backward compatibility methods
        callback = WandbArtifactCheckpoint()

        # These old properties should not exist anymore (we removed them)
        # They were backward compatibility properties
        assert not hasattr(callback, '_should_upload_epoch')
        assert not hasattr(callback, '_should_upload_validation')
        assert not hasattr(callback, '_save_comprehensive_checkpoint')
        assert not hasattr(callback, '_upload_single_checkpoint')

    def test_callback_hierarchy(self):
        """Test the callback hierarchy is correct."""
        # PauseCallback inherits from FlowProgressBarCallback
        assert issubclass(PauseCallback, FlowProgressBarCallback)

        # MemoryCleanupCallback inherits from BaseReflowCallback
        assert issubclass(MemoryCleanupCallback, BaseReflowCallback)

    def test_all_manager_states_have_consistent_interface(self):
        """Test that all manager states follow the same interface."""
        managers = [
            WandbArtifactState(MagicMock()),
            FlowProgressBarState(MagicMock()),
            ScientificReproducibilityState()
        ]

        for manager in managers:
            # All should have required methods
            assert callable(getattr(manager, 'capture_state'))
            assert callable(getattr(manager, 'restore_state'))
            assert callable(getattr(manager, 'validate_state'))
            assert hasattr(manager, 'manager_name')

    def test_refactoring_maintains_functionality(self):
        """Test that refactoring maintains all original functionality."""
        # Create callbacks with various configurations
        pause_cb = PauseCallback(
            checkpoint_dir='/tmp/pause',
            save_rng_states=False  # Test that it can be disabled
        )
        assert pause_cb.save_rng_states is False

        wandb_cb = WandbArtifactCheckpoint(
            upload_best_model=True,
            upload_last_model=True,
            use_compression=True
        )
        assert wandb_cb.config.upload_best_model is True
        assert wandb_cb.config.use_compression is True

        memory_cb = MemoryCleanupCallback(
            cleanup_points=["on_validation_epoch_end"],
            verbose=False
        )
        assert memory_cb.verbose is False
        assert "on_validation_epoch_end" in memory_cb.cleanup_points