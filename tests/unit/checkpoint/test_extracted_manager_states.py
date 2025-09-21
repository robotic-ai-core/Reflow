"""
Test extracted manager state classes.

This test validates that the refactored manager state classes
(WandbArtifactState and FlowProgressBarState) work correctly.
"""

import pytest
from unittest.mock import MagicMock, Mock
from typing import Dict, Any

from lightning_reflow.utils.checkpoint.wandb_artifact_state import WandbArtifactState
from lightning_reflow.utils.checkpoint.flow_progress_bar_state import FlowProgressBarState
from lightning_reflow.utils.checkpoint.scientific_reproducibility_state import ScientificReproducibilityState


class TestWandbArtifactState:
    """Test the extracted WandbArtifactState manager."""

    def test_wandb_artifact_state_initialization(self):
        """Test that WandbArtifactState can be initialized with a callback."""
        # Create mock callback with required attributes
        mock_callback = MagicMock()
        mock_callback.config = MagicMock()
        mock_callback.config.__dict__ = {'upload_best_model': True, 'use_compression': True}
        mock_callback.state = MagicMock()
        mock_callback.state.__dict__ = {'has_uploaded': False, 'epoch_count': 0}

        # Initialize state manager
        state_manager = WandbArtifactState(mock_callback)

        assert state_manager.callback == mock_callback
        assert state_manager.manager_name == "wandb_artifact_checkpoint"

    def test_wandb_artifact_state_capture(self):
        """Test state capture for WandbArtifactState."""
        # Create mock callback
        mock_callback = MagicMock()
        mock_callback.config = MagicMock()
        mock_callback.config.__dict__ = {
            'upload_best_model': True,
            'upload_last_model': False,
            'use_compression': True
        }
        mock_callback.state = MagicMock()
        mock_callback.state.__dict__ = {
            'has_uploaded': True,
            'epoch_count': 5,
            'validation_count': 10
        }

        # Capture state
        state_manager = WandbArtifactState(mock_callback)
        captured_state = state_manager.capture_state()

        assert captured_state['version'] == '2.0'
        assert captured_state['config']['upload_best_model'] is True
        assert captured_state['state']['epoch_count'] == 5
        assert captured_state['state']['validation_count'] == 10

    def test_wandb_artifact_state_restore(self):
        """Test state restoration for WandbArtifactState."""
        # Create mock callback
        mock_callback = MagicMock()
        mock_callback.config = MagicMock()
        mock_callback.state = MagicMock()

        # Initialize with empty state
        mock_callback.state.has_uploaded = False
        mock_callback.state.epoch_count = 0
        mock_callback.state.validation_count = 0

        # Create state to restore
        state_to_restore = {
            'version': '2.0',
            'config': {'upload_best_model': True},
            'state': {
                'has_uploaded': True,
                'epoch_count': 10,
                'validation_count': 20
            }
        }

        # Restore state
        state_manager = WandbArtifactState(mock_callback)
        success = state_manager.restore_state(state_to_restore)

        assert success is True
        assert mock_callback.state.has_uploaded is True
        assert mock_callback.state.epoch_count == 10
        assert mock_callback.state.validation_count == 20

    def test_wandb_artifact_state_validation(self):
        """Test state validation for WandbArtifactState."""
        state_manager = WandbArtifactState(MagicMock())

        # Valid state
        valid_state = {'version': '2.0', 'state': {}}
        assert state_manager.validate_state(valid_state) is True

        # Invalid state - not a dict
        assert state_manager.validate_state("not_a_dict") is False

        # Invalid state - missing version
        assert state_manager.validate_state({'state': {}}) is False


class TestFlowProgressBarState:
    """Test the extracted FlowProgressBarState manager."""

    def test_flow_progress_bar_state_initialization(self):
        """Test that FlowProgressBarState can be initialized with a callback."""
        mock_callback = MagicMock()
        state_manager = FlowProgressBarState(mock_callback)

        assert state_manager.callback == mock_callback
        assert state_manager.manager_name == "flow_progress_bar"

    def test_flow_progress_bar_state_capture(self):
        """Test state capture for FlowProgressBarState."""
        # Create mock callback with progress tracking attributes
        mock_callback = MagicMock()
        mock_callback._validation_count = 5
        mock_callback._last_validation_batch = 100
        mock_callback._refresh_rate = 1
        mock_callback.global_bar_metrics = ['loss', 'acc']
        mock_callback.interval_bar_metrics = ['val_loss']
        mock_callback._bar_colour = 'green'

        # Capture state
        state_manager = FlowProgressBarState(mock_callback)
        captured_state = state_manager.capture_state()

        assert captured_state['version'] == '1.0.0'
        assert captured_state['validation_count'] == 5
        assert captured_state['last_validation_batch'] == 100
        assert captured_state['configuration']['refresh_rate'] == 1
        assert captured_state['configuration']['global_bar_metrics'] == ['loss', 'acc']
        assert 'timestamp' in captured_state

    def test_flow_progress_bar_state_restore(self, capsys):
        """Test state restoration for FlowProgressBarState."""
        # Create mock callback
        mock_callback = MagicMock()
        mock_callback._validation_count = 0
        mock_callback._last_validation_batch = 0
        mock_callback._global_metric_keys_cache = "some_cache"
        mock_callback._interval_metric_keys_cache = "some_cache"
        mock_callback._available_metric_keys_cache = "some_cache"

        # Create state to restore
        state_to_restore = {
            'version': '1.0.0',
            'validation_count': 10,
            'last_validation_batch': 200,
            'configuration': {},
            'timestamp': 123456789
        }

        # Restore state
        state_manager = FlowProgressBarState(mock_callback)
        success = state_manager.restore_state(state_to_restore)

        assert success is True
        assert mock_callback._validation_count == 10
        assert mock_callback._last_validation_batch == 200
        # Check that caches were cleared
        assert mock_callback._global_metric_keys_cache is None
        assert mock_callback._interval_metric_keys_cache is None
        assert mock_callback._available_metric_keys_cache is None

    def test_flow_progress_bar_state_validation(self, capsys):
        """Test state validation for FlowProgressBarState."""
        state_manager = FlowProgressBarState(MagicMock())

        # Valid state
        valid_state = {
            'version': '1.0.0',
            'validation_count': 5,
            'last_validation_batch': 100,
            'configuration': {}
        }
        assert state_manager.validate_state(valid_state) is True

        # Invalid state - wrong version
        invalid_state = {
            'version': '2.0.0',
            'validation_count': 5,
            'last_validation_batch': 100
        }
        assert state_manager.validate_state(invalid_state) is False

        # Check error message was printed
        captured = capsys.readouterr()
        assert "Incompatible state version" in captured.out

        # Invalid state - missing validation_count
        invalid_state = {
            'version': '1.0.0',
            'last_validation_batch': 100
        }
        assert state_manager.validate_state(invalid_state) is False

        # Invalid state - missing both last_validation_batch and last_validation_step
        invalid_state = {
            'version': '1.0.0',
            'validation_count': 5
        }
        assert state_manager.validate_state(invalid_state) is False

    def test_flow_progress_bar_backward_compatibility(self):
        """Test backward compatibility with old field names."""
        mock_callback = MagicMock()
        mock_callback._validation_count = 0
        mock_callback._last_validation_batch = 0

        # Use old field name 'last_validation_step'
        state_with_old_field = {
            'version': '1.0.0',
            'validation_count': 15,
            'last_validation_step': 300,  # Old field name
            'configuration': {}
        }

        state_manager = FlowProgressBarState(mock_callback)
        success = state_manager.restore_state(state_with_old_field)

        assert success is True
        assert mock_callback._validation_count == 15
        assert mock_callback._last_validation_batch == 300  # Mapped from old field


class TestScientificReproducibilityState:
    """Test the ScientificReproducibilityState manager."""

    def test_scientific_reproducibility_state_basics(self):
        """Test basic functionality of ScientificReproducibilityState."""
        state_manager = ScientificReproducibilityState()

        assert state_manager.manager_name == "scientific_reproducibility"

        # Capture state
        captured_state = state_manager.capture_state()
        assert 'version' in captured_state
        assert 'rng_states' in captured_state
        assert 'compile_info' in captured_state
        assert 'deterministic_settings' in captured_state

    def test_all_manager_states_compatible(self):
        """Test that all manager states follow the same interface."""
        # Create instances
        wandb_state = WandbArtifactState(MagicMock())
        flow_state = FlowProgressBarState(MagicMock())
        scientific_state = ScientificReproducibilityState()

        # All should have manager_name property
        assert hasattr(wandb_state, 'manager_name')
        assert hasattr(flow_state, 'manager_name')
        assert hasattr(scientific_state, 'manager_name')

        # All should have the three required methods
        for state in [wandb_state, flow_state, scientific_state]:
            assert callable(getattr(state, 'capture_state'))
            assert callable(getattr(state, 'restore_state'))
            assert callable(getattr(state, 'validate_state'))

        # All manager names should be unique
        names = [
            wandb_state.manager_name,
            flow_state.manager_name,
            scientific_state.manager_name
        ]
        assert len(names) == len(set(names)), "Manager names must be unique"


class TestManagerStateIntegration:
    """Test integration of manager states with the registration system."""

    def test_manager_registration(self):
        """Test that managers can be registered successfully."""
        from lightning_reflow.utils.checkpoint.manager_state import (
            register_manager,
            unregister_manager,
            capture_all_manager_states,
            get_global_registry
        )

        # Get the registry to check state
        registry = get_global_registry()

        # Register our managers
        wandb_state = WandbArtifactState(MagicMock())
        flow_state = FlowProgressBarState(MagicMock())

        register_manager(wandb_state)
        register_manager(flow_state)

        # Verify they're registered by capturing all states
        all_states = capture_all_manager_states()
        assert "wandb_artifact_checkpoint" in all_states
        assert "flow_progress_bar" in all_states

        # Clean up
        unregister_manager("wandb_artifact_checkpoint")
        unregister_manager("flow_progress_bar")