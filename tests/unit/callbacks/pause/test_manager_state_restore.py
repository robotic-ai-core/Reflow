"""
Tests for manager state restore flow in PauseCallback.on_load_checkpoint.

Verifies that manager states saved during on_save_checkpoint are properly
restored during on_load_checkpoint, including RNG states, DataModule state,
TrainerConfig state, and Environment state.
"""

import random
import time
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pytest
import torch

from lightning_reflow.callbacks.pause.pause_callback import PauseCallback
from lightning_reflow.utils.checkpoint.manager_state import (
    ManagerStateRegistry,
    capture_all_manager_states,
    get_global_registry,
    register_manager,
    restore_all_manager_states,
    unregister_manager,
)
from lightning_reflow.utils.checkpoint.scientific_reproducibility_state import (
    ScientificReproducibilityState,
)


@pytest.fixture(autouse=True)
def clean_registry():
    """Ensure a clean global registry for each test."""
    registry = get_global_registry()
    original_managers = dict(registry.managers)
    registry.managers.clear()
    yield registry
    registry.managers.clear()
    registry.managers.update(original_managers)


@pytest.fixture
def pause_callback(temp_dir):
    """Create a PauseCallback with RNG state saving enabled."""
    return PauseCallback(
        checkpoint_dir=str(temp_dir / "checkpoints"),
        save_rng_states=True,
        skip_dependency_check=True,
    )


@pytest.fixture
def pause_callback_no_rng(temp_dir):
    """Create a PauseCallback with RNG state saving disabled."""
    return PauseCallback(
        checkpoint_dir=str(temp_dir / "checkpoints"),
        save_rng_states=False,
        skip_dependency_check=True,
    )


def _make_trainer_mock(**overrides):
    """Helper to build a consistent mock trainer."""
    trainer = Mock()
    trainer.current_epoch = overrides.get("current_epoch", 5)
    trainer.global_step = overrides.get("global_step", 100)
    trainer.is_global_zero = True
    trainer.max_epochs = overrides.get("max_epochs", 10)
    trainer.logger = Mock()
    return trainer


def _make_pl_module_mock():
    """Helper to build a pl_module mock that is compatible with named_modules()."""
    pl_module = Mock()
    pl_module.named_modules.return_value = iter([])
    return pl_module


class TestManagerStateRoundTrip:
    """Verify the full save -> load cycle restores manager states."""

    def test_on_save_captures_manager_states(self, pause_callback):
        """on_save_checkpoint should embed manager_states in the checkpoint."""
        trainer = _make_trainer_mock()
        pl_module = _make_pl_module_mock()
        checkpoint = {}

        with patch.object(pause_callback, "add_config_metadata"):
            pause_callback.on_save_checkpoint(trainer, pl_module, checkpoint)

        metadata = checkpoint.get("pause_callback_metadata", {})
        assert "manager_states" in metadata
        # The reproducibility manager should be registered and captured
        assert "scientific_reproducibility" in metadata["manager_states"]
        rng = metadata["manager_states"]["scientific_reproducibility"]
        assert "rng_states" in rng
        assert "version" in rng

    def test_on_load_restores_manager_states(self, pause_callback):
        """on_load_checkpoint should call restore_all_manager_states with saved states."""
        trainer = _make_trainer_mock()
        pl_module = _make_pl_module_mock()
        checkpoint = {}

        # Save phase
        with patch.object(pause_callback, "add_config_metadata"):
            pause_callback.on_save_checkpoint(trainer, pl_module, checkpoint)

        # Load phase -- verify restore_all_manager_states is called by checking
        # that RNG state from checkpoint was applied
        with patch(
            "lightning_reflow.utils.checkpoint.manager_state.ManagerStateRegistry.restore_all_states",
            wraps=get_global_registry().restore_all_states,
        ) as mock_restore:
            with patch.object(type(pause_callback).__bases__[0], "on_load_checkpoint"):
                pause_callback.on_load_checkpoint(trainer, pl_module, checkpoint)

            mock_restore.assert_called_once()
            args = mock_restore.call_args[0][0]
            assert "scientific_reproducibility" in args

    def test_rng_states_round_trip(self, pause_callback):
        """RNG states captured during save should be faithfully restored during load."""
        trainer = _make_trainer_mock()
        pl_module = _make_pl_module_mock()
        checkpoint = {}

        # Set deterministic RNG seeds so we know the state
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        # Draw some random numbers to advance state
        _ = random.random()
        _ = np.random.rand(5)
        _ = torch.rand(5)

        # Capture snapshot of current RNG state before save
        py_state_before = random.getstate()
        np_state_before = np.random.get_state()
        torch_state_before = torch.get_rng_state()

        # Save
        with patch.object(pause_callback, "add_config_metadata"):
            pause_callback.on_save_checkpoint(trainer, pl_module, checkpoint)

        # Now perturb RNG states
        random.seed(999)
        np.random.seed(999)
        torch.manual_seed(999)

        # Load -- should restore the states we saved
        with patch.object(type(pause_callback).__bases__[0], "on_load_checkpoint"):
            pause_callback.on_load_checkpoint(trainer, pl_module, checkpoint)

        # Verify Python random state was restored
        assert random.getstate() == py_state_before
        # Verify NumPy random state was restored
        np_restored = np.random.get_state()
        assert np_restored[0] == np_state_before[0]
        assert (np_restored[1] == np_state_before[1]).all()
        # Verify PyTorch CPU state was restored
        assert torch.equal(torch.get_rng_state(), torch_state_before)


class TestOnLoadWithoutManagerStates:
    """Verify graceful handling when checkpoint has no manager_states."""

    def test_no_metadata_key(self, pause_callback):
        """on_load_checkpoint should not fail if pause_callback_metadata is missing."""
        trainer = _make_trainer_mock()
        pl_module = _make_pl_module_mock()
        checkpoint = {}  # No pause_callback_metadata at all

        with patch.object(type(pause_callback).__bases__[0], "on_load_checkpoint"):
            pause_callback.on_load_checkpoint(trainer, pl_module, checkpoint)
        # No exception means success

    def test_empty_manager_states(self, pause_callback):
        """on_load_checkpoint should handle empty manager_states gracefully."""
        trainer = _make_trainer_mock()
        pl_module = _make_pl_module_mock()
        checkpoint = {"pause_callback_metadata": {"manager_states": {}}}

        with patch.object(type(pause_callback).__bases__[0], "on_load_checkpoint"):
            pause_callback.on_load_checkpoint(trainer, pl_module, checkpoint)
        # Empty dict is falsy so restore_all_manager_states is not called

    def test_metadata_without_manager_states_key(self, pause_callback):
        """on_load_checkpoint should handle metadata missing manager_states key."""
        trainer = _make_trainer_mock()
        pl_module = _make_pl_module_mock()
        checkpoint = {"pause_callback_metadata": {"pause_timestamp": time.time()}}

        with patch.object(type(pause_callback).__bases__[0], "on_load_checkpoint"):
            pause_callback.on_load_checkpoint(trainer, pl_module, checkpoint)


class TestOnLoadWithDisabledRng:
    """Verify behaviour when save_rng_states=False."""

    def test_no_rng_manager_registered(self, pause_callback_no_rng, clean_registry):
        """When RNG saving is disabled, no reproducibility manager is registered."""
        assert "scientific_reproducibility" not in clean_registry.managers

    def test_load_with_rng_disabled_and_states_present(self, pause_callback_no_rng):
        """Even with states in checkpoint, disabled RNG should skip post_restoration_hook."""
        trainer = _make_trainer_mock()
        pl_module = _make_pl_module_mock()
        checkpoint = {
            "pause_callback_metadata": {
                "manager_states": {
                    "scientific_reproducibility": {
                        "version": "1.0.0",
                        "rng_states": {},
                        "compile_info": {},
                        "deterministic_settings": {},
                    }
                }
            }
        }

        with patch.object(type(pause_callback_no_rng).__bases__[0], "on_load_checkpoint"):
            pause_callback_no_rng.on_load_checkpoint(trainer, pl_module, checkpoint)
        # Should not crash even though the manager isn't registered


class TestCustomManagerRestore:
    """Verify that non-RNG managers saved in checkpoint are also restored."""

    def test_custom_manager_restored(self, pause_callback, clean_registry):
        """Register a custom manager and verify it gets restored during on_load_checkpoint."""

        class DummyManager:
            manager_name = "dummy_test_manager"

            def __init__(self):
                self.restored_value = None

            def capture_state(self):
                return {"value": 42, "version": "1.0.0"}

            def restore_state(self, state):
                self.restored_value = state.get("value")
                return True

            def validate_state(self, state):
                return isinstance(state, dict) and "value" in state

        dummy = DummyManager()
        register_manager(dummy)

        trainer = _make_trainer_mock()
        pl_module = _make_pl_module_mock()
        checkpoint = {}

        # Save phase captures both scientific_reproducibility and dummy_test_manager
        with patch.object(pause_callback, "add_config_metadata"):
            pause_callback.on_save_checkpoint(trainer, pl_module, checkpoint)

        states = checkpoint["pause_callback_metadata"]["manager_states"]
        assert "dummy_test_manager" in states
        assert states["dummy_test_manager"]["value"] == 42

        # Load phase should restore both
        with patch.object(type(pause_callback).__bases__[0], "on_load_checkpoint"):
            pause_callback.on_load_checkpoint(trainer, pl_module, checkpoint)

        assert dummy.restored_value == 42

        # Cleanup
        unregister_manager("dummy_test_manager")

    def test_partial_restore_failure_logged(self, pause_callback, clean_registry, caplog):
        """If one manager fails to restore, others should still be restored."""

        class FailingManager:
            manager_name = "failing_manager"

            def capture_state(self):
                return {"version": "1.0.0", "rng_states": {}}

            def restore_state(self, state):
                raise RuntimeError("Intentional failure")

            def validate_state(self, state):
                return True

        class SuccessManager:
            manager_name = "success_manager"

            def __init__(self):
                self.was_restored = False

            def capture_state(self):
                return {"ok": True}

            def restore_state(self, state):
                self.was_restored = True
                return True

            def validate_state(self, state):
                return True

        failing = FailingManager()
        success = SuccessManager()
        register_manager(failing)
        register_manager(success)

        trainer = _make_trainer_mock()
        pl_module = _make_pl_module_mock()
        checkpoint = {}

        with patch.object(pause_callback, "add_config_metadata"):
            pause_callback.on_save_checkpoint(trainer, pl_module, checkpoint)

        import logging
        with caplog.at_level(logging.WARNING, logger="lightning_reflow.callbacks.pause.pause_callback"):
            with patch.object(type(pause_callback).__bases__[0], "on_load_checkpoint"):
                pause_callback.on_load_checkpoint(trainer, pl_module, checkpoint)

        # The success manager should still have been restored
        assert success.was_restored is True

        # Failing manager should be reported in the warning log
        assert any("failed" in r.message.lower() for r in caplog.records)

        # Cleanup
        unregister_manager("failing_manager")
        unregister_manager("success_manager")


class TestRestoreBeforePostRestorationHook:
    """Verify ordering: restore_all_manager_states runs BEFORE post_restoration_hook."""

    def test_restore_order(self, pause_callback):
        """Manager states must be restored before post_restoration_hook is called."""
        trainer = _make_trainer_mock()
        pl_module = _make_pl_module_mock()
        checkpoint = {}

        with patch.object(pause_callback, "add_config_metadata"):
            pause_callback.on_save_checkpoint(trainer, pl_module, checkpoint)

        call_order = []

        original_restore = get_global_registry().restore_all_states

        def tracked_restore(states):
            call_order.append("restore_all_manager_states")
            return original_restore(states)

        def tracked_post_hook():
            call_order.append("post_restoration_hook")

        with patch.object(
            get_global_registry(),
            "restore_all_states",
            side_effect=tracked_restore,
        ):
            with patch.object(
                pause_callback._reproducibility_manager,
                "post_restoration_hook",
                side_effect=tracked_post_hook,
            ):
                with patch.object(
                    type(pause_callback).__bases__[0], "on_load_checkpoint"
                ):
                    pause_callback.on_load_checkpoint(trainer, pl_module, checkpoint)

        assert call_order == ["restore_all_manager_states", "post_restoration_hook"]
