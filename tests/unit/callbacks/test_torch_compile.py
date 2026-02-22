"""
Tests for TorchCompileCallback.

Test coverage:
- Module navigation (success and failure cases)
- Whole-model compilation
- Multiple module compilation
- Cleanup behavior
- Checkpoint metadata handling
- State tracking
- Error handling
"""

import pytest
import torch
import torch.nn as nn
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback
from lightning_reflow.callbacks.torch_compile import TorchCompileCallback, CompilationMetadata


class SimpleModule(nn.Module):
    """Simple module for testing."""
    def __init__(self, input_dim=10, output_dim=10):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class TestModel(pl.LightningModule):
    """Simple LightningModule for testing."""
    def __init__(self):
        super().__init__()
        self.encoder = SimpleModule(10, 20)
        self.decoder = SimpleModule(20, 10)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


class SimpleDummyDataset(torch.utils.data.Dataset):
    """Simple dataset for testing."""
    def __init__(self, size=100):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = torch.randn(10)
        y = torch.randn(10)
        return x, y


class TestTorchCompileCallback:
    """Test suite for TorchCompileCallback."""

    @pytest.fixture
    def model(self):
        """Create a test model."""
        return TestModel()

    @pytest.fixture
    def dataloader(self):
        """Create a test dataloader."""
        dataset = SimpleDummyDataset(size=10)
        return torch.utils.data.DataLoader(dataset, batch_size=2)

    def test_callback_creation(self):
        """Test that callback can be created with default parameters."""
        callback = TorchCompileCallback()
        assert callback.enabled is True
        assert callback.mode == "default"
        assert callback.dynamic is None
        assert callback.fullgraph is None
        assert callback.backend is None
        assert callback.options is None
        assert callback.disable is None
        assert callback.target_modules == []

    def test_callback_disabled(self, model, dataloader):
        """Test that callback does nothing when disabled."""
        callback = TorchCompileCallback(enabled=False, verbose=False)

        trainer = Trainer(
            max_epochs=1,
            max_steps=2,
            callbacks=[callback],
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
        )

        trainer.fit(model, dataloader)

        # No modules should be compiled
        assert len(callback.metadata.compiled_modules) == 0

    def test_module_navigation_success(self, model):
        """Test successful module navigation and compilation."""
        callback = TorchCompileCallback(
            enabled=True,
            target_modules=["encoder"],
            verbose=False,
        )

        # Simulate setup call
        callback.setup(None, model, "fit")

        # Check that encoder was compiled
        assert "encoder" in callback.metadata.compiled_modules
        assert callback.metadata.compiled_modules["encoder"]["mode"] == "default"

    def test_module_navigation_nested(self, model):
        """Test navigation to nested modules."""
        callback = TorchCompileCallback(
            enabled=True,
            target_modules=["encoder.linear"],
            verbose=False,
        )

        callback.setup(None, model, "fit")

        # Check that nested module was compiled
        assert "encoder.linear" in callback.metadata.compiled_modules

    def test_none_module_skipped(self, model):
        """None module (e.g., optional augmentation) is skipped gracefully."""
        model.augmentation = None
        callback = TorchCompileCallback(
            enabled=True,
            target_modules=["encoder", "augmentation"],
            verbose=False,
        )
        callback.setup(None, model, "fit")

        # encoder should be compiled, augmentation skipped (not an error)
        assert "encoder" in callback.metadata.compiled_modules
        assert "augmentation" not in callback.metadata.compiled_modules
        assert len(callback.metadata.compilation_errors) == 0
        assert "augmentation" not in callback.metadata.fallback_modules

    def test_module_navigation_invalid_path(self, model):
        """Test error handling for invalid module paths."""
        callback = TorchCompileCallback(
            enabled=True,
            target_modules=["nonexistent_module"],
            verbose=False,
        )

        # Should not raise, but should record error
        callback.setup(None, model, "fit")

        # Check that error was recorded
        assert len(callback.metadata.compilation_errors) > 0
        assert "nonexistent_module" in callback.metadata.fallback_modules

    def test_whole_model_compilation_empty_list(self, model):
        """Test whole-model compilation with empty target_modules list."""
        callback = TorchCompileCallback(
            enabled=True,
            target_modules=[],  # Empty list = compile entire model
            verbose=False,
        )

        callback.setup(None, model, "fit")

        # Check that entire model was marked as compiled
        assert "<entire_model>" in callback.metadata.compiled_modules

    def test_whole_model_compilation_dot(self, model):
        """Test whole-model compilation with '.' notation."""
        callback = TorchCompileCallback(
            enabled=True,
            target_modules=["."],  # Dot = compile entire model
            verbose=False,
        )

        callback.setup(None, model, "fit")

        # Check that entire model was marked as compiled
        assert "<entire_model>" in callback.metadata.compiled_modules

    def test_multiple_modules_compilation(self, model):
        """Test compiling multiple modules."""
        callback = TorchCompileCallback(
            enabled=True,
            target_modules=["encoder", "decoder"],
            verbose=False,
        )

        callback.setup(None, model, "fit")

        # Check that both modules were compiled
        assert "encoder" in callback.metadata.compiled_modules
        assert "decoder" in callback.metadata.compiled_modules
        assert len(callback.metadata.compiled_modules) == 2

    def test_module_specific_configs(self, model):
        """Test per-module configuration."""
        callback = TorchCompileCallback(
            enabled=True,
            module_configs=[
                {"module_path": "encoder", "mode": "max-autotune", "dynamic": False},
                {"module_path": "decoder", "mode": "default", "dynamic": True},
            ],
            verbose=False,
        )

        callback.setup(None, model, "fit")

        # Check that modules were compiled with different settings
        assert callback.metadata.compiled_modules["encoder"]["mode"] == "max-autotune"
        assert callback.metadata.compiled_modules["decoder"]["mode"] == "default"
        assert callback.metadata.compiled_modules["encoder"]["dynamic"] is False
        assert callback.metadata.compiled_modules["decoder"]["dynamic"] is True

    def test_compilation_metadata_tracking(self, model):
        """Test that compilation metadata is properly tracked."""
        callback = TorchCompileCallback(
            enabled=True,
            target_modules=["encoder"],
            verbose=False,
        )

        callback.setup(None, model, "fit")

        # Check metadata
        assert callback.metadata.torch_version == torch.__version__
        assert "encoder" in callback.metadata.compiled_modules
        assert "encoder" in callback.metadata.compilation_time_ms
        assert callback.metadata.compilation_time_ms["encoder"] >= 0

    def test_checkpoint_save_metadata_only(self, model):
        """Test that checkpoint saves metadata only, not compiled graphs."""
        callback = TorchCompileCallback(
            enabled=True,
            target_modules=["encoder"],
            verbose=False,
        )

        callback.setup(None, model, "fit")

        # Create mock checkpoint
        checkpoint = {}
        callback.on_save_checkpoint(None, model, checkpoint)

        # Check that metadata was saved
        assert "torch_compile_metadata" in checkpoint
        metadata = checkpoint["torch_compile_metadata"]
        assert "torch_version" in metadata
        assert "compiled_modules" in metadata
        assert "encoder" in metadata["compiled_modules"]

        # Ensure no compiled graphs or large objects are saved
        import sys
        checkpoint_size = sys.getsizeof(str(checkpoint))
        assert checkpoint_size < 10000  # Should be small (just metadata)

    def test_checkpoint_load_metadata(self, model):
        """Test loading metadata from checkpoint."""
        callback = TorchCompileCallback(
            enabled=True,
            verbose=False,
        )

        # Create mock checkpoint with metadata
        checkpoint = {
            "torch_compile_metadata": {
                "torch_version": "2.0.0",
                "compiled_modules": ["encoder", "decoder"],
                "compilation_config": {
                    "mode": "default",
                    "dynamic": None,
                    "fullgraph": None,
                    "backend": None,
                    "options": None,
                    "disable": None,
                },
            }
        }

        # Should not raise
        callback.on_load_checkpoint(None, model, checkpoint)

    def test_cleanup_on_fit_end(self, model, dataloader):
        """Test that cleanup is called on fit end."""
        callback = TorchCompileCallback(
            enabled=True,
            target_modules=["encoder"],
            cleanup_on_fit_end=True,
            verbose=False,
        )

        trainer = Trainer(
            max_epochs=1,
            max_steps=2,
            callbacks=[callback],
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
        )

        trainer.fit(model, dataloader)

        # Cleanup should have been called (we can't easily verify dynamo reset,
        # but we can check the callback completed without errors)
        assert callback.metadata.compiled_modules  # Should have compiled something

    def test_cleanup_on_exception(self, model):
        """Test that cleanup is called on exception."""
        callback = TorchCompileCallback(
            enabled=True,
            target_modules=["encoder"],
            cleanup_on_exception=True,
            verbose=False,
        )

        # Simulate exception handling
        try:
            raise ValueError("Test exception")
        except ValueError as e:
            callback.on_exception(None, model, e)

        # Should complete without raising

    def test_validation_invalid_mode(self):
        """Test that invalid compilation mode raises error."""
        with pytest.raises(ValueError, match="Invalid compilation mode"):
            TorchCompileCallback(enabled=True, mode="invalid_mode")

    def test_state_dict(self):
        """Test that callback state can be serialized."""
        callback = TorchCompileCallback(
            enabled=True,
            mode="max-autotune",
            dynamic=False,
            fullgraph=True,
            backend="inductor",
            target_modules=["encoder"],
        )

        state = callback.state_dict()

        assert state["enabled"] is True
        assert state["mode"] == "max-autotune"
        assert state["dynamic"] is False
        assert state["fullgraph"] is True
        assert state["backend"] == "inductor"
        assert state["target_modules"] == ["encoder"]

    def test_load_state_dict(self):
        """Test that callback state can be loaded."""
        callback = TorchCompileCallback()

        state = {
            "enabled": False,
            "mode": "reduce-overhead",
            "dynamic": True,
            "fullgraph": False,
            "backend": "inductor",
            "target_modules": ["decoder"],
        }

        callback.load_state_dict(state)

        assert callback.enabled is False
        assert callback.mode == "reduce-overhead"
        assert callback.dynamic is True
        assert callback.fullgraph is False
        assert callback.backend == "inductor"
        assert callback.target_modules == ["decoder"]

    def test_compilation_with_different_modes(self):
        """Test compilation with different modes."""
        for mode in ["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"]:
            # Create fresh model for each mode to avoid recompilation issues
            model = TestModel()

            callback = TorchCompileCallback(
                enabled=True,
                mode=mode,
                target_modules=["encoder"],
                verbose=False,
            )

            callback.setup(None, model, "fit")

            assert callback.metadata.compiled_modules["encoder"]["mode"] == mode

    def test_skip_compilation_outside_fit_stage(self, model):
        """Test that compilation is skipped outside fit stage."""
        callback = TorchCompileCallback(
            enabled=True,
            target_modules=["encoder"],
            verbose=False,
        )

        # Try setup in test stage
        callback.setup(None, model, "test")

        # Should not have compiled anything
        assert len(callback.metadata.compiled_modules) == 0

    def test_none_passthrough_default_params(self, model):
        """Test that None parameters are not passed to torch.compile (use PyTorch defaults)."""
        callback = TorchCompileCallback(
            enabled=True,
            mode="default",
            dynamic=None,  # Should NOT be passed to torch.compile
            fullgraph=None,  # Should NOT be passed to torch.compile
            backend=None,  # Should NOT be passed to torch.compile
            options=None,  # Should NOT be passed to torch.compile
            disable=None,  # Should NOT be passed to torch.compile
            target_modules=["encoder"],
            verbose=False,
        )

        callback.setup(None, model, "fit")

        # Check that only mode is in the compiled config
        config = callback.metadata.compiled_modules["encoder"]
        assert "mode" in config
        assert config["mode"] == "default"
        # None values should not be in config
        assert "dynamic" not in config
        assert "fullgraph" not in config
        assert "backend" not in config
        assert "options" not in config
        assert "disable" not in config

    def test_explicit_params_passed_through(self, model):
        """Test that explicit (non-None) parameters are passed to torch.compile."""
        callback = TorchCompileCallback(
            enabled=True,
            mode="max-autotune",
            dynamic=False,  # Explicit value should be passed
            fullgraph=True,  # Explicit value should be passed
            backend="inductor",  # Explicit value should be passed
            target_modules=["encoder"],
            verbose=False,
        )

        callback.setup(None, model, "fit")

        config = callback.metadata.compiled_modules["encoder"]
        assert config["mode"] == "max-autotune"
        assert config["dynamic"] is False
        assert config["fullgraph"] is True
        assert config["backend"] == "inductor"

    def test_mixed_none_and_explicit_params(self, model):
        """Test mixed None and explicit parameters."""
        callback = TorchCompileCallback(
            enabled=True,
            mode="reduce-overhead",
            dynamic=True,  # Explicit
            fullgraph=None,  # None - should not be passed
            backend="inductor",  # Explicit
            options=None,  # None - should not be passed
            disable=None,  # None - should not be passed
            target_modules=["encoder"],
            verbose=False,
        )

        callback.setup(None, model, "fit")

        config = callback.metadata.compiled_modules["encoder"]
        assert config["mode"] == "reduce-overhead"
        assert config["dynamic"] is True
        assert config["backend"] == "inductor"
        assert "fullgraph" not in config
        assert "options" not in config
        assert "disable" not in config

    def test_per_module_none_passthrough(self, model):
        """Test None-passthrough works with per-module configs."""
        callback = TorchCompileCallback(
            enabled=True,
            mode="default",  # Global default
            dynamic=None,  # Global None
            module_configs=[
                {
                    "module_path": "encoder",
                    "mode": "max-autotune",
                    "dynamic": False,  # Explicit override
                    "fullgraph": None,  # None - should not be passed
                },
                {
                    "module_path": "decoder",
                    "mode": "default",
                    # dynamic inherited from global (None) - should not be passed
                    "backend": "inductor",  # Explicit
                },
            ],
            verbose=False,
        )

        callback.setup(None, model, "fit")

        # Check encoder config
        encoder_config = callback.metadata.compiled_modules["encoder"]
        assert encoder_config["mode"] == "max-autotune"
        assert encoder_config["dynamic"] is False
        assert "fullgraph" not in encoder_config

        # Check decoder config
        decoder_config = callback.metadata.compiled_modules["decoder"]
        assert decoder_config["mode"] == "default"
        assert decoder_config["backend"] == "inductor"
        assert "dynamic" not in decoder_config  # Inherited None should not be passed


class TestTargetMethods:
    """Tests for target_methods (method-level compilation)."""

    @pytest.fixture
    def model(self):
        return TestModel()

    def test_compile_method(self, model):
        """Test that a method can be compiled by path."""
        callback = TorchCompileCallback(
            enabled=True,
            target_modules=[],
            target_methods=["forward"],
            verbose=False,
        )
        callback.setup(None, model, "fit")
        assert "forward" in callback.metadata.compiled_modules

    def test_compile_method_invalid_path(self, model):
        """Test error handling for invalid method paths."""
        callback = TorchCompileCallback(
            enabled=True,
            target_modules=[],
            target_methods=["nonexistent_method"],
            verbose=False,
        )
        callback.setup(None, model, "fit")
        assert len(callback.metadata.compilation_errors) > 0
        assert "nonexistent_method" in callback.metadata.fallback_modules

    def test_compile_method_not_callable(self, model):
        """Test error when path points to a non-callable."""
        # loss_fn is an nn.Module (callable), but let's add a non-callable attr
        model.some_value = 42
        callback = TorchCompileCallback(
            enabled=True,
            target_modules=[],
            target_methods=["some_value"],
            verbose=False,
        )
        callback.setup(None, model, "fit")
        assert len(callback.metadata.compilation_errors) > 0

    def test_compile_method_replaces_on_instance(self, model):
        """Compiled method should be callable and produce correct results."""
        original_output = model.forward(torch.randn(1, 10))

        callback = TorchCompileCallback(
            enabled=True,
            target_modules=[],
            target_methods=["forward"],
            verbose=False,
        )
        callback.setup(None, model, "fit")

        # Method should still be callable
        compiled_output = model.forward(torch.randn(1, 10))
        assert compiled_output.shape == original_output.shape

    def test_modules_and_methods_together(self, model):
        """Both target_modules and target_methods can be used together."""
        callback = TorchCompileCallback(
            enabled=True,
            target_modules=["encoder"],
            target_methods=["forward"],
            verbose=False,
        )
        callback.setup(None, model, "fit")
        assert "encoder" in callback.metadata.compiled_modules
        assert "forward" in callback.metadata.compiled_modules

    def test_empty_methods_no_whole_model(self):
        """Empty target_modules + non-empty target_methods should NOT compile whole model."""
        model = TestModel()
        callback = TorchCompileCallback(
            enabled=True,
            target_modules=[],
            target_methods=["forward"],
            verbose=False,
        )
        callback.setup(None, model, "fit")
        assert "<entire_model>" not in callback.metadata.compiled_modules
        assert "forward" in callback.metadata.compiled_modules

    def test_state_dict_includes_target_methods(self):
        callback = TorchCompileCallback(
            target_methods=["forward", "_compute_loss_impl"],
        )
        state = callback.state_dict()
        assert state["target_methods"] == ["forward", "_compute_loss_impl"]

    def test_load_state_dict_restores_target_methods(self):
        callback = TorchCompileCallback()
        callback.load_state_dict({"target_methods": ["my_method"]})
        assert callback.target_methods == ["my_method"]


class TestCompilationMetadata:
    """Test suite for CompilationMetadata dataclass."""

    def test_metadata_creation(self):
        """Test that metadata can be created."""
        metadata = CompilationMetadata()

        assert isinstance(metadata.compiled_modules, dict)
        assert isinstance(metadata.compilation_time_ms, dict)
        assert isinstance(metadata.fallback_modules, list)
        assert isinstance(metadata.compilation_errors, list)
        assert metadata.torch_version == ""

    def test_metadata_recording(self):
        """Test recording compilation data."""
        metadata = CompilationMetadata()
        metadata.torch_version = torch.__version__

        metadata.compiled_modules["encoder"] = {"mode": "default", "dynamic": True}
        metadata.compilation_time_ms["encoder"] = 123.45

        assert "encoder" in metadata.compiled_modules
        assert metadata.compilation_time_ms["encoder"] == 123.45

    def test_error_recording(self):
        """Test recording compilation errors."""
        metadata = CompilationMetadata()

        error_info = {
            "module": "encoder",
            "error": "Test error message",
            "config": {"mode": "default"},
        }
        metadata.compilation_errors.append(error_info)
        metadata.fallback_modules.append("encoder")

        assert len(metadata.compilation_errors) == 1
        assert "encoder" in metadata.fallback_modules


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
