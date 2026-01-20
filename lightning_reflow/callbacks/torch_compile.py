"""
TorchCompileCallback for PyTorch Lightning models.

Provides declarative, centralized torch.compile configuration for training workflows.
Supports module-level compilation, whole-model compilation, and comprehensive cleanup
for HPO trial isolation.
"""

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback


@dataclass
class CompilationMetadata:
    """
    Track compilation state for monitoring and debugging.

    This dataclass records:
    - Which modules were compiled and with what settings
    - How long compilation took for each module
    - Which modules fell back to eager mode
    - Any compilation errors that occurred
    """
    compiled_modules: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    compilation_time_ms: Dict[str, float] = field(default_factory=dict)
    fallback_modules: List[str] = field(default_factory=list)
    compilation_errors: List[Dict[str, str]] = field(default_factory=list)
    torch_version: str = ""


class TorchCompileCallback(Callback):
    """
    Production-ready callback for torch.compile integration with PyTorch Lightning.

    Features:
    - String path module navigation (e.g., "dynamics_model.transformer_encoder")
    - Whole-model compilation support (target_modules: [] or ["."])
    - Comprehensive state tracking via CompilationMetadata
    - Enhanced cleanup (dynamo, CUDA cache, cuBLAS workspaces)
    - Checkpoint metadata (config only, NOT compiled graphs)
    - HPO trial isolation (cleanup on fit_end and on_exception)
    - Performance metrics logging to trainer
    - Graceful error handling with detailed messages

    Example:
        ```yaml
        trainer:
          callbacks:
            - class_path: lightning_reflow.callbacks.TorchCompileCallback
              init_args:
                enabled: true
                mode: "max-autotune"
                target_modules: ["dynamics_model.transformer_encoder"]
        ```
    """

    def __init__(
        self,
        enabled: bool = True,
        mode: str = "default",
        dynamic: Optional[bool] = None,
        fullgraph: Optional[bool] = None,
        backend: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        disable: Optional[bool] = None,
        inductor_config: Optional[Dict[str, Any]] = None,
        dynamo_config: Optional[Dict[str, Any]] = None,
        target_modules: Optional[List[str]] = None,
        module_configs: Optional[List[Dict[str, Any]]] = None,
        cleanup_on_fit_end: bool = True,
        cleanup_on_exception: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize TorchCompileCallback.

        Args:
            enabled: Whether compilation is enabled
            mode: Compilation mode ("default", "reduce-overhead", "max-autotune",
                "max-autotune-no-cudagraphs"). The last option provides max-autotune
                kernel optimization without CUDA graph overhead, reducing VRAM usage.
            dynamic: Control dynamic shape handling. None (default) defers to torch.compile's
                mode-specific defaults. True enables dynamic shapes. False requires static shapes
                (enables CUDA graphs when supported by the mode).
            fullgraph: If True, requires entire function to be capturable in a single graph.
                None (default) defers to torch.compile's default (False).
            backend: Backend to use for compilation. None (default) uses torch.compile's
                default ('inductor').
            options: Additional options dict passed to the backend. None (default) uses
                torch.compile's defaults.
            disable: If True, compilation is disabled at torch.compile level.
                None (default) defers to torch.compile's default (False).
            inductor_config: Dict of torch._inductor.config settings to apply before compilation.
                Example: {"triton.cudagraph_skip_dynamic_graphs": True}
                Keys use dot notation and are applied via setattr on nested config objects.
            dynamo_config: Dict of torch._dynamo.config settings to apply before compilation.
                Example: {"cache_size_limit": 256}
                Keys use dot notation and are applied via setattr on nested config objects.
            target_modules: List of module paths to compile (empty list = whole model)
            module_configs: List of per-module configs (overrides global settings)
            cleanup_on_fit_end: Clean up compilation state after training
            cleanup_on_exception: Clean up compilation state on exception
            verbose: Print compilation status messages
        """
        super().__init__()
        self.enabled = enabled
        self.mode = mode
        self.dynamic = dynamic
        self.fullgraph = fullgraph
        self.backend = backend
        self.options = options
        self.disable = disable
        self.inductor_config = inductor_config or {}
        self.dynamo_config = dynamo_config or {}
        self.target_modules = target_modules or []
        self.module_configs = module_configs or []
        self.cleanup_on_fit_end = cleanup_on_fit_end
        self.cleanup_on_exception = cleanup_on_exception
        self.verbose = verbose

        # State tracking
        self.metadata = CompilationMetadata()
        self.metadata.torch_version = torch.__version__

        # Validate configuration
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate callback configuration."""
        valid_modes = ["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"]
        if self.mode not in valid_modes:
            raise ValueError(
                f"Invalid compilation mode: {self.mode}. "
                f"Must be one of {valid_modes}"
            )

    def _apply_inductor_config(self) -> None:
        """
        Apply torch._inductor.config settings.

        Supports nested config keys using dot notation:
        - "triton.cudagraph_skip_dynamic_graphs" -> torch._inductor.config.triton.cudagraph_skip_dynamic_graphs
        - "max_autotune" -> torch._inductor.config.max_autotune
        """
        if not hasattr(torch, '_inductor'):
            if self.verbose:
                warnings.warn(
                    "torch._inductor not available, skipping inductor_config settings",
                    UserWarning,
                    stacklevel=2
                )
            return

        for key, value in self.inductor_config.items():
            try:
                # Handle nested keys with dot notation
                parts = key.split('.')
                config_obj = torch._inductor.config

                # Navigate to nested object
                for part in parts[:-1]:
                    config_obj = getattr(config_obj, part)

                # Set the final attribute
                setattr(config_obj, parts[-1], value)

                if self.verbose:
                    print(f"🔧 Set torch._inductor.config.{key} = {value}")

            except AttributeError as e:
                if self.verbose:
                    warnings.warn(
                        f"Failed to set inductor config '{key}': {e}",
                        UserWarning,
                        stacklevel=2
                    )

    def _apply_dynamo_config(self) -> None:
        """
        Apply torch._dynamo.config settings.

        Supports nested config keys using dot notation:
        - "cache_size_limit" -> torch._dynamo.config.cache_size_limit
        - "suppress_errors" -> torch._dynamo.config.suppress_errors
        """
        if not hasattr(torch, '_dynamo'):
            if self.verbose:
                warnings.warn(
                    "torch._dynamo not available, skipping dynamo_config settings",
                    UserWarning,
                    stacklevel=2
                )
            return

        for key, value in self.dynamo_config.items():
            try:
                # Handle nested keys with dot notation
                parts = key.split('.')
                config_obj = torch._dynamo.config

                # Navigate to nested object
                for part in parts[:-1]:
                    config_obj = getattr(config_obj, part)

                # Set the final attribute
                setattr(config_obj, parts[-1], value)

                if self.verbose:
                    print(f"🔧 Set torch._dynamo.config.{key} = {value}")

            except AttributeError as e:
                if self.verbose:
                    warnings.warn(
                        f"Failed to set dynamo config '{key}': {e}",
                        UserWarning,
                        stacklevel=2
                    )

    def _uses_cudagraphs(self) -> bool:
        """Check if current configuration uses CUDA graphs."""
        # CUDA graphs are used when:
        # 1. mode is "reduce-overhead" or "max-autotune"
        # 2. AND dynamic=False (or not set, which defaults to static for these modes)
        # 3. AND triton.cudagraphs is not explicitly disabled
        if self.mode in ("reduce-overhead", "max-autotune"):
            # If dynamic is explicitly True, CUDA graphs won't work
            if self.dynamic is True:
                return False
            # Check if cudagraphs is explicitly disabled in inductor_config
            if self.inductor_config.get("triton.cudagraphs") is False:
                return False
            return True
        return False

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        """
        Apply compilation during setup phase (after model is on device).

        Args:
            trainer: PyTorch Lightning trainer
            pl_module: LightningModule to compile
            stage: Training stage ("fit", "validate", "test", "predict")
        """
        if not self.enabled:
            if self.verbose:
                print("torch.compile is disabled, skipping compilation")
            return

        if stage != "fit":
            # Only compile during fit stage
            return

        # Apply inductor and dynamo config settings before compilation
        if self.inductor_config:
            self._apply_inductor_config()
        if self.dynamo_config:
            self._apply_dynamo_config()

        # Use module-specific configs if provided
        if self.module_configs:
            self._compile_with_module_configs(pl_module)
        else:
            # Use global config for all target modules
            self._compile_with_global_config(pl_module)

    def _compile_with_global_config(self, pl_module: pl.LightningModule) -> None:
        """Compile modules using global configuration."""
        base_config = {"mode": self.mode}

        # Only add parameters if explicitly configured (not None)
        # This allows torch.compile to use its mode-specific and parameter-specific defaults
        if self.dynamic is not None:
            base_config["dynamic"] = self.dynamic
        if self.fullgraph is not None:
            base_config["fullgraph"] = self.fullgraph
        if self.backend is not None:
            base_config["backend"] = self.backend
        if self.options is not None:
            base_config["options"] = self.options
        if self.disable is not None:
            base_config["disable"] = self.disable

        # Check for whole-model compilation
        if not self.target_modules or self.target_modules == ["."]:
            self._compile_entire_model(pl_module, base_config)
        else:
            # Compile specific modules
            for module_path in self.target_modules:
                if module_path == ".":
                    self._compile_entire_model(pl_module, base_config)
                else:
                    self._compile_module(pl_module, module_path, base_config)

    def _compile_with_module_configs(self, pl_module: pl.LightningModule) -> None:
        """Compile modules using per-module configurations."""
        for config in self.module_configs:
            module_path = config.get("module_path", ".")
            compile_config = {"mode": config.get("mode", self.mode)}

            # Only add parameters if explicitly configured (not None)
            # Prioritize per-module config, then fall back to global config
            for param_name, global_value in [
                ("dynamic", self.dynamic),
                ("fullgraph", self.fullgraph),
                ("backend", self.backend),
                ("options", self.options),
                ("disable", self.disable),
            ]:
                param_value = config.get(param_name, global_value)
                if param_value is not None:
                    compile_config[param_name] = param_value

            if module_path == ".":
                self._compile_entire_model(pl_module, compile_config)
            else:
                self._compile_module(pl_module, module_path, compile_config)

    def _compile_entire_model(self, pl_module: pl.LightningModule, config: Dict[str, Any]) -> None:
        """
        Compile the entire LightningModule.

        This compiles forward(), training_step(), and validation_step() since
        Lightning calls these methods directly during training/validation.
        Consider compiling specific modules for better debugging.
        """
        if self.verbose:
            warnings.warn(
                "⚠️  Compiling entire LightningModule! "
                "Consider compiling specific modules for better debugging.",
                UserWarning,
                stacklevel=2
            )

        compiled_methods = []
        try:
            start_time = time.perf_counter()

            # Compile training_step (main training path in Lightning)
            if hasattr(pl_module, "training_step"):
                original_training_step = pl_module.training_step
                pl_module.training_step = torch.compile(original_training_step, **config)
                compiled_methods.append("training_step")

            # Compile validation_step if it exists and is overridden
            if hasattr(pl_module, "validation_step"):
                # Check if it's actually overridden (not just inherited default)
                if pl_module.__class__.validation_step is not pl.LightningModule.validation_step:
                    original_validation_step = pl_module.validation_step
                    pl_module.validation_step = torch.compile(original_validation_step, **config)
                    compiled_methods.append("validation_step")

            # Compile forward for inference/predict use cases
            if hasattr(pl_module, "forward"):
                original_forward = pl_module.forward
                pl_module.forward = torch.compile(original_forward, **config)
                compiled_methods.append("forward")

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Record metadata
            module_name = "<entire_model>"
            self.metadata.compiled_modules[module_name] = config.copy()
            self.metadata.compilation_time_ms[module_name] = elapsed_ms

            if self.verbose:
                methods_str = ", ".join(compiled_methods)
                print(f"✅ Compiled {methods_str} in {elapsed_ms:.1f}ms (mode={config['mode']})")

        except Exception as e:
            error_info = {
                "module": "<entire_model>",
                "error": str(e),
                "config": config,
            }
            self.metadata.compilation_errors.append(error_info)
            self.metadata.fallback_modules.append("<entire_model>")

            if self.verbose:
                print(f"⚠️  torch.compile failed for entire model: {e}")
                print("    Running without compilation.")

    def _compile_module(
        self,
        pl_module: pl.LightningModule,
        module_path: str,
        config: Dict[str, Any]
    ) -> None:
        """
        Compile a specific module by string path navigation.

        Args:
            pl_module: Parent LightningModule
            module_path: Dotted path to module (e.g., "dynamics_model.encoder")
            config: Compilation configuration
        """
        try:
            # Navigate to target module
            target_module = pl_module
            path_parts = module_path.split(".")

            for part in path_parts:
                if not hasattr(target_module, part):
                    available_attrs = [
                        attr for attr in dir(target_module)
                        if not attr.startswith("_") and hasattr(target_module, attr)
                    ]
                    raise AttributeError(
                        f"Module path '{module_path}' invalid: "
                        f"'{part}' not found in {type(target_module).__name__}.\n"
                        f"Available attributes: {', '.join(available_attrs[:10])}"
                        f"{' ...' if len(available_attrs) > 10 else ''}"
                    )
                target_module = getattr(target_module, part)

            # Verify it's a module
            if not isinstance(target_module, torch.nn.Module):
                raise TypeError(
                    f"Module path '{module_path}' points to {type(target_module)}, "
                    f"not a torch.nn.Module"
                )

            # Compile the module
            start_time = time.perf_counter()
            compiled_module = torch.compile(target_module, **config)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Replace the module in the parent
            parent = pl_module
            for part in path_parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, path_parts[-1], compiled_module)

            # Record metadata
            self.metadata.compiled_modules[module_path] = config.copy()
            self.metadata.compilation_time_ms[module_path] = elapsed_ms

            if self.verbose:
                print(
                    f"✅ Compiled {module_path} in {elapsed_ms:.1f}ms "
                    f"(mode={config['mode']})"
                )

        except Exception as e:
            error_info = {
                "module": module_path,
                "error": str(e),
                "config": config,
            }
            self.metadata.compilation_errors.append(error_info)
            self.metadata.fallback_modules.append(module_path)

            if self.verbose:
                print(f"⚠️  torch.compile failed for '{module_path}': {e}")
                print("    Running without compilation for this module.")

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """
        Mark start of training step for CUDA graph management.

        This is required for CUDA graphs to work properly during training.
        Without this, PyTorch will warn: "Unable to hit fast path of CUDAGraphs
        because of pending, uninvoked backwards."

        The cudagraph_mark_step_begin() call tells the CUDA graph tree system
        that a new training step is starting, allowing it to properly manage
        graph capture and replay.
        """
        if not self.enabled:
            return

        if self._uses_cudagraphs():
            if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()

    def on_validation_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Mark start of validation step for CUDA graph management."""
        if not self.enabled:
            return

        if self._uses_cudagraphs():
            if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Clean up compilation state after training completes."""
        if self.cleanup_on_fit_end:
            self._comprehensive_cleanup()
            if self.verbose:
                print("🧹 Cleaned up torch.compile state after training")

    def on_exception(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        exception: BaseException
    ) -> None:
        """Clean up compilation state on exception (important for HPO trials)."""
        if self.cleanup_on_exception:
            self._comprehensive_cleanup()
            if self.verbose:
                print("🧹 Cleaned up torch.compile state after exception")

    def _post_checkpoint_cleanup(self) -> None:
        """
        Clean up after checkpoint loading to reduce memory fragmentation.

        On resume, checkpoint loading can fragment GPU memory because:
        1. Optimizer state is loaded (2x model size for AdamW)
        2. Tensors are allocated in different order than fresh training
        3. This fragmentation can cause OOM during CUDA graph capture

        This cleanup:
        - Resets dynamo state (clears any partially-captured graphs)
        - Clears CUDA cache (consolidates free memory)
        - Does NOT reset peak memory stats (useful for debugging)
        """
        import gc

        # Force Python garbage collection first
        gc.collect()

        # Reset dynamo compilation state to clear any cached graphs
        if hasattr(torch, '_dynamo'):
            torch._dynamo.reset()

        # Clean up CUDA resources
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            # Clear cuBLAS workspaces if available
            if hasattr(torch._C, '_cuda_clearCublasWorkspaces'):
                torch._C._cuda_clearCublasWorkspaces()

        if self.verbose:
            print("🔄 Cleared dynamo state and CUDA cache after checkpoint load")

    def _comprehensive_cleanup(self) -> None:
        """
        Comprehensive cleanup strategy for HPO trial isolation.

        Cleans up:
        - torch._dynamo state
        - CUDA cache and memory pools
        - Peak memory stats
        - cuBLAS workspaces (if available)

        Note: Triton cache is NOT cleaned for performance reasons.
        """
        # Reset dynamo compilation state
        if hasattr(torch, '_dynamo'):
            torch._dynamo.reset()

        # Clean up CUDA resources
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

            # Clear cuBLAS workspaces if available
            if hasattr(torch._C, '_cuda_clearCublasWorkspaces'):
                torch._C._cuda_clearCublasWorkspaces()

    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: Dict[str, Any]
    ) -> None:
        """
        Save compilation metadata to checkpoint (NOT compiled graphs).

        IMPORTANT: Only metadata is saved:
        - Which modules were compiled
        - What compilation settings were used
        - Torch version for compatibility

        Compiled FX graphs, CUDA graphs, and optimization artifacts
        are NOT saved because they're not portable/deterministic.

        Model parameters ARE saved normally via Lightning.
        On resume, modules are recompiled automatically in setup().
        """
        checkpoint["torch_compile_metadata"] = {
            "torch_version": self.metadata.torch_version,
            "compiled_modules": list(self.metadata.compiled_modules.keys()),
            "compilation_config": {
                "mode": self.mode,
                "dynamic": self.dynamic,
                "fullgraph": self.fullgraph,
                "backend": self.backend,
                "options": self.options,
                "disable": self.disable,
                "inductor_config": self.inductor_config,
                "dynamo_config": self.dynamo_config,
                "target_modules": self.target_modules,
            },
            "fallback_modules": self.metadata.fallback_modules,
            "errors": self.metadata.compilation_errors,
        }

    def on_load_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: Dict[str, Any]
    ) -> None:
        """
        Load compilation metadata from checkpoint.

        This is primarily for logging/debugging. The actual compilation
        will happen in setup() with the current configuration.

        IMPORTANT: After checkpoint load, we reset dynamo state and clear CUDA cache
        to prevent memory fragmentation issues. On resume:
        1. setup() wraps modules with torch.compile (lazy compilation)
        2. Checkpoint loads model weights + optimizer state
        3. This hook runs - we clear any partially-captured graphs
        4. Training starts with clean compilation state
        """
        if "torch_compile_metadata" in checkpoint:
            saved_metadata = checkpoint["torch_compile_metadata"]

            if self.verbose:
                print("📋 Loaded torch.compile metadata from checkpoint:")
                print(f"   Torch version: {saved_metadata.get('torch_version')}")
                print(f"   Compiled modules: {len(saved_metadata.get('compiled_modules', []))}")

                # Warn if torch version changed
                current_version = torch.__version__
                saved_version = saved_metadata.get('torch_version', '')
                if current_version != saved_version:
                    warnings.warn(
                        f"Torch version mismatch: checkpoint was created with {saved_version}, "
                        f"but current version is {current_version}. "
                        "Recompilation will occur.",
                        UserWarning,
                        stacklevel=2
                    )

        # Reset dynamo state and clear CUDA cache after checkpoint load
        # This prevents memory fragmentation from checkpoint loading interfering
        # with CUDA graph capture during training
        if self.enabled:
            self._post_checkpoint_cleanup()

    def state_dict(self) -> Dict[str, Any]:
        """Return callback state for checkpointing."""
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "dynamic": self.dynamic,
            "fullgraph": self.fullgraph,
            "backend": self.backend,
            "options": self.options,
            "disable": self.disable,
            "inductor_config": self.inductor_config,
            "dynamo_config": self.dynamo_config,
            "target_modules": self.target_modules,
            "module_configs": self.module_configs,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load callback state from checkpoint."""
        self.enabled = state_dict.get("enabled", self.enabled)
        self.mode = state_dict.get("mode", self.mode)
        self.dynamic = state_dict.get("dynamic", self.dynamic)
        self.fullgraph = state_dict.get("fullgraph", self.fullgraph)
        self.backend = state_dict.get("backend", self.backend)
        self.options = state_dict.get("options", self.options)
        self.disable = state_dict.get("disable", self.disable)
        self.inductor_config = state_dict.get("inductor_config", self.inductor_config)
        self.dynamo_config = state_dict.get("dynamo_config", self.dynamo_config)
        self.target_modules = state_dict.get("target_modules", self.target_modules)
        self.module_configs = state_dict.get("module_configs", self.module_configs)
