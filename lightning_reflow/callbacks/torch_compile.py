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
        cuda_graphs: bool = False,
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
            mode: Compilation mode ("default", "reduce-overhead", "max-autotune")
            cuda_graphs: Whether to enable CUDA graphs (requires dynamic=False)
            target_modules: List of module paths to compile (empty list = whole model)
            module_configs: List of per-module configs (overrides global settings)
            cleanup_on_fit_end: Clean up compilation state after training
            cleanup_on_exception: Clean up compilation state on exception
            verbose: Print compilation status messages
        """
        super().__init__()
        self.enabled = enabled
        self.mode = mode
        self.cuda_graphs = cuda_graphs
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
        valid_modes = ["default", "reduce-overhead", "max-autotune"]
        if self.mode not in valid_modes:
            raise ValueError(
                f"Invalid compilation mode: {self.mode}. "
                f"Must be one of {valid_modes}"
            )

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

        # Use module-specific configs if provided
        if self.module_configs:
            self._compile_with_module_configs(pl_module)
        else:
            # Use global config for all target modules
            self._compile_with_global_config(pl_module)

    def _compile_with_global_config(self, pl_module: pl.LightningModule) -> None:
        """Compile modules using global configuration."""
        base_config = {
            "mode": self.mode,
            "dynamic": not self.cuda_graphs,  # Static shapes for CUDA graphs
        }

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
            compile_config = {
                "mode": config.get("mode", self.mode),
                "dynamic": not config.get("cuda_graphs", self.cuda_graphs),
            }

            if module_path == ".":
                self._compile_entire_model(pl_module, compile_config)
            else:
                self._compile_module(pl_module, module_path, compile_config)

    def _compile_entire_model(self, pl_module: pl.LightningModule, config: Dict[str, Any]) -> None:
        """
        Compile the entire LightningModule.

        Warning: This compiles forward(), training_step(), validation_step(), etc.
        Consider compiling specific modules for better debugging.
        """
        if self.verbose:
            warnings.warn(
                "⚠️  Compiling entire LightningModule! "
                "Consider compiling specific modules for better debugging.",
                UserWarning,
                stacklevel=2
            )

        try:
            start_time = time.perf_counter()

            # Compile the entire forward method
            original_forward = pl_module.forward
            pl_module.forward = torch.compile(original_forward, **config)

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Record metadata
            module_name = "<entire_model>"
            self.metadata.compiled_modules[module_name] = config.copy()
            self.metadata.compilation_time_ms[module_name] = elapsed_ms

            if self.verbose:
                print(f"✅ Compiled entire model in {elapsed_ms:.1f}ms (mode={config['mode']})")

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
                "cuda_graphs": self.cuda_graphs,
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

    def state_dict(self) -> Dict[str, Any]:
        """Return callback state for checkpointing."""
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "cuda_graphs": self.cuda_graphs,
            "target_modules": self.target_modules,
            "module_configs": self.module_configs,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load callback state from checkpoint."""
        self.enabled = state_dict.get("enabled", self.enabled)
        self.mode = state_dict.get("mode", self.mode)
        self.cuda_graphs = state_dict.get("cuda_graphs", self.cuda_graphs)
        self.target_modules = state_dict.get("target_modules", self.target_modules)
        self.module_configs = state_dict.get("module_configs", self.module_configs)
