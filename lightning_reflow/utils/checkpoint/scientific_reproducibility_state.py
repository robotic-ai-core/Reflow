"""
Scientific Reproducibility Manager State for checkpoint persistence.

This module provides comprehensive scientific reproducibility state management
for perfect checkpoint resume, including RNG states, torch.compile info,
and deterministic computation settings.
"""

import logging
import random
from typing import Dict, Any, Optional, List
import numpy as np
import torch
import functools

logger = logging.getLogger(__name__)


class ScientificReproducibilityState:
    """
    Manager state for scientific reproducibility across checkpoint operations.

    Handles:
    - RNG state preservation (Python, NumPy, PyTorch, CUDA)
    - torch.compile metadata and recompilation
    - Deterministic computation settings
    - Functional cache management
    """

    def __init__(self):
        self.rng_states: Dict[str, Any] = {}
        self.compile_info: Dict[str, Any] = {}
        self.deterministic_settings: Dict[str, Any] = {}
        self._model_reference: Optional[Any] = None
        self._trainer_reference: Optional[Any] = None

    @property
    def manager_name(self) -> str:
        return "scientific_reproducibility"

    def set_references(self, model: Any = None, trainer: Any = None) -> None:
        """Set references to model and trainer for operations."""
        self._model_reference = model
        self._trainer_reference = trainer

    def capture_state(self) -> Dict[str, Any]:
        """Capture comprehensive scientific state for persistence."""
        state = {
            'rng_states': self._capture_rng_states(),
            'compile_info': self._capture_compile_info(),
            'deterministic_settings': self._capture_deterministic_settings(),
            'version': '1.0.0'
        }

        logger.debug(f"Captured scientific reproducibility state: "
                    f"RNG states: {list(state['rng_states'].keys())}, "
                    f"Compile info: {bool(state['compile_info'])}")
        return state

    def restore_state(self, state: Dict[str, Any]) -> bool:
        """Restore scientific state from persistence."""
        try:
            # Check version compatibility
            version = state.get('version', '0.0.0')
            if version != '1.0.0':
                logger.warning(f"Version mismatch in scientific reproducibility state: {version} != 1.0.0")

            # Restore components
            success = True

            # Restore RNG states
            if 'rng_states' in state:
                self._restore_rng_states(state['rng_states'])
                logger.info("✅ Restored RNG states for scientific reproducibility")

            # Store compile info for later recompilation
            if 'compile_info' in state:
                self.compile_info = state['compile_info']
                # Recompilation happens in post-restoration hook
                logger.info("✅ Stored compile info for recompilation")

            # Restore deterministic settings
            if 'deterministic_settings' in state:
                self._restore_deterministic_settings(state['deterministic_settings'])
                logger.info("✅ Restored deterministic computation settings")

            return success

        except Exception as e:
            logger.error(f"Failed to restore scientific reproducibility state: {e}")
            return False

    def validate_state(self, state: Dict[str, Any]) -> bool:
        """Validate that the state is compatible."""
        return (isinstance(state, dict) and
                'version' in state and
                isinstance(state.get('rng_states'), dict))

    def post_restoration_hook(self) -> None:
        """
        Called after checkpoint restoration to perform model-dependent operations.
        This should be called by the framework after the model is loaded.
        """
        # Recompile model if needed
        if self.compile_info and self._model_reference:
            self._recompile_model()

        # Clear caches
        if self._model_reference:
            self._clear_model_caches()

    def _capture_rng_states(self) -> Dict[str, Any]:
        """Capture all RNG states."""
        states = {}

        # Python random
        try:
            states['python'] = random.getstate()
        except Exception as e:
            logger.debug(f"Could not capture Python random state: {e}")

        # NumPy random
        try:
            states['numpy'] = np.random.get_state()
        except Exception as e:
            logger.debug(f"Could not capture NumPy random state: {e}")

        # PyTorch CPU
        try:
            states['torch_cpu'] = torch.get_rng_state()
        except Exception as e:
            logger.debug(f"Could not capture PyTorch CPU state: {e}")

        # PyTorch CUDA (all devices)
        if torch.cuda.is_available():
            try:
                states['torch_cuda'] = {
                    'all': torch.cuda.get_rng_state_all(),
                    'current_device': torch.cuda.current_device()
                }
            except Exception as e:
                logger.debug(f"Could not capture CUDA state: {e}")

        return states

    def _restore_rng_states(self, states: Dict[str, Any]) -> None:
        """Restore all RNG states."""

        # Python random
        if 'python' in states and states['python'] is not None:
            try:
                random.setstate(states['python'])
            except Exception as e:
                logger.debug(f"Could not restore Python random state: {e}")

        # NumPy random
        if 'numpy' in states and states['numpy'] is not None:
            try:
                np.random.set_state(states['numpy'])
            except Exception as e:
                logger.debug(f"Could not restore NumPy random state: {e}")

        # PyTorch CPU
        if 'torch_cpu' in states and states['torch_cpu'] is not None:
            try:
                torch.set_rng_state(states['torch_cpu'])
            except Exception as e:
                logger.debug(f"Could not restore PyTorch CPU state: {e}")

        # PyTorch CUDA
        if 'torch_cuda' in states and states['torch_cuda'] is not None and torch.cuda.is_available():
            try:
                cuda_state = states['torch_cuda']
                if isinstance(cuda_state, dict) and 'all' in cuda_state:
                    torch.cuda.set_rng_state_all(cuda_state['all'])
                else:
                    # Legacy format
                    torch.cuda.set_rng_state(cuda_state)
            except Exception as e:
                logger.debug(f"Could not restore CUDA state: {e}")

    def _capture_compile_info(self) -> Dict[str, Any]:
        """Capture torch.compile information from the model."""
        info = {}

        if not self._model_reference:
            return info

        model = self._model_reference

        # Check if model is compiled
        if hasattr(model, '_orig_mod'):
            info['model_compiled'] = True
            logger.debug("Model is compiled")

        # Check for compiled submodules
        compiled_modules = []
        for name, module in model.named_modules():
            if hasattr(module, '_orig_mod'):
                compiled_modules.append(name)

        if compiled_modules:
            info['compiled_modules'] = compiled_modules
            logger.debug(f"Found compiled modules: {compiled_modules}")

        # Store torch.compile settings if available
        if hasattr(model, 'hparams') and hasattr(model.hparams, 'torch_compile_settings'):
            info['compile_settings'] = dict(model.hparams.torch_compile_settings)

        return info

    def _recompile_model(self) -> None:
        """Recompile the model based on stored compile info."""
        if not self._model_reference or not self.compile_info:
            return

        model = self._model_reference

        # If model has its own recompilation method, use it
        if hasattr(model, '_apply_torch_compile'):
            try:
                model._apply_torch_compile()
                logger.info("🔄 Recompiled model using _apply_torch_compile()")
                return
            except Exception as e:
                logger.warning(f"Failed to recompile using model method: {e}")

        # Otherwise, try to recompile based on stored info
        compile_settings = self.compile_info.get('compile_settings', {})
        compiled_modules = self.compile_info.get('compiled_modules', [])

        if compiled_modules:
            # Prepare compilation kwargs
            compile_kwargs = {}
            if compile_settings.get('mode'):
                compile_kwargs['mode'] = compile_settings['mode']
            if compile_settings.get('backend'):
                compile_kwargs['backend'] = compile_settings['backend']

            if not compile_kwargs:
                compile_kwargs = {'mode': 'default'}

            # Recompile each module
            for module_name in compiled_modules:
                try:
                    # Get the module by name
                    parts = module_name.split('.')
                    module = model
                    for part in parts:
                        module = getattr(module, part, None)
                        if module is None:
                            break

                    if module and not hasattr(module, '_orig_mod'):
                        # Not already compiled, compile it
                        compiled = torch.compile(module, **compile_kwargs)

                        # Set it back
                        parent = model
                        for part in parts[:-1]:
                            parent = getattr(parent, part)
                        setattr(parent, parts[-1], compiled)

                        logger.info(f"🔄 Recompiled module: {module_name}")
                except Exception as e:
                    logger.warning(f"Could not recompile {module_name}: {e}")

    def _capture_deterministic_settings(self) -> Dict[str, Any]:
        """Capture current deterministic computation settings."""
        settings = {}

        # PyTorch deterministic settings
        settings['torch_deterministic'] = torch.are_deterministic_algorithms_enabled()

        # CUDNN settings
        if torch.cuda.is_available():
            settings['cudnn_deterministic'] = torch.backends.cudnn.deterministic
            settings['cudnn_benchmark'] = torch.backends.cudnn.benchmark

        # Attention backend settings (if available)
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            settings['flash_sdp'] = torch.backends.cuda.flash_sdp_enabled()
            settings['mem_efficient_sdp'] = torch.backends.cuda.mem_efficient_sdp_enabled()
            settings['math_sdp'] = torch.backends.cuda.math_sdp_enabled()

        return settings

    def _restore_deterministic_settings(self, settings: Dict[str, Any]) -> None:
        """Restore deterministic computation settings."""

        # PyTorch deterministic
        if 'torch_deterministic' in settings:
            try:
                torch.use_deterministic_algorithms(settings['torch_deterministic'], warn_only=True)
            except Exception as e:
                logger.debug(f"Could not restore deterministic algorithms setting: {e}")

        # CUDNN settings
        if torch.cuda.is_available():
            if 'cudnn_deterministic' in settings:
                torch.backends.cudnn.deterministic = settings['cudnn_deterministic']
            if 'cudnn_benchmark' in settings:
                torch.backends.cudnn.benchmark = settings['cudnn_benchmark']

        # Attention backends
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            if 'flash_sdp' in settings:
                torch.backends.cuda.enable_flash_sdp(settings['flash_sdp'])
            if 'mem_efficient_sdp' in settings:
                torch.backends.cuda.enable_mem_efficient_sdp(settings['mem_efficient_sdp'])
            if 'math_sdp' in settings:
                torch.backends.cuda.enable_math_sdp(settings['math_sdp'])

    def _clear_model_caches(self) -> None:
        """Clear functional caches in the model."""
        if not self._model_reference:
            return

        model = self._model_reference

        # Clear model's own cache if it has a clear_cache method
        if hasattr(model, 'clear_cache') and callable(model.clear_cache):
            try:
                model.clear_cache()
                logger.debug("Cleared model cache via clear_cache()")
            except Exception as e:
                logger.debug(f"Could not clear model cache: {e}")

        # Clear any lru_cache decorated methods
        for name in dir(model):
            try:
                attr = getattr(model, name)
                if hasattr(attr, 'cache_clear'):
                    attr.cache_clear()
                    logger.debug(f"Cleared cache for {name}")
            except Exception:
                pass

        # Clear caches in submodules
        for name, module in model.named_modules():
            if module is model:
                continue

            if hasattr(module, 'clear_cache') and callable(module.clear_cache):
                try:
                    module.clear_cache()
                    logger.debug(f"Cleared cache for submodule {name}")
                except Exception:
                    pass


# Export for use
__all__ = ['ScientificReproducibilityState']