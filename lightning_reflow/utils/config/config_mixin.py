"""
ConfigMixin for nn.Module classes to support config-based serialization and reconstruction.

This mixin enables the "ignore + manual passing" pattern for checkpoint loading with
minimal boilerplate. It provides:

- save_config(): Captures __init__ arguments automatically via frame inspection
- get_config(): Returns a serializable config dict (nested modules become nested configs)
- from_config(): Class method to reconstruct module from config

Usage:
    from lightning_reflow.utils.config import ConfigMixin

    class MyModel(ConfigMixin, nn.Module):
        def __init__(self, hidden_dim: int, num_layers: int):
            super().__init__()
            self.save_config()  # Call right after super().__init__()

            self.hidden_dim = hidden_dim
            self.layers = nn.ModuleList([...])

    # Save config
    config = model.get_config()  # {'hidden_dim': 64, 'num_layers': 4}

    # Reconstruct
    model2 = MyModel.from_config(config)

For nested ConfigMixin modules, configs are automatically extracted:

    class OuterModel(ConfigMixin, nn.Module):
        def __init__(self, inner: nn.Module, scale: float):
            super().__init__()
            self.save_config()  # inner's config is automatically nested
            self.inner = inner

    # config = {
    #     'inner': {'__class_path__': 'mymodule.InnerModel', '__config__': {...}},
    #     'scale': 2.0
    # }
"""

import inspect
import importlib
import logging
from typing import Any, Dict, List, Optional, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T', bound='ConfigMixin')


class ConfigMixin:
    """
    Mixin for nn.Module classes to support config-based serialization.

    Provides automatic capture of __init__ arguments and reconstruction from config.
    For nested ConfigMixin modules, their configs are automatically extracted and
    nested, enabling full reconstruction of module hierarchies.
    """

    _config: Dict[str, Any]

    def save_config(self, ignore: Optional[List[str]] = None) -> None:
        """
        Capture __init__ arguments as config.

        Call this right after super().__init__() in your __init__ method.
        Uses frame inspection to capture the original argument values.

        For arguments that are ConfigMixin instances, their config is automatically
        extracted (not the module itself), enabling serialization.

        Args:
            ignore: List of argument names to exclude from config
        """
        ignore_set = set(ignore or [])
        ignore_set.add('self')

        frame = inspect.currentframe()
        if frame is None:
            raise RuntimeError("Could not get current frame for config capture")

        try:
            # Go up one level to get the __init__ frame
            caller_frame = frame.f_back
            if caller_frame is None:
                raise RuntimeError("Could not get caller frame for config capture")

            local_vars = caller_frame.f_locals

            # Get parameter names from __init__ signature
            # This ensures we only capture actual parameters, not local variables
            init_sig = inspect.signature(self.__class__.__init__)
            param_names = [
                p.name for p in init_sig.parameters.values()
                if p.name not in ignore_set
            ]

            # Build config from parameters
            config: Dict[str, Any] = {}
            for name in param_names:
                if name in local_vars:
                    value = local_vars[name]
                    config[name] = self._serialize_value(value)

            self._config = config
            logger.debug(f"{self.__class__.__name__}.save_config() captured: {list(config.keys())}")

        finally:
            del frame  # Avoid reference cycles

    def _serialize_value(self, value: Any) -> Any:
        """
        Serialize a value for config storage.

        Handles:
        - ConfigMixin instances: Extracts their config with class path
        - Dicts/lists: Recursively serializes contents
        - Primitives: Returns as-is
        """
        if isinstance(value, ConfigMixin):
            # Nested configurable module - save class path and config
            return {
                '__class_path__': f"{value.__class__.__module__}.{value.__class__.__name__}",
                '__config__': value.get_config()
            }
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._serialize_value(v) for v in value]
        elif isinstance(value, tuple):
            # Preserve tuple type
            return {'__tuple__': [self._serialize_value(v) for v in value]}
        else:
            # Primitive or other serializable value
            return value

    def get_config(self) -> Dict[str, Any]:
        """
        Return the saved config dict.

        Returns:
            Dict containing the __init__ arguments captured by save_config()

        Raises:
            RuntimeError: If save_config() was not called in __init__
        """
        if not hasattr(self, '_config'):
            raise RuntimeError(
                f"{self.__class__.__name__}.save_config() was not called in __init__(). "
                f"Add 'self.save_config()' right after 'super().__init__()'."
            )
        return self._config.copy()

    @classmethod
    def from_config(cls: Type[T], config: Dict[str, Any]) -> T:
        """
        Reconstruct module from a config dict.

        Handles nested ConfigMixin modules by recursively instantiating them.

        Args:
            config: Config dict as returned by get_config()

        Returns:
            New instance of the class constructed with the config values
        """
        processed = {}
        for key, value in config.items():
            processed[key] = _deserialize_value(value)
        return cls(**processed)


def _deserialize_value(value: Any) -> Any:
    """
    Deserialize a value from config storage.

    Handles:
    - Nested ConfigMixin configs: Reconstructs the module
    - Tuples: Reconstructs from __tuple__ marker
    - Dicts/lists: Recursively deserializes contents
    - Primitives: Returns as-is
    """
    if isinstance(value, dict):
        if '__class_path__' in value and '__config__' in value:
            # Nested ConfigMixin module - reconstruct it
            cls = _import_class(value['__class_path__'])
            return cls.from_config(value['__config__'])
        elif '__tuple__' in value:
            # Tuple reconstruction
            return tuple(_deserialize_value(v) for v in value['__tuple__'])
        else:
            return {k: _deserialize_value(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_deserialize_value(v) for v in value]
    else:
        return value


def _import_class(class_path: str) -> Type:
    """
    Import a class from its fully qualified path.

    Args:
        class_path: Fully qualified class path (e.g., 'mymodule.submodule.MyClass')

    Returns:
        The imported class

    Raises:
        ImportError: If the module cannot be imported
        AttributeError: If the class doesn't exist in the module
    """
    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
