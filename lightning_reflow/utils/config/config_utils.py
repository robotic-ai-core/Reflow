"""
Configuration utilities for LightningReflow.

Provides utilities for handling Lightning's class_path configurations,
including recursive instantiation of nested model structures.
"""

from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)


def instantiate_class_path_recursive(config: Any, parent_key: str = "root") -> Any:
    """
    Recursively instantiate Lightning class_path configurations.

    This function traverses a configuration dictionary and instantiates any
    nested structures that use Lightning's class_path pattern.

    Lightning's class_path pattern looks like:
    {
        "class_path": "module.path.ClassName",
        "init_args": {
            "arg1": "value1",
            "nested_model": {
                "class_path": "module.path.NestedClass",
                "init_args": {...}
            }
        }
    }

    This function will recursively instantiate nested_model before instantiating
    the parent, ensuring all dependencies are objects rather than dicts.

    Args:
        config: Configuration value (can be dict, list, or primitive)
        parent_key: Key name for debugging purposes (default: "root")

    Returns:
        Instantiated object if config has class_path, otherwise returns config as-is
        (with nested class_paths recursively instantiated)

    Example:
        >>> config = {
        ...     "class_path": "torch.nn.Linear",
        ...     "init_args": {
        ...         "in_features": 10,
        ...         "out_features": 5,
        ...         "bias": True
        ...     }
        ... }
        >>> model = instantiate_class_path_recursive(config)
        >>> isinstance(model, torch.nn.Linear)
        True
    """
    # Base case 1: Not a dict, return as-is
    if not isinstance(config, dict):
        return config

    # Base case 2: Empty dict
    if not config:
        return config

    # Base case 3: Already an instantiated object (has __class__ but not 'class_path')
    if hasattr(config, '__class__') and not isinstance(config, dict):
        return config

    # Check if this dict has a class_path (Lightning's instantiation pattern)
    has_class_path = 'class_path' in config

    if has_class_path:
        # This is a Lightning class_path config - instantiate it
        try:
            from lightning.pytorch.cli import instantiate_class

            # First, recursively process init_args to instantiate any nested models
            init_args = config.get('init_args', {})
            if isinstance(init_args, dict):
                processed_init_args = {}
                for key, value in init_args.items():
                    # Recursively process each argument
                    processed_init_args[key] = instantiate_class_path_recursive(
                        value, parent_key=f"{parent_key}.{key}"
                    )

                # Create modified config with processed init_args
                processed_config = config.copy()
                processed_config['init_args'] = processed_init_args
            else:
                processed_config = config

            # Now instantiate using Lightning's built-in function
            instance = instantiate_class(tuple(), processed_config)

            logger.debug(
                f"Instantiated {config['class_path']} for {parent_key}"
            )
            return instance

        except Exception as e:
            logger.warning(
                f"Failed to instantiate class_path config for {parent_key}: {e}\n"
                f"Config: {config}\n"
                f"Returning config as-is (dict)"
            )
            # Fall through to dict processing below

    # Not a class_path config - recursively process dict values
    if isinstance(config, dict):
        result = {}
        for key, value in config.items():
            if isinstance(value, (dict, list)):
                # Recursively process nested structures
                result[key] = instantiate_class_path_recursive(
                    value, parent_key=f"{parent_key}.{key}"
                )
            else:
                # Primitive value - keep as-is
                result[key] = value
        return result

    # For lists, recursively process each element
    if isinstance(config, list):
        return [
            instantiate_class_path_recursive(item, parent_key=f"{parent_key}[{i}]")
            for i, item in enumerate(config)
        ]

    # Fallback: return as-is
    return config


def should_instantiate_nested_configs(model_args: Dict[str, Any]) -> bool:
    """
    Determine if model_args contains nested class_path configs that need instantiation.

    Args:
        model_args: Model initialization arguments

    Returns:
        True if any argument contains a class_path configuration
    """
    if not isinstance(model_args, dict):
        return False

    for key, value in model_args.items():
        if isinstance(value, dict) and 'class_path' in value:
            return True
        # Check nested dicts
        if isinstance(value, dict):
            if should_instantiate_nested_configs(value):
                return True
        # Check lists of dicts
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and 'class_path' in item:
                    return True

    return False
