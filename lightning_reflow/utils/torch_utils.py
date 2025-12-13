import logging
import torch
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Common keys used in batch dictionaries for input data
BATCH_SIZE_KEYS = ('observation.images', 'input', 'x', 'image', 'data')


def extract_batch_size(batch: Any, default: int = 1) -> int:
    """
    Extract batch size from various batch formats.

    Handles common batch formats:
    - Direct tensor: returns tensor.shape[0]
    - List/tuple: returns first element's shape[0] if it has shape
    - Dict: tries common keys first, then falls back to first tensor found

    Args:
        batch: The batch data in any common format
        default: Default value to return if batch size cannot be determined

    Returns:
        The batch size, or default if it cannot be determined
    """
    if batch is None:
        return default

    # Direct tensor
    if isinstance(batch, torch.Tensor):
        return batch.shape[0]

    # Dictionary batch
    if isinstance(batch, dict):
        # Try common keys first
        for key in BATCH_SIZE_KEYS:
            if key in batch:
                tensor = batch[key]
                if hasattr(tensor, 'shape') and len(tensor.shape) > 0:
                    return tensor.shape[0]

        # Fallback: get first tensor's batch size
        for value in batch.values():
            if isinstance(value, torch.Tensor):
                return value.shape[0]
            if hasattr(value, 'shape') and len(value.shape) > 0:
                return value.shape[0]

    # List or tuple batch
    if isinstance(batch, (list, tuple)) and len(batch) > 0:
        first_elem = batch[0]
        if isinstance(first_elem, torch.Tensor):
            return first_elem.shape[0]
        if hasattr(first_elem, 'shape') and len(first_elem.shape) > 0:
            return first_elem.shape[0]

    # Generic object with shape attribute
    if hasattr(batch, 'shape') and len(batch.shape) > 0:
        return batch.shape[0]

    logger.debug(f"Could not determine batch size from {type(batch)}, using default={default}")
    return default


def get_torch_generator_from_seed(seed: Optional[int] = None) -> torch.Generator:
    """
    Get a torch.Generator object from a seed.
    
    Args:
        seed: The seed to use. If None, a random seed will be used.
        
    Returns:
        A torch.Generator object.
    """
    if seed is None:
        return torch.Generator()
    else:
        return torch.Generator().manual_seed(seed) 