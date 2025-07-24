import torch
from typing import Optional

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