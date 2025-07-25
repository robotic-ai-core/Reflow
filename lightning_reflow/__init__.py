"""
Lightning Reflow - Advanced PyTorch Lightning framework extension.

A self-contained Lightning PyTorch framework extension with advanced pause/resume,
W&B integration, and CLI capabilities.
"""

__version__ = "0.1.0"
__author__ = "Lightning Reflow Contributors"
__email__ = ""
__license__ = "MIT"

# Core imports - expose main components
from lightning_reflow.core import LightningReflow
from lightning_reflow.cli import LightningReflowCLI
from lightning_reflow.models import SimpleReflowModel
from lightning_reflow.data import SimpleDataModule

# Callback imports
from lightning_reflow.callbacks import (
    PauseCallback,
    WandbWatchCallback,
    ConfigSummaryLogger,
    StepOutputLoggerCallback,
    MemoryCleanupCallback,
    EnvironmentCallback,
    FlowProgressBarCallback,
)

# Utility imports
from lightning_reflow.utils import get_torch_generator_from_seed
from lightning_reflow.utils.logging import EnvironmentManager

__all__ = [
    # Core components
    "LightningReflow",
    "LightningReflowCLI",
    "SimpleReflowModel", 
    "SimpleDataModule",
    # Callbacks
    "PauseCallback",
    "WandbWatchCallback",
    "ConfigSummaryLogger",
    "StepOutputLoggerCallback",
    "MemoryCleanupCallback",
    "EnvironmentCallback",
    "FlowProgressBarCallback",
    # Utilities
    "get_torch_generator_from_seed",
    "EnvironmentManager",
    # Metadata
    "__version__",
    "__author__",
    "__license__",
]
