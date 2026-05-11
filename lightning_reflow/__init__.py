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
from .core import LightningReflow
from .cli import LightningReflowCLI
from .models import SimpleReflowModel
from .data import SimpleDataModule

# Callback imports — keep top-level exports aligned with callbacks/__init__.py
from .callbacks import (
    ConfigSummaryLogger,
    EnvironmentCallback,
    FlowProgressBarCallback,
    GradientNormMonitorCallback,
    LossRecorderCallback,
    MemoryCleanupCallback,
    PauseCallback,
    StepOutputLoggerCallback,
    ThroughputMonitorCallback,
    TorchCompileCallback,
    TrainingProfilerCallback,
    WandbArtifactCheckpoint,
    WandbConfigLoggerCallback,
    WandbWatchCallback,
)

# Utility imports
from .utils import get_torch_generator_from_seed
from .utils.logging import EnvironmentManager

__all__ = [
    # Core components
    "LightningReflow",
    "LightningReflowCLI",
    "SimpleReflowModel",
    "SimpleDataModule",
    # Callbacks (mirrors callbacks/__init__.py)
    "ConfigSummaryLogger",
    "EnvironmentCallback",
    "FlowProgressBarCallback",
    "GradientNormMonitorCallback",
    "LossRecorderCallback",
    "MemoryCleanupCallback",
    "PauseCallback",
    "StepOutputLoggerCallback",
    "ThroughputMonitorCallback",
    "TorchCompileCallback",
    "TrainingProfilerCallback",
    "WandbArtifactCheckpoint",
    "WandbConfigLoggerCallback",
    "WandbWatchCallback",
    # Utilities
    "get_torch_generator_from_seed",
    "EnvironmentManager",
    # Metadata
    "__version__",
    "__author__",
    "__license__",
]
