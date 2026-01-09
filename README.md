# LightningReflow

A PyTorch Lightning extension framework providing advanced training capabilities including pause/resume functionality, W&B checkpoint management, debugging utilities, and an enhanced CLI.

## Features

- **Pause/Resume Training** - Press 'p' to pause training at validation boundaries
- **W&B Checkpoint Artifacts** - Automatic checkpoint uploads to Weights & Biases
- **Enhanced CLI** - Resume subcommand for checkpoint/artifact sources
- **ConfigMixin** - Serialize and reconstruct nn.Module configurations
- **Debugging Utilities** - Crash-resistant logging, thread monitoring
- **Monitoring Callbacks** - Gradient norms, config logging, loss recording

## Installation

```bash
git clone <repo>
cd LightningReflow
pip install -e .
```

## TL;DR (Quickstart)

### Basic Training with Pause/Resume

```python
from lightning_reflow import LightningReflowCLI

# Use as drop-in replacement for LightningCLI
cli = LightningReflowCLI(
    MyModel,
    MyDataModule,
    seed_everything_default=42,
    run=True,
)
```

```bash
# Start training
python train.py fit --config config.yaml

# Resume from pause checkpoint
python train.py resume --checkpoint-path pause_checkpoints/model.ckpt

# Resume from W&B artifact
python train.py resume --checkpoint-artifact user/project/artifact:latest
```

### Using Callbacks Directly

```python
import lightning.pytorch as pl
from lightning_reflow.callbacks import (
    PauseCallback,
    FlowProgressBarCallback,
    GradientNormMonitorCallback,
    WandbConfigLoggerCallback,
)

trainer = pl.Trainer(
    callbacks=[
        PauseCallback(checkpoint_dir="checkpoints", enable_pause=True),
        FlowProgressBarCallback(),
        GradientNormMonitorCallback(log_every_n_steps=10),
        WandbConfigLoggerCallback(flatten=True),
    ]
)
trainer.fit(model, datamodule)
```

## Dual Progress Bars

```bash
Global Steps:  15%|██████████▌                                | 6395/42150 [03:32<19:45, 30.16it/s, lr=0.001]
Interval 2 - Press 'p' to pause:  52%|███████████████████▏     | 2181/4215 [00:47<00:44, train/loss=0.2049]
```

## Pause/Resume Output

```bash
Pause checkpoint uploaded to W&B: user/project/run-pause:latest
Training paused successfully at validation boundary

Training paused. Resume options:
  Local resume:  python train.py resume --checkpoint-path pause_checkpoints/epoch=19_step=16860.ckpt
  W&B resume:    python train.py resume --checkpoint-artifact user/project/run-pause:latest
```

---

## Callbacks Reference

### PauseCallback

Enables pause/resume functionality during training.

```python
from lightning_reflow.callbacks import PauseCallback

PauseCallback(
    checkpoint_dir="pause_checkpoints",
    enable_pause=True,
    pause_key="p",
    upload_key="w",  # Manual W&B upload
)
```

### GradientNormMonitorCallback

Monitors gradient norms and detects clipping events.

```python
from lightning_reflow.callbacks import GradientNormMonitorCallback

GradientNormMonitorCallback(
    log_every_n_steps=10,
    norm_type=2.0,
)
```

**Logged Metrics:**
- `grad/total_norm` - Total gradient norm across all parameters
- `grad/max_norm` - Maximum gradient norm for any parameter
- `grad/clipped` - Whether gradients were clipped this step

### WandbConfigLoggerCallback

Logs full YAML configuration to W&B experiment config.

```python
from lightning_reflow.callbacks import WandbConfigLoggerCallback

WandbConfigLoggerCallback(flatten=True)  # Flatten nested config for W&B UI
```

### LossRecorderCallback

Records training losses to JSON for bit-exactness testing.

```python
from lightning_reflow.callbacks import LossRecorderCallback

LossRecorderCallback(
    output_path="losses.json",
    record_interval=100,  # Record every 100 steps
    max_steps=1000,
)
```

**Output Format:**
```json
{"steps": [100, 200, 300], "losses": [0.123, 0.456, 0.789]}
```

---

## Debugging Utilities

### CrashResistantLogger

Captures all output in a rolling buffer with frequent flushing to survive crashes.

```python
from lightning_reflow.utils.debugging import CrashResistantLogger, setup_crash_resistant_logging

# Quick setup
logger = setup_crash_resistant_logging(
    log_dir="/tmp/crash_logs",
    prefix="training",
    max_buffer_lines=1000,
    auto_start=True,
)

# Or use as context manager
with CrashResistantLogger(log_dir="/tmp/logs", max_buffer_lines=500) as logger:
    # All print() statements are now captured
    print("Training started...")
    # If crash occurs, last 500 lines are preserved
```

**Output Files:**
- `training_<timestamp>_full.log` - Complete output (can grow large)
- `training_<timestamp>_last_1000.log` - Circular buffer (last N lines)
- `training_<timestamp>_metadata.txt` - Run metadata (PID, command, etc.)

### ThreadMonitor

Tracks thread count over time and warns on accumulation (useful for debugging resource leaks).

```python
from lightning_reflow.utils.debugging import ThreadMonitor

# As context manager
with ThreadMonitor(interval=30, warn_threshold=20) as monitor:
    # Training code here
    pass  # Summary printed on exit

# Or manual control
monitor = ThreadMonitor(interval=30, warn_threshold=10)
monitor.start(daemon=True)
# ... training ...
monitor.print_summary()
monitor.stop()
```

---

## ConfigMixin for Module Serialization

`ConfigMixin` solves a known issue in PyTorch Lightning where `save_hyperparameters()` doesn't work well with `nn.Module` arguments. It enables config-based serialization and reconstruction of modules with minimal boilerplate.

### Basic Usage

```python
from lightning_reflow.utils.config import ConfigMixin
import torch.nn as nn

class MyModel(ConfigMixin, nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.save_config()  # Call right after super().__init__()

        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

# Get config (JSON-serializable)
model = MyModel(hidden_dim=64, num_layers=4)
config = model.get_config()
# {'hidden_dim': 64, 'num_layers': 4, 'dropout': 0.1}

# Reconstruct from config
model2 = MyModel.from_config(config)
```

### Nested Module Support

For models that wrap other ConfigMixin modules, configs are automatically nested:

```python
class InnerModel(ConfigMixin, nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.save_config()
        self.linear = nn.Linear(dim, dim)

class OuterModel(ConfigMixin, nn.Module):
    def __init__(self, inner: nn.Module, scale: float):
        super().__init__()
        self.save_config()  # inner's config is automatically extracted
        self.inner = inner

inner = InnerModel(dim=32)
outer = OuterModel(inner=inner, scale=2.0)
config = outer.get_config()
# {
#     'inner': {
#         '__class_path__': 'mymodule.InnerModel',
#         '__config__': {'dim': 32}
#     },
#     'scale': 2.0
# }

# Full reconstruction including nested modules
outer2 = OuterModel.from_config(config)
```

### Integration with LightningModule

Use ConfigMixin with Lightning's "ignore + manual passing" pattern:

```python
class WorldModel(LightningModule):
    def __init__(self, dynamics_model: nn.Module, learning_rate: float = 1e-4):
        super().__init__()
        # Ignore the module to avoid slow pickling
        self.save_hyperparameters(ignore=['dynamics_model'])

        # Save module config separately using ConfigMixin
        if isinstance(dynamics_model, ConfigMixin):
            self.hparams['dynamics_model_config'] = {
                '__class_path__': f"{dynamics_model.__class__.__module__}.{dynamics_model.__class__.__name__}",
                '__config__': dynamics_model.get_config()
            }

        self.dynamics_model = dynamics_model
```

Then load checkpoints with automatic reconstruction:

```python
from lightning_reflow.utils.config import _deserialize_value

def load_checkpoint(path):
    ckpt = torch.load(path)
    hparams = ckpt['hyper_parameters']

    # Reconstruct module from saved config
    dynamics_config = hparams.get('dynamics_model_config')
    if dynamics_config:
        dynamics_model = _deserialize_value(dynamics_config)

    return WorldModel.load_from_checkpoint(path, dynamics_model=dynamics_model)
```

### API Reference

- `save_config(ignore=None)`: Capture `__init__` args. Call right after `super().__init__()`
- `get_config()`: Return saved config dict (JSON-serializable)
- `from_config(config)`: Class method to reconstruct module from config
- `_deserialize_value(value)`: Utility to deserialize nested configs
- `_import_class(class_path)`: Utility to import class from fully qualified path

---

## YAML Configuration Example

```yaml
seed_everything: 42

trainer:
  max_epochs: 2000
  accelerator: auto
  precision: "16-mixed"
  gradient_clip_val: 1.0
  callbacks:
    - class_path: lightning_reflow.callbacks.PauseCallback
      init_args:
        checkpoint_dir: pause_checkpoints
        enable_pause: true
        pause_key: "p"
    - class_path: lightning_reflow.callbacks.GradientNormMonitorCallback
      init_args:
        log_every_n_steps: 10
    - class_path: lightning_reflow.callbacks.WandbConfigLoggerCallback
      init_args:
        flatten: true

model:
  class_path: myproject.models.MyModel
  init_args:
    learning_rate: 1e-4

data:
  class_path: myproject.data.MyDataModule
  init_args:
    batch_size: 32
```

---

## Integration Example (World Model Training)

Here's a complete example based on the ProtoWorld project:

```python
# scripts/train_world_model.py
from lightning_reflow import LightningReflowCLI
from world_model.models.world_model import WorldModel
from world_model.data.datamodule import LeRobotDataModule

def main():
    cli = LightningReflowCLI(
        WorldModel,
        LeRobotDataModule,
        auto_configure_optimizers=False,  # Model configures its own optimizer
        seed_everything_default=42,
        subclass_mode_model=True,
        subclass_mode_data=True,
        run=True,
    )

if __name__ == "__main__":
    main()
```

```bash
# Training
python scripts/train_world_model.py fit --config configs/world_model.yaml

# Resume from pause
python scripts/train_world_model.py resume --checkpoint-path pause_checkpoints/model.ckpt

# Resume from W&B artifact
python scripts/train_world_model.py resume --checkpoint-artifact user/project/run-pause:latest
```

---

## Notes

- Pause/resume via `PauseCallback`; W&B integration optional
- CLI offers `resume` subcommand for checkpoint/artifact sources
- Designed to be minimally invasive: use callbacks or the CLI
- All callbacks can be configured via YAML or instantiated directly
