# LightningReflow

A PyTorch Lightning extension framework providing advanced training capabilities including pause/resume functionality, W&B checkpoint saving, and an enhanced CLI.

Dual progress bars:
```bash
Global Steps:  15%|██████████▌                                                           | 6395/42150 [03:32<19:45, 30.16it/s, lr-AdamW=0.001000]
Interval 2 (Steps to Val) - Press 'p' to pause:  52%|███████████████████▏                 | 2181/4215 [00:47<00:44, 45.73it/s, train/loss=0.2049]
```

Prints out resume options:
```bash
✅ Pause checkpoint uploaded to W&B successfully
✅ Pause checkpoint uploaded to W&B: neiltan/VibeDiffusion/auannr4y-pause:latest
🔄 Training paused successfully at validation boundary

🔄 Training paused. Resume options:
📁 Local resume:    python train_lightning.py resume --checkpoint-path pause_checkpoints/upload_epoch=19_step=16860_1757340895.ckpt
☁️  W&B resume:     python train_lightning.py resume --checkpoint-artifact neiltan/VibeDiffusion/auannr4y-pause:latest
```

## TL;DR (Quickstart)

```bash
git clone <repo>
cd external/LightningReflow
pip install -e .
```

```python
import lightning.pytorch as pl
from lightning_reflow.callbacks import PauseCallback, FlowProgressBarCallback

trainer = pl.Trainer(
    callbacks=[
        PauseCallback(checkpoint_dir="checkpoints", enable_pause=True),
        FlowProgressBarCallback()
    ]
)
trainer.fit(model, datamodule)
```


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

## Notes

- Pause/resume via `PauseCallback`; W&B integration optional
- CLI offers `resume` subcommand for checkpoint/artifact sources
- Designed to be minimally invasive: use callbacks or the CLI