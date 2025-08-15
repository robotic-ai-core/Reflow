# LightningReflow

A PyTorch Lightning extension framework providing advanced training capabilities including pause/resume functionality, W&B integration, and an enhanced CLI.

## Overview

LightningReflow extends PyTorch Lightning with production-ready features for ML training workflows:

- **Advanced Pause/Resume**: Interrupt and resume training with complete state preservation
- **W&B Integration**: Deep Weights & Biases integration with artifact management and run continuity
- **Enhanced CLI**: Extended Lightning CLI with resume commands and config management
- **Rich Callbacks**: Comprehensive callback system for monitoring, environment management, and training control
- **Config Embedding**: Checkpoint-embedded configurations for reproducible training

## Installation

### As a Git Submodule

```bash
# Add as submodule
git submodule add https://github.com/neil-tan/LightningReflow.git lib/lightning_reflow

# Install in editable mode
pip install -e lib/lightning_reflow/
```

### Direct Installation

```bash
# Clone and install
git clone https://github.com/neil-tan/LightningReflow.git
cd LightningReflow
pip install -e .
```

## Project Structure

```
lightning_reflow/
├── callbacks/          # Training callbacks
│   ├── core/          # Core callbacks (config embedding, memory cleanup)
│   ├── environment/   # Environment variable management
│   ├── monitoring/    # Training monitoring (progress bars, metrics)
│   ├── pause/         # Pause/resume functionality
│   ├── wandb/         # W&B integration (WandbArtifactCheckpoint)
│   └── __init__.py    # Callback exports
├── cli/               # Enhanced Lightning CLI
│   ├── __init__.py
│   ├── __main__.py    # CLI entry point
│   └── lightning_cli.py
├── core/              # Core framework components
│   └── lightning_reflow.py
├── strategies/        # Resume strategies
│   └── wandb_artifact_resume_strategy.py
├── utils/             # Utilities
│   ├── checkpoint/    # Checkpoint management
│   ├── config/        # Configuration utilities
│   └── wandb/         # W&B utilities
└── __init__.py        # Package exports
```

## Quick Start

### Basic Usage in Your Project

```python
import lightning.pytorch as pl
from lightning_reflow import LightningReflow
from lightning_reflow.callbacks import (
    PauseCallback,
    FlowProgressBarCallback,
    EnvironmentCallback
)

# Using with your own model and data
model = YourLightningModule()
data = YourDataModule()

# Option 1: Use callbacks directly with PyTorch Lightning
trainer = pl.Trainer(
    max_epochs=100,
    callbacks=[
        PauseCallback(
            checkpoint_dir="pause_checkpoints",
            enable_pause=True,
            pause_key='p'
        ),
        FlowProgressBarCallback(
            refresh_rate=1,
            global_bar_metrics=['loss', 'val_loss'],
            interval_bar_metrics=['lr-*']
        ),
        EnvironmentCallback(
            env_vars={
                'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:128',
                'CUDA_VISIBLE_DEVICES': '0,1'
            }
        )
    ]
)
trainer.fit(model, data)

# Option 2: Use LightningReflow for enhanced functionality
reflow = LightningReflow()
reflow.fit(config="path/to/config.yaml")
```

### CLI Usage

```bash
# Training with config file
python -m lightning_reflow.cli fit --config config.yaml

# Training with CLI overrides
python -m lightning_reflow.cli fit --config config.yaml --trainer.max_epochs=100 --model.learning_rate=0.001

# Resume from local checkpoint
python -m lightning_reflow.cli resume --checkpoint-path /path/to/checkpoint.ckpt

# Resume from W&B artifact
python -m lightning_reflow.cli resume --checkpoint-artifact entity/project/run-id:latest

# Resume with config overrides
python -m lightning_reflow.cli resume --checkpoint-path checkpoint.ckpt --trainer.max_epochs=200
```

### Configuration Files

```yaml
# config.yaml
model:
  class_path: your_project.models.YourModel
  init_args:
    input_dim: 784
    hidden_dim: 256
    output_dim: 10
    learning_rate: 0.001

data:
  class_path: your_project.data.YourDataModule
  init_args:
    batch_size: 64
    num_workers: 4

trainer:
  max_epochs: 100
  accelerator: gpu
  devices: 2
  callbacks:
    - class_path: lightning_reflow.callbacks.PauseCallback
      init_args:
        checkpoint_dir: pause_checkpoints
        enable_pause: true
        pause_key: p
        upload_key: w
    - class_path: lightning_reflow.callbacks.FlowProgressBarCallback
      init_args:
        refresh_rate: 1
        global_bar_metrics: ['loss', 'val_loss']
    - class_path: lightning_reflow.callbacks.EnvironmentCallback
      init_args:
        env_vars:
          PYTORCH_CUDA_ALLOC_CONF: "max_split_size_mb:128"
  logger:
    class_path: lightning.pytorch.loggers.WandbLogger
    init_args:
      project: my-project
      save_dir: wandb_logs
```

## Key Components

### 1. PauseCallback

Interactive pause/resume during training:

```python
from lightning_reflow.callbacks import PauseCallback

callback = PauseCallback(
    checkpoint_dir="pause_checkpoints",
    enable_pause=True,
    pause_key='p',           # Press 'p' to pause
    upload_key='w',          # Press 'w' to upload to W&B
    refresh_rate=1,          # Update frequency in seconds
    bar_colour='#fcac17'     # Custom progress bar color
)
```

**Features:**
- Real-time keyboard control during training
- Checkpoint saving with embedded configuration
- W&B artifact upload with resume commands
- Complete state preservation (RNG, optimizer, etc.)

### 2. FlowProgressBarCallback

Enhanced progress bars with metric tracking:

```python
from lightning_reflow.callbacks import FlowProgressBarCallback

callback = FlowProgressBarCallback(
    refresh_rate=1,
    global_bar_metrics=['loss', 'val_loss', 'epoch'],
    interval_bar_metrics=['lr-*', '*/loss'],  # Glob patterns supported
    leave=True
)
```

**Features:**
- Dual progress bars (global/interval)
- Pattern matching for metrics
- Efficient caching for performance
- Correct interval calculation for validation

### 3. EnvironmentCallback

Environment variable management:

```python
from lightning_reflow.callbacks import EnvironmentCallback

callback = EnvironmentCallback(
    env_vars={
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:128,expandable_segments:True',
        'CUDA_VISIBLE_DEVICES': '0,1',
        'TOKENIZERS_PARALLELISM': 'false'
    },
    merge_strategy='override'  # or 'merge'
)
```

**Features:**
- Environment setup before training
- State preservation in checkpoints
- Config precedence handling (config > checkpoint > system)
- Validation and conflict resolution

### 4. WandbArtifactCheckpoint

Advanced checkpoint management with W&B artifact uploads:

```python
from lightning_reflow.callbacks.wandb import WandbArtifactCheckpoint

callback = WandbArtifactCheckpoint(
    upload_best_model=True,
    upload_last_model=True,
    upload_every_n_epoch=1,
    monitor_pause_checkpoints=True,
    use_compression=True,
    create_emergency_checkpoints=True
)
```

**Features:**
- Automatic checkpoint upload to W&B as artifacts
- Pause checkpoint monitoring and upload
- Emergency checkpoint creation on exceptions
- Configurable compression and upload strategies
- Seamless integration with PauseCallback

### 5. Enhanced CLI

Extended Lightning CLI with resume capabilities:

```python
from lightning_reflow.cli import LightningReflowCLI

# Direct usage
cli = LightningReflowCLI(
    model_class=YourModel,
    datamodule_class=YourDataModule,
    save_config_callback=None  # Auto-handled
)
```

**Features:**
- `resume` subcommand for checkpoint/artifact resumption
- Config embedding and extraction
- Advanced override merging
- W&B artifact download and management

## API Usage Examples

### Using Callbacks in Your Training Script

```python
import lightning.pytorch as pl
from lightning_reflow.callbacks import PauseCallback, FlowProgressBarCallback

class YourModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # Your model definition
    
    def training_step(self, batch, batch_idx):
        # Your training logic
        loss = self.compute_loss(batch)
        self.log('loss', loss)
        return loss

# Create trainer with LightningReflow callbacks
trainer = pl.Trainer(
    max_epochs=100,
    callbacks=[
        PauseCallback(
            checkpoint_dir="checkpoints/pause",
            enable_pause=True
        ),
        FlowProgressBarCallback(
            global_bar_metrics=['loss', 'val_loss']
        )
    ]
)

# Train - press 'p' to pause anytime
trainer.fit(model, dataloader)
```

### Programmatic Resume

```python
from lightning_reflow.core import LightningReflow
from lightning_reflow.utils.checkpoint import extract_embedded_config

# Resume from checkpoint with embedded config
reflow = LightningReflow()
result = reflow.resume(
    resume_source="path/to/checkpoint.ckpt",
    additional_config={"trainer": {"max_epochs": 200}}
)

# Or extract config for inspection
checkpoint = torch.load("checkpoint.ckpt")
embedded_config = extract_embedded_config(checkpoint)
print(embedded_config)  # View the training configuration
```

### Custom Callback Integration

```python
from lightning_reflow.callbacks.core import CallbackConfigMixin

class YourCustomCallback(pl.Callback, CallbackConfigMixin):
    def __init__(self, param1: str, param2: int = 10):
        super().__init__()
        self.param1 = param1
        self.param2 = param2
    
    def on_train_start(self, trainer, pl_module):
        # Your callback logic
        pass
    
    def state_dict(self):
        # Will be saved in checkpoint
        return {"param1": self.param1, "param2": self.param2}
    
    def load_state_dict(self, state_dict):
        # Will be restored on resume
        self.param1 = state_dict.get("param1", self.param1)
        self.param2 = state_dict.get("param2", self.param2)
```

## Common Use Cases

### 1. Long-Running Training with Interruption Support

```yaml
# long_training_config.yaml
trainer:
  max_epochs: 1000
  callbacks:
    - class_path: lightning_reflow.callbacks.PauseCallback
      init_args:
        checkpoint_dir: pause_checkpoints
        enable_pause: true
        pause_on_batch_end: true  # Can pause mid-epoch
```

### 2. Multi-GPU Training with Environment Setup

```python
from lightning_reflow.callbacks import EnvironmentCallback

callback = EnvironmentCallback(
    env_vars={
        'CUDA_VISIBLE_DEVICES': '0,1,2,3',
        'NCCL_DEBUG': 'INFO',
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512'
    }
)
```

### 3. W&B Integration with Artifact Tracking

```python
trainer = pl.Trainer(
    logger=pl.loggers.WandbLogger(
        project="my-project",
        save_dir="logs"
    ),
    callbacks=[
        PauseCallback(
            checkpoint_dir="checkpoints",
            enable_pause=True,
            upload_to_wandb=True  # Auto-upload on pause
        )
    ]
)
```

## Architecture & Design

### Design Principles

1. **Modular**: Each component can be used independently
2. **Non-invasive**: Works with existing PyTorch Lightning code
3. **State-preserving**: Full training state recovery on resume
4. **Config-driven**: YAML-based configuration with CLI overrides

### Integration Points

- **PyTorch Lightning**: Extends pl.Callback, pl.Trainer, LightningCLI
- **Weights & Biases**: Optional deep integration for experiment tracking
- **Checkpoint Format**: Compatible with standard Lightning checkpoints
- **CLI Interface**: Drop-in replacement for LightningCLI

## Important Notes

### PyTorch Lightning Unit Inconsistency

⚠️ **Critical**: PyTorch Lightning has an inconsistency in how it counts steps:

- **`max_steps`**: Counts **optimization steps** (gradient updates after `optimizer.step()`)
- **`val_check_interval`** (when integer): Counts **training batches** (forward passes)

This means with `accumulate_grad_batches=16`:
- `val_check_interval: 1600` = validation every 1600 forward passes = 100 optimization steps
- To validate every 1600 optimization steps, use `val_check_interval: 25600` (1600 × 16)

This inconsistency is documented in:
- [PyTorch Lightning Discussion #12220](https://github.com/Lightning-AI/pytorch-lightning/discussions/12220)
- [PyTorch Lightning Issue #17207](https://github.com/Lightning-AI/pytorch-lightning/issues/17207)

The FlowProgressBarCallback in LightningReflow correctly handles this by showing:
- **Global progress bar**: Optimization steps (matches `max_steps`)
- **Interval progress bar**: Training batches (matches `val_check_interval`)

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure package is installed
   pip install -e /path/to/lightning_reflow
   ```

2. **Pause Not Working**
   - Ensure terminal supports keyboard input
   - Check `enable_pause=True` in PauseCallback
   - Verify callback is in trainer.callbacks list

3. **W&B Resume Issues**
   - Ensure W&B is logged in: `wandb login`
   - Check artifact permissions
   - Verify run ID extraction from checkpoint

4. **Environment Variables Not Set**
   - Check EnvironmentCallback is early in callback list
   - Verify no conflicting environment settings
   - Check config precedence (config > checkpoint > system)

## Dependencies

- PyTorch Lightning >= 2.0
- PyTorch >= 2.0
- wandb (optional, for W&B integration)
- pyyaml
- tqdm
- jsonargparse

## License

See LICENSE file in the repository root.

## Contributing

Contributions welcome! Please ensure:
- Code follows existing patterns
- Tests are added for new features
- Documentation is updated
- All tests pass before submitting PR