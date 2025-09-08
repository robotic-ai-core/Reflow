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


## Notes

- Pause/resume via `PauseCallback`; W&B integration optional
- CLI offers `resume` subcommand for checkpoint/artifact sources
- Designed to be minimally invasive: use callbacks or the CLI