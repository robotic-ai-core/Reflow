"""
Loss Recorder Callback for bit-exactness testing.

This callback records training losses at specific steps to a JSON file
without introducing GPU synchronization overhead from loggers like CSVLogger.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import lightning.pytorch as pl
import torch


class LossRecorderCallback(pl.Callback):
    """
    Records training losses to a JSON file at specified step intervals.

    This callback is designed for bit-exactness testing where we need to:
    1. Extract training losses for comparison
    2. Avoid GPU synchronization overhead from traditional loggers
    3. Write to a simple, human-readable format

    Args:
        output_path: Path to write the losses JSON file
        record_interval: Record loss every N steps (default: 100)
        max_steps: Maximum step to record (default: 1000)

    The output JSON format:
    {
        "steps": [100, 200, 300, ...],
        "losses": [0.123, 0.456, 0.789, ...]
    }

    Usage in YAML config:
        callbacks:
          - class_path: lightning_reflow.callbacks.monitoring.LossRecorderCallback
            init_args:
              output_path: losses.json
              record_interval: 100
              max_steps: 1000
    """

    def __init__(
        self,
        output_path: str = "losses.json",
        record_interval: int = 100,
        max_steps: int = 1000,
    ):
        super().__init__()
        self.output_path = Path(output_path)
        self.record_interval = record_interval
        self.max_steps = max_steps
        self.recorded_steps: List[int] = []
        self.recorded_losses: List[float] = []

    def state_dict(self) -> Dict[str, Any]:
        """Save callback state for checkpointing."""
        return {
            "recorded_steps": self.recorded_steps,
            "recorded_losses": self.recorded_losses,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore callback state from checkpoint."""
        self.recorded_steps = state_dict.get("recorded_steps", [])
        self.recorded_losses = state_dict.get("recorded_losses", [])

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Record loss at specified intervals."""
        step = trainer.global_step

        # Record at intervals (100, 200, 300, ...)
        if step % self.record_interval == 0 and step > 0 and step <= self.max_steps:
            # Extract loss from outputs
            if isinstance(outputs, dict):
                loss = outputs.get("loss", None)
            elif isinstance(outputs, torch.Tensor):
                loss = outputs
            else:
                loss = None

            if loss is not None:
                loss_value = loss.item()
                self.recorded_steps.append(step)
                self.recorded_losses.append(loss_value)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Write recorded losses to JSON file."""
        data = {"steps": self.recorded_steps, "losses": self.recorded_losses}

        # Resolve output path relative to trainer's default_root_dir if it's relative
        output_path = self.output_path
        if not output_path.is_absolute() and trainer.default_root_dir:
            output_path = Path(trainer.default_root_dir) / output_path

        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to JSON
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Loss recorder: Saved {len(self.recorded_steps)} losses to {output_path}")
