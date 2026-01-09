"""
Gradient norm monitoring callback for PyTorch Lightning.

Monitors gradient norms during training to verify gradient clipping behavior.
Helps distinguish between:
- Healthy training (occasional clipping of outliers)
- Hyperparameter issues (constant clipping every step)
"""

import logging
from typing import Any, Dict

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback

logger = logging.getLogger(__name__)


class GradientNormMonitorCallback(Callback):
    """
    Monitor gradient norms before optimizer step.

    Logs:
    - train/grad_norm_unclipped: Actual gradient norm before clipping
    - train/grad_norm_clipped: Whether clipping was triggered (1.0 if yes, 0.0 if no)
    - train/grad_clip_ratio: Fraction of steps where clipping triggered (running average)

    Interpreting Results:
    - grad_norm_unclipped << clip_val most steps: Healthy (clipping rarely needed)
    - grad_norm_unclipped > clip_val occasionally: Good (catching outliers)
    - grad_clip_ratio < 0.05 (5%): Optimal
    - grad_clip_ratio > 0.5 (50%): Hyperparameters likely wrong

    Args:
        log_every_n_steps: Log gradient norms every N steps.
            Default: 1 (every step)
        print_epoch_summary: Whether to print a summary at end of each epoch.
            Default: True

    Usage in YAML config:
        callbacks:
          - class_path: lightning_reflow.callbacks.monitoring.GradientNormMonitorCallback
            init_args:
              log_every_n_steps: 10
    """

    def __init__(
        self,
        log_every_n_steps: int = 1,
        print_epoch_summary: bool = True,
    ):
        """
        Initialize gradient norm monitor.

        Args:
            log_every_n_steps: Log gradient norms every N steps (default: 1)
            print_epoch_summary: Print summary at end of each epoch (default: True)
        """
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.print_epoch_summary = print_epoch_summary
        self._clip_count = 0
        self._total_steps = 0

    def state_dict(self) -> Dict[str, Any]:
        """Save callback state for checkpointing."""
        return {
            "clip_count": self._clip_count,
            "total_steps": self._total_steps,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore callback state from checkpoint."""
        self._clip_count = state_dict.get("clip_count", 0)
        self._total_steps = state_dict.get("total_steps", 0)

    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer,
        *args,
        **kwargs,
    ) -> None:
        """
        Called before optimizer.step().

        Measures gradient norm before any clipping is applied by the trainer.
        """
        # Only log every N steps to reduce overhead
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        # Calculate global gradient norm before clipping
        # Use same method as PyTorch's clip_grad_norm_
        parameters = [p for p in pl_module.parameters() if p.grad is not None]

        if len(parameters) == 0:
            return

        # Compute total norm (same as torch.nn.utils.clip_grad_norm_ with max_norm=inf)
        device = parameters[0].grad.device
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), 2.0).to(device) for p in parameters]),
            2.0
        )

        # Get clip value from trainer config
        clip_val = trainer.gradient_clip_val or float('inf')

        # Track clipping statistics
        self._total_steps += 1
        is_clipped = float(total_norm > clip_val)
        self._clip_count += is_clipped
        clip_ratio = self._clip_count / self._total_steps

        # Log metrics
        pl_module.log(
            "train/grad_norm_unclipped",
            total_norm,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )

        pl_module.log(
            "train/grad_norm_clipped",
            is_clipped,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )

        pl_module.log(
            "train/grad_clip_ratio",
            clip_ratio,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Print summary at end of each epoch."""
        if not self.print_epoch_summary:
            return

        if self._total_steps > 0:
            clip_ratio = self._clip_count / self._total_steps
            clip_pct = clip_ratio * 100

            if clip_pct < 5:
                status = "HEALTHY"
            elif clip_pct < 20:
                status = "MODERATE"
            else:
                status = "EXCESSIVE"

            logger.info(
                f"Gradient Clipping Summary (Epoch {trainer.current_epoch}): "
                f"{status} - {clip_pct:.1f}% of steps clipped "
                f"({int(self._clip_count)}/{self._total_steps})"
            )
