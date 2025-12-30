"""
EarlyPauseCallback - Early stopping that triggers pause instead of termination.

This callback extends PyTorch Lightning's EarlyStopping to trigger a pause
checkpoint instead of stopping training. This allows resuming training later
if desired, which is useful for:
- Long-running experiments that may plateau temporarily
- Hyperparameter tuning where you want to save promising checkpoints
- Experiments where you want human review before termination
"""

from typing import Any, Dict, Optional

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping

from .pause_callback import PauseCallback


class EarlyPauseCallback(EarlyStopping):
    """
    EarlyStopping variant that triggers pause instead of termination.

    When the monitored metric stops improving beyond patience, instead of
    stopping training, this callback schedules a pause checkpoint with
    optional W&B upload. This allows resuming training later if desired.

    Uses PauseCallback's public API (`request_pause()`) for clean integration.

    Args:
        monitor: Metric to monitor (e.g., "val_loss", "val_accuracy")
        patience: Number of checks with no improvement before triggering pause
        mode: "min" for metrics to minimize, "max" for metrics to maximize
        min_delta: Minimum change to qualify as an improvement
        upload_to_wandb: Whether to upload checkpoint to W&B when pausing
        reset_patience_on_resume: Reset wait_count to 0 on resume (default: True)
        verbose: Enable verbose logging
        strict: Raise error if metric not found (default: True)
        check_finite: Stop if metric becomes NaN or infinite
        check_on_train_epoch_end: Check on train epoch end instead of validation end

    Example:
        callbacks = [
            PauseCallback(checkpoint_dir="checkpoints"),
            EarlyPauseCallback(
                monitor="val_loss",
                patience=10,
                mode="min",
                upload_to_wandb=True,
            ),
        ]
        trainer = Trainer(callbacks=callbacks)

    Note:
        - Requires PauseCallback to be present in trainer callbacks
        - The pause occurs at the next validation boundary
        - Training can be resumed using the pause checkpoint
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 3,
        mode: str = "min",
        min_delta: float = 0.0,
        upload_to_wandb: bool = True,
        reset_patience_on_resume: bool = True,
        verbose: bool = True,
        strict: bool = True,
        check_finite: bool = True,
        check_on_train_epoch_end: bool | None = None,
        log_rank_zero_only: bool = False,
    ):
        super().__init__(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            mode=mode,
            verbose=verbose,
            strict=strict,
            check_finite=check_finite,
            stopping_threshold=None,  # We handle stopping ourselves
            divergence_threshold=None,  # Let divergence stop training (serious problem)
            check_on_train_epoch_end=check_on_train_epoch_end,
            log_rank_zero_only=log_rank_zero_only,
        )
        self.upload_to_wandb = upload_to_wandb
        self.reset_patience_on_resume = reset_patience_on_resume
        self._pause_callback: Optional[PauseCallback] = None
        self._pause_triggered = False

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Find PauseCallback in trainer's callbacks."""
        super().setup(trainer, pl_module, stage)
        self._find_pause_callback(trainer)

    def _find_pause_callback(self, trainer: Trainer) -> None:
        """Find and store reference to PauseCallback."""
        for callback in trainer.callbacks:
            if isinstance(callback, PauseCallback):
                self._pause_callback = callback
                return

        raise RuntimeError(
            "EarlyPauseCallback requires PauseCallback to be present. "
            "Add PauseCallback to trainer callbacks before EarlyPauseCallback."
        )

    def _run_early_stopping_check(self, trainer: Trainer) -> None:
        """Override to trigger pause instead of stop."""
        logs = trainer.callback_metrics

        if self.monitor not in logs:
            if self.strict:
                # Parent class handles this - let it raise/warn
                super()._run_early_stopping_check(trainer)
            return

        current = logs[self.monitor].squeeze()

        # Check for non-finite values
        if self.check_finite and not torch.isfinite(current):
            # Let training stop for NaN/Inf - this is a serious problem
            if self.verbose:
                print(f"\n⚠️ Metric {self.monitor} is {current}. Stopping training.")
            trainer.should_stop = True
            return

        should_stop, reason = self._evaluate_stopping_criteria(current)

        if should_stop and not self._pause_triggered:
            self._pause_triggered = True
            self.stopped_epoch = trainer.current_epoch
            self._trigger_pause(trainer, reason)

    def _trigger_pause(self, trainer: Trainer, reason: str | None) -> None:
        """Schedule a pause checkpoint via PauseCallback's public API."""
        if self._pause_callback is None:
            print("⚠️ EarlyPauseCallback: PauseCallback not found, falling back to stop")
            trainer.should_stop = True
            return

        # Build descriptive reason
        full_reason = f"Early stopping at epoch {trainer.current_epoch}"
        if reason:
            full_reason += f": {reason}"

        # Use PauseCallback's public API
        self._pause_callback.request_pause(
            upload=self.upload_to_wandb,
            reason=full_reason,
        )

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Reset pause trigger on new training run."""
        super().on_train_start(trainer, pl_module)
        self._pause_triggered = False

    def state_dict(self) -> Dict[str, Any]:
        """Save callback state for checkpointing."""
        state = super().state_dict()
        state["pause_triggered"] = self._pause_triggered
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load callback state from checkpoint."""
        super().load_state_dict(state_dict)

        # Reset pause_triggered to allow new pause on resume
        self._pause_triggered = False

        # Optionally reset patience to give model fresh chance
        if self.reset_patience_on_resume:
            self.wait_count = 0
            if self.verbose:
                print(f"🔄 EarlyPauseCallback: Reset patience (wait_count=0) on resume")
