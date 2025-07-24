import lightning.pytorch as pl
import torch
import warnings
from typing import Any, Dict, List, Optional

class StepOutputLoggerCallback(pl.Callback):
    """
    Logs scalar metrics returned in a dictionary by training_step,
    validation_step, and test_step.
    """
    def __init__(
        self,
        train_prog_bar_metrics: Optional[List[str]] = None,
        val_prog_bar_metrics: Optional[List[str]] = None,
        test_prog_bar_metrics: Optional[List[str]] = None,
    ):
        super().__init__()
        self.train_prog_bar_metrics = train_prog_bar_metrics if train_prog_bar_metrics is not None else []
        self.val_prog_bar_metrics = val_prog_bar_metrics if val_prog_bar_metrics is not None else []
        self.test_prog_bar_metrics = test_prog_bar_metrics if test_prog_bar_metrics is not None else []
        
        self.train_prog_bar_metrics = list(set(self.train_prog_bar_metrics + ['loss']))
        self.val_prog_bar_metrics = list(set(self.val_prog_bar_metrics + ['val_loss']))

    def _get_batch_size(self, batch: Any) -> int:
        if isinstance(batch, torch.Tensor):
            return batch.shape[0]
        elif isinstance(batch, (list, tuple)) and len(batch) > 0 and isinstance(batch[0], torch.Tensor):
            return batch[0].shape[0]
        elif isinstance(batch, dict):
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    return value.shape[0]
        return 1

    def _log_metrics_from_dict(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Dict[str, Any],
        batch: Any,
        prefix: str,
        on_step: bool,
        on_epoch: bool,
        prog_bar_metrics: List[str],
    ):
        if not isinstance(outputs, dict):
            warnings.warn(f"StepOutputLoggerCallback: Expected 'outputs' to be a dict, got {type(outputs)}. Skipping logging for this step.", UserWarning)
            return

        batch_size = self._get_batch_size(batch)

        for key, value in outputs.items():
            metric_to_log: Optional[torch.Tensor] = None

            if isinstance(value, (int, float)):
                # Create tensor on CPU to avoid GPU memory accumulation
                metric_to_log = torch.tensor(float(value), device='cpu')
            elif isinstance(value, torch.Tensor):
                if value.numel() == 1 and not value.is_complex():
                    # Detach and move to CPU to prevent GPU memory accumulation
                    metric_to_log = value.detach().cpu().float()

            if metric_to_log is not None:
                # Scale loss metrics to account for Lightning's automatic normalization
                # Lightning normalizes losses by accumulate_grad_batches internally
                if key == 'loss' and prefix == 'train' and trainer.accumulate_grad_batches > 1:
                    metric_to_log = metric_to_log * trainer.accumulate_grad_batches
                
                # If key already contains a prefix (e.g., "val/val_loss"), use it as-is
                # Otherwise, add the prefix (e.g., "loss" -> "train/loss")
                if "/" in key:
                    log_key = key
                else:
                    log_key = f"{prefix}/{key}"
                prog_bar = key in prog_bar_metrics
                
                try:
                    pl_module.log(
                        log_key,
                        metric_to_log,
                        on_step=on_step,
                        on_epoch=on_epoch,
                        prog_bar=prog_bar,
                        logger=True,
                        batch_size=batch_size,
                        sync_dist=True, 
                    )
                except Exception as e:
                    warnings.warn(f"StepOutputLoggerCallback: Error logging metric '{log_key}': {e}", UserWarning)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[Dict[str, Any]], 
        batch: Any,
        batch_idx: int,
    ):
        if outputs:
            self._log_metrics_from_dict(
                trainer,
                pl_module,
                outputs,
                batch,
                prefix="train",
                on_step=True, 
                on_epoch=False, 
                prog_bar_metrics=self.train_prog_bar_metrics,
            )

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[Dict[str, Any]], 
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        if outputs:
            effective_prefix = "val"
            if dataloader_idx > 0 and trainer.num_val_dataloaders > 1 :
                 effective_prefix = f"val_dl_{dataloader_idx}"

            self._log_metrics_from_dict(
                trainer,
                pl_module,
                outputs,
                batch,
                prefix=effective_prefix, 
                on_step=False, 
                on_epoch=True,
                prog_bar_metrics=self.val_prog_bar_metrics,
            )

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[Dict[str, Any]], 
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        if outputs:
            effective_prefix = "test"
            if dataloader_idx > 0 and trainer.num_test_dataloaders > 1:
                effective_prefix = f"test_dl_{dataloader_idx}"

            self._log_metrics_from_dict(
                trainer,
                pl_module,
                outputs,
                batch,
                prefix=effective_prefix,
                on_step=False,
                on_epoch=True,
                prog_bar_metrics=self.test_prog_bar_metrics,
            ) 