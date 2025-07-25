import lightning.pytorch as pl
import warnings
from lightning.pytorch.utilities import rank_zero_only
from lightning_reflow.utils.logging.logging_config import get_logger

class WandbWatchCallback(pl.Callback):
    """Handles the setup of wandb.watch() for logging model gradients and parameters."""
    def __init__(self, log_level: str = "gradients", log_freq: int = 1000, log_graph: bool = False):
        super().__init__()
        self.log_level = log_level
        self.log_freq = log_freq
        self.log_graph = log_graph
        self.logger = get_logger(__name__)
        if log_graph:
            warnings.warn(
                "wandb.watch(log_graph=True) can sometimes cause issues with torch.compile. "
                "It is disabled by default in this callback. If you encounter issues, "
                "consider setting log_graph=False.", UserWarning
            )

    @rank_zero_only  
    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if trainer.logger and hasattr(trainer.logger.experiment, 'watch'):
            try:
                # By on_train_start, model compilation (if enabled) has already happened
                # Check if the model is actually compiled by looking for _orig_mod attribute
                is_compiled = hasattr(pl_module.backbone, '_orig_mod') or hasattr(pl_module, '_orig_mod')
                
                # Adjust logging level for compiled models to avoid backward hook issues
                if is_compiled and self.log_level == "gradients":
                    self.logger.info("WandbWatchCallback: Compiled model detected - using 'parameters' logging to avoid backward hook warnings")
                    actual_log_level = "parameters"  # Still log parameters, skip gradients
                else:
                    actual_log_level = self.log_level
                
                trainer.logger.experiment.watch(
                    pl_module, 
                    log=actual_log_level, 
                    log_freq=self.log_freq, 
                    log_graph=self.log_graph
                )
                self.logger.info("WandbWatchCallback: Watching model with log='%s', log_freq=%d, log_graph=%s", 
                                 actual_log_level, self.log_freq, self.log_graph)
            except Exception as e:
                warnings.warn(f"WandbWatchCallback: Failed to call wandb.watch: {e}", UserWarning)
        else:
            warnings.warn("WandbWatchCallback: Trainer.logger or logger.experiment does not support .watch().", UserWarning) 