"""
Callback to log the full YAML config to Weights & Biases.

The standard PyTorch Lightning workflow only logs hyperparameters captured via
`save_hyperparameters()` in the model's `__init__`. This misses:
- Trainer configuration (max_epochs, gradient_clip_val, precision, etc.)
- Data configuration (batch_size, context_length, auto_steps, augmentation settings)
- Callback configurations
- Nested model configurations (like dynamics_model internals)

This callback reads the full config from the CLI and logs it to wandb's
config section, making all hyperparameters searchable and filterable in the
W&B dashboard.
"""

import logging
from typing import Any, Dict, List, Optional

import lightning.pytorch as pl
from lightning.pytorch.utilities import rank_zero_only

logger = logging.getLogger(__name__)


def _flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '/') -> Dict[str, Any]:
    """
    Flatten a nested dictionary into a single-level dict with path keys.

    Args:
        d: Dictionary to flatten
        parent_key: Prefix for keys (used in recursion)
        sep: Separator between nested keys

    Returns:
        Flattened dictionary with keys like "model/dynamics_model/latent_dim"
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep).items())
        elif isinstance(v, list):
            # Handle lists - convert to string representation for logging
            # but also try to expand if it's a list of dicts
            if v and isinstance(v[0], dict):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        items.extend(_flatten_dict(item, f"{new_key}/{i}", sep).items())
                    else:
                        items.append((f"{new_key}/{i}", item))
            else:
                items.append((new_key, v))
        else:
            items.append((new_key, v))
    return dict(items)


class WandbConfigLoggerCallback(pl.Callback):
    """
    Callback to log the full YAML configuration to W&B.

    This callback captures the complete training configuration from the
    LightningCLI and logs it to W&B's config section at the start of training.

    Configuration is logged in two formats:
    1. Flattened (default): Nested keys become paths like "model/learning_rate"
    2. Raw (optional): Preserve nested structure using W&B's nested config support

    Usage in YAML config:
        callbacks:
          - class_path: lightning_reflow.callbacks.logging.WandbConfigLoggerCallback
            init_args:
              flatten: true  # Flatten nested config (recommended for searchability)
              exclude_keys: ["trainer/callbacks"]  # Optional: exclude verbose sections
    """

    def __init__(
        self,
        flatten: bool = True,
        exclude_keys: Optional[List[str]] = None,
        log_model_hparams: bool = True,
        log_datamodule_hparams: bool = True,
    ):
        """
        Initialize the config logger callback.

        Args:
            flatten: If True, flatten nested config into path-style keys.
                    Recommended for better W&B searchability.
            exclude_keys: List of key paths to exclude from logging.
                         Useful for excluding verbose sections like callbacks.
            log_model_hparams: Also log model.hparams (from save_hyperparameters)
            log_datamodule_hparams: Also log datamodule hparams if available
        """
        super().__init__()
        self.flatten = flatten
        self.exclude_keys = exclude_keys or []
        self.log_model_hparams = log_model_hparams
        self.log_datamodule_hparams = log_datamodule_hparams

    def _get_wandb_logger(self, trainer: pl.Trainer):
        """Get the W&B logger from trainer if available."""
        if trainer.logger is None:
            return None

        # Handle single logger
        if hasattr(trainer.logger, 'experiment') and hasattr(trainer.logger.experiment, 'config'):
            return trainer.logger

        # Handle logger collection
        if hasattr(trainer.logger, 'experiment'):
            for exp_logger in getattr(trainer.logger, '_loggers', [trainer.logger]):
                if hasattr(exp_logger, 'experiment') and hasattr(exp_logger.experiment, 'config'):
                    return exp_logger

        return None

    def _get_cli_config(self, trainer: pl.Trainer) -> Optional[Dict[str, Any]]:
        """Extract the full config from LightningCLI if available."""
        # LightningReflowCLI stores itself as trainer.cli
        cli = getattr(trainer, 'cli', None)
        if cli is None:
            return None

        # Get the config - it's usually an omegaconf.DictConfig or dict
        config = getattr(cli, 'config', None)
        if config is None:
            return None

        # Convert to dict if needed (handles OmegaConf)
        try:
            from omegaconf import OmegaConf, DictConfig
            if isinstance(config, DictConfig):
                config = OmegaConf.to_container(config, resolve=True)
        except ImportError:
            pass

        # Handle jsonargparse Namespace (used by LightningCLI)
        if hasattr(config, 'as_dict'):
            config = config.as_dict()
        elif hasattr(config, '__dict__') and not isinstance(config, dict):
            config = vars(config)

        return config if isinstance(config, dict) else None

    def _filter_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out excluded keys from config."""
        if not self.exclude_keys:
            return config

        if self.flatten:
            # For flattened config, filter by key prefix
            return {
                k: v for k, v in config.items()
                if not any(k.startswith(ex) for ex in self.exclude_keys)
            }
        else:
            # For nested config, we'd need recursive filtering
            # For simplicity, skip complex nested filtering
            return config

    @rank_zero_only
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Log the full config to W&B at the start of training."""
        wandb_logger = self._get_wandb_logger(trainer)
        if wandb_logger is None:
            logger.warning("WandbConfigLoggerCallback: No W&B logger found, skipping config logging")
            return

        # Get wandb experiment (the actual wandb.run object)
        experiment = wandb_logger.experiment
        if experiment is None:
            logger.warning("WandbConfigLoggerCallback: W&B experiment not initialized")
            return

        # Set up x-axis to use trainer/global_step for all metrics
        # This ensures wandb plots show actual training steps, not wandb's internal step counter
        try:
            import wandb
            wandb.define_metric("trainer/global_step")
            wandb.define_metric("*", step_metric="trainer/global_step")
            logger.debug("WandbConfigLoggerCallback: Set trainer/global_step as x-axis metric")
        except Exception as e:
            logger.debug(f"Could not set up wandb x-axis metric: {e}")

        config_logged = False

        # 1. Log CLI config (full YAML config)
        cli_config = self._get_cli_config(trainer)
        if cli_config:
            if self.flatten:
                flat_config = _flatten_dict(cli_config)
                filtered_config = self._filter_config(flat_config)
            else:
                filtered_config = self._filter_config(cli_config)

            # Update wandb config
            experiment.config.update(filtered_config, allow_val_change=True)
            logger.info(f"WandbConfigLoggerCallback: Logged {len(filtered_config)} config parameters to W&B")
            config_logged = True
        else:
            logger.info("WandbConfigLoggerCallback: CLI config not available")

        # 2. Log model hyperparameters (ensure they're captured even without CLI)
        if self.log_model_hparams and hasattr(pl_module, 'hparams'):
            try:
                model_hparams = dict(pl_module.hparams)
                if self.flatten:
                    model_hparams = _flatten_dict({'model_hparams': model_hparams})
                experiment.config.update(model_hparams, allow_val_change=True)
                logger.debug("WandbConfigLoggerCallback: Logged model hparams")
            except Exception as e:
                logger.debug(f"Could not log model hparams: {e}")

        # 3. Log datamodule hyperparameters if available
        if self.log_datamodule_hparams and trainer.datamodule is not None:
            try:
                dm = trainer.datamodule
                dm_hparams = {}
                # Common datamodule attributes to log
                for attr in ['batch_size', 'context_length', 'auto_steps', 'val_ar_steps', 'num_workers', 'repo_id']:
                    if hasattr(dm, attr):
                        dm_hparams[f"datamodule/{attr}"] = getattr(dm, attr)
                # Also check for hparams
                if hasattr(dm, 'hparams'):
                    for k, v in dm.hparams.items():
                        dm_hparams[f"datamodule/{k}"] = v
                if dm_hparams:
                    experiment.config.update(dm_hparams, allow_val_change=True)
                    logger.debug("WandbConfigLoggerCallback: Logged datamodule hparams")
            except Exception as e:
                logger.debug(f"Could not log datamodule hparams: {e}")

        if config_logged:
            logger.info("WandbConfigLoggerCallback: Full config logged to W&B successfully")
