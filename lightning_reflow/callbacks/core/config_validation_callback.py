"""
Config validation callback for early detection of stale configs.

This callback validates config freshness at the start of training/resume,
failing fast before any work is done if config issues are detected.
"""

import logging
from pathlib import Path
from typing import Any

from lightning.pytorch import Trainer, LightningModule, Callback

from .config_embedding_mixin import ConfigEmbeddingMixin

logger = logging.getLogger(__name__)


class ConfigValidationCallback(Callback, ConfigEmbeddingMixin):
    """
    Validates config at training start to detect stale configs early.

    This callback runs validation during on_fit_start, before any training happens,
    to fail fast if config issues are detected. This saves time by catching problems
    immediately instead of waiting until the first checkpoint save.

    Usage:
        trainer = Trainer(
            callbacks=[ConfigValidationCallback()]
        )
    """

    def __init__(self):
        super().__init__()
        logger.info("🔍 ConfigValidationCallback initialized - will validate config at training start")

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        Validate config at the start of fit (training or resume).

        This catches stale config issues early before any training work is done.
        """
        logger.info("🔍 Validating config freshness at training start...")

        # Check if we can validate (need CLI context)
        if not self._can_embed_config(trainer):
            logger.warning("⚠️  Cannot validate config - no CLI context available")
            return

        # Perform validation (this will raise RuntimeError if config is invalid)
        try:
            lightning_log_dir = self._get_lightning_log_dir(trainer)
            if not lightning_log_dir:
                raise RuntimeError(
                    "CRITICAL: Could not find Lightning log directory at training start. "
                    "Config validation requires a valid log directory."
                )

            config_path = lightning_log_dir / "config.yaml"
            if not config_path.exists():
                raise RuntimeError(
                    f"CRITICAL: Lightning config not found at training start: {config_path}\n"
                    f"This usually means save_config_callback=None was set in LightningCLI, "
                    f"which disables config saving. Config embedding REQUIRES Lightning to save "
                    f"the config file. Remove save_config_callback=None from your CLI initialization."
                )

            # Validate config freshness
            self._validate_config_freshness(config_path, lightning_log_dir, trainer)

            logger.info(f"✅ Config validation passed: {config_path}")

        except RuntimeError as e:
            # Config validation failed - fail immediately before any training
            logger.error(f"❌ Config validation failed at training start: {e}")
            raise RuntimeError(
                f"Training aborted: config validation failed.\n\n{e}\n\n"
                f"This is a safety measure to prevent training with stale/invalid configuration "
                f"that would cause resume failures later. Fix the config issue and restart training."
            ) from e
