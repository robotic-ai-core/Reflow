"""
Core Lightning Reflow class for programmatic training orchestration.

This module provides the main LightningReflow class that can be used programmatically
for training, with support for configuration loading, callbacks, and resumption.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Type, Callable
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback

from .config_loader import ConfigLoader
from ..strategies import ResumeStrategy, LocalPathResumeStrategy, WandbArtifactResumeStrategy
from ..utils.logging.logging_config import configure_logging

logger = logging.getLogger(__name__)


class LightningReflow:
    """
    Core Lightning Reflow class for programmatic training orchestration.
    
    This class provides a clean, programmatic interface for training PyTorch Lightning
    models with advanced features like pause/resume, W&B integration, and sophisticated
    configuration management.
    
    Example usage:
        # Basic usage
        reflow = LightningReflow(
            model_class=MyModel,
            datamodule_class=MyDataModule,
            trainer_defaults={"max_epochs": 10}
        )
        result = reflow.fit()
        
        # With config file and overrides (e.g., for Ray Tune)
        reflow = LightningReflow(
            config_files="config.yaml",
            config_overrides={"model.learning_rate": 0.001},
            callbacks=[RayTrainReportCallback()]
        )
        result = reflow.fit()
        
        # Resume from checkpoint
        reflow = LightningReflow(config_files="config.yaml")
        result = reflow.resume("path/to/checkpoint.ckpt")
    """
    
    def __init__(
        self,
        model_class: Optional[Type[pl.LightningModule]] = None,
        datamodule_class: Optional[Type[pl.LightningDataModule]] = None,
        config_files: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        model_init_args: Optional[Dict[str, Any]] = None,
        datamodule_init_args: Optional[Dict[str, Any]] = None,
        trainer_defaults: Optional[Dict[str, Any]] = None,
        callbacks: Optional[List[Callback]] = None,
        seed_everything: Optional[int] = None,
        resume_strategies: Optional[List[ResumeStrategy]] = None,
        auto_configure_logging: bool = True,
        **kwargs
    ):
        """
        Initialize Lightning Reflow.
        
        Args:
            model_class: PyTorch Lightning model class
            datamodule_class: PyTorch Lightning data module class
            config_files: Path(s) to YAML configuration files
            config_overrides: Dictionary of configuration overrides
            model_init_args: Arguments for model instantiation
            datamodule_init_args: Arguments for datamodule instantiation
            trainer_defaults: Default trainer configuration
            callbacks: List of additional callbacks to include
            seed_everything: Random seed for reproducibility
            resume_strategies: Custom resume strategies (uses defaults if None)
            auto_configure_logging: Whether to automatically configure logging
            **kwargs: Additional arguments
        """
        # Configure logging early
        if auto_configure_logging:
            configure_logging()
        
        logger.info("🚀 Initializing Lightning Reflow")
        
        # Store constructor arguments
        self.model_class = model_class
        self.datamodule_class = datamodule_class
        self.model_init_args = model_init_args or {}
        self.datamodule_init_args = datamodule_init_args or {}
        self.trainer_defaults = trainer_defaults or {}
        self.additional_callbacks = callbacks or []
        self.seed_everything = seed_everything
        
        # Initialize configuration loader
        self.config_loader = ConfigLoader()
        
        # Load and merge configuration (this sets environment variables early!)
        self.config = self.config_loader.load_config(
            config_files=config_files,
            config_overrides=config_overrides,
            apply_env_vars=True
        )
        
        # Set up resume strategies
        if resume_strategies is None:
            self.resume_strategies = [
                LocalPathResumeStrategy(),
                WandbArtifactResumeStrategy()
            ]
        else:
            self.resume_strategies = resume_strategies
        
        # Initialize components
        self.model = None
        self.datamodule = None
        self.trainer = None
        self.result = None
        
        # Set seed early if specified
        if self.seed_everything is not None:
            pl.seed_everything(self.seed_everything)
            logger.info(f"🎲 Seed set to {self.seed_everything}")
    
    def fit(
        self,
        model: Optional[pl.LightningModule] = None,
        datamodule: Optional[pl.LightningDataModule] = None,
        trainer: Optional[pl.Trainer] = None,
        ckpt_path: Optional[str] = None
    ) -> Any:
        """
        Run the training process.
        
        Args:
            model: Pre-instantiated model (optional, will create from config if None)
            datamodule: Pre-instantiated datamodule (optional, will create from config if None)
            trainer: Pre-instantiated trainer (optional, will create from config if None)
            ckpt_path: Path to checkpoint to resume from
            
        Returns:
            Training result from trainer.fit()
        """
        logger.info("🏋️ Starting Lightning Reflow training")
        
        try:
            # Create or use provided components
            self.model = model or self._create_model()
            self.datamodule = datamodule or self._create_datamodule()
            self.trainer = trainer or self._create_trainer()
            
            # Log configuration summary
            self._log_training_summary()
            
            # Run training
            self.result = self.trainer.fit(
                model=self.model,
                datamodule=self.datamodule,
                ckpt_path=ckpt_path
            )
            
            logger.info("✅ Training completed successfully")
            return self.result
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
        finally:
            # Cleanup resume strategies
            self._cleanup_strategies()
    
    def resume(
        self,
        resume_source: str,
        use_wandb_config: bool = False,
        **resume_kwargs
    ) -> Any:
        """
        Resume training from a checkpoint.
        
        Args:
            resume_source: Source to resume from (local path or W&B artifact)
            use_wandb_config: Whether to use config from W&B run (for W&B artifacts)
            **resume_kwargs: Additional arguments for resume strategies
            
        Returns:
            Training result from trainer.fit()
        """
        logger.info(f"🔄 Resuming Lightning Reflow training from: {resume_source}")
        
        try:
            # Find appropriate resume strategy
            strategy = self._select_resume_strategy(resume_source)
            
            # Prepare for resumption
            checkpoint_path, additional_config = strategy.prepare_resume(
                resume_source=resume_source,
                use_wandb_config=use_wandb_config,
                **resume_kwargs
            )
            
            # Merge additional config if available
            if additional_config:
                logger.info("Applying additional configuration from checkpoint source")
                merged_config = self.config_loader._apply_overrides(self.config, additional_config)
                self.config_loader.config = merged_config
                self.config = merged_config
            
            # Run training with checkpoint
            return self.fit(ckpt_path=str(checkpoint_path))
            
        except Exception as e:
            logger.error(f"❌ Resume failed: {e}")
            raise
    
    def _create_model(self) -> pl.LightningModule:
        """Create model from configuration."""
        if self.model_class:
            # Use provided model class with init args
            model_args = {**self.model_init_args}
            
            # Merge with config if available - check both locations
            config_model_args = self.config_loader.get_section("model.init_args", {})
            if config_model_args:
                model_args = {**config_model_args, **model_args}
            
            # Also check for direct model parameters (from overrides like "model.learning_rate")
            model_section = self.config_loader.get_section("model", {})
            if model_section:
                # Extract non-init_args parameters that should go to model constructor
                for key, value in model_section.items():
                    if key not in ["class_path", "init_args"]:
                        model_args[key] = value
            
            logger.info(f"Creating model: {self.model_class.__name__} with args: {model_args}")
            return self.model_class(**model_args)
        
        else:
            # Create from config
            model_config = self.config_loader.get_section("model")
            if not model_config:
                raise ValueError("No model configuration found and no model_class provided")
            
            class_path = model_config.get("class_path")
            init_args = model_config.get("init_args", {})
            
            if not class_path:
                raise ValueError("model.class_path not specified in configuration")
            
            # Import and instantiate the model class
            model_class = self._import_class(class_path)
            logger.info(f"Creating model from config: {class_path}")
            return model_class(**init_args)
    
    def _create_datamodule(self) -> Optional[pl.LightningDataModule]:
        """Create datamodule from configuration."""
        if self.datamodule_class:
            # Use provided datamodule class with init args
            datamodule_args = {**self.datamodule_init_args}
            
            # Merge with config if available - check both locations
            config_datamodule_args = self.config_loader.get_section("data.init_args", {})
            if config_datamodule_args:
                datamodule_args = {**config_datamodule_args, **datamodule_args}
            
            # Also check for direct data parameters (from overrides like "data.batch_size")
            data_section = self.config_loader.get_section("data", {})
            if data_section:
                # Extract non-init_args parameters that should go to datamodule constructor
                for key, value in data_section.items():
                    if key not in ["class_path", "init_args"]:
                        datamodule_args[key] = value
            
            logger.info(f"Creating datamodule: {self.datamodule_class.__name__} with args: {datamodule_args}")
            return self.datamodule_class(**datamodule_args)
        
        else:
            # Create from config
            data_config = self.config_loader.get_section("data")
            if not data_config:
                logger.info("No datamodule configuration found, trainer will expect dataloaders from model")
                return None
            
            class_path = data_config.get("class_path")
            init_args = data_config.get("init_args", {})
            
            if not class_path:
                return None
            
            # Import and instantiate the datamodule class
            datamodule_class = self._import_class(class_path)
            logger.info(f"Creating datamodule from config: {class_path}")
            return datamodule_class(**init_args)
    
    def _create_trainer(self) -> pl.Trainer:
        """Create trainer from configuration."""
        # Start with default trainer config
        trainer_config = {**self.trainer_defaults}
        
        # Merge with config file trainer settings
        config_trainer = self.config_loader.get_section("trainer", {})
        if config_trainer:
            trainer_config.update(config_trainer)
        
        # Handle callbacks
        callbacks = self._prepare_callbacks(trainer_config.get("callbacks", []))
        trainer_config["callbacks"] = callbacks
        
        logger.info(f"Creating trainer with {len(callbacks)} callbacks")
        return pl.Trainer(**trainer_config)
    
    def _prepare_callbacks(self, config_callbacks: List[Any]) -> List[Callback]:
        """Prepare the full list of callbacks."""
        all_callbacks = []
        
        # Add callbacks from trainer config
        for callback_config in config_callbacks:
            if isinstance(callback_config, Callback):
                all_callbacks.append(callback_config)
            elif isinstance(callback_config, dict):
                # Handle config-based callback instantiation
                callback = self._create_callback_from_config(callback_config)
                if callback:
                    all_callbacks.append(callback)
        
        # Add additional callbacks provided programmatically
        all_callbacks.extend(self.additional_callbacks)
        
        return all_callbacks
    
    def _create_callback_from_config(self, callback_config: Dict[str, Any]) -> Optional[Callback]:
        """Create a callback from configuration."""
        try:
            class_path = callback_config.get("class_path")
            init_args = callback_config.get("init_args", {})
            
            if not class_path:
                logger.warning("Callback config missing class_path")
                return None
            
            callback_class = self._import_class(class_path)
            return callback_class(**init_args)
            
        except Exception as e:
            logger.warning(f"Failed to create callback from config: {e}")
            return None
    
    def _select_resume_strategy(self, resume_source: str) -> ResumeStrategy:
        """Select the appropriate resume strategy for the source."""
        for strategy in self.resume_strategies:
            if strategy.validate_source(resume_source):
                logger.info(f"Selected resume strategy: {strategy.__class__.__name__}")
                return strategy
        
        # Default to local path strategy
        logger.info("No specific strategy matched, defaulting to LocalPathResumeStrategy")
        return LocalPathResumeStrategy()
    
    def _import_class(self, class_path: str) -> Type:
        """Import a class from a string path."""
        module_path, class_name = class_path.rsplit(".", 1)
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    
    def _log_training_summary(self) -> None:
        """Log a summary of the training configuration."""
        logger.info("📋 Training Summary:")
        logger.info(f"   Model: {self.model.__class__.__name__}")
        logger.info(f"   Datamodule: {self.datamodule.__class__.__name__ if self.datamodule else 'None'}")
        logger.info(f"   Trainer: {len(self.trainer.callbacks)} callbacks, {self.trainer.max_epochs} max epochs")
        
        if hasattr(self.model, 'hparams'):
            logger.info(f"   Model hyperparameters: {dict(self.model.hparams)}")
    
    def _cleanup_strategies(self) -> None:
        """Clean up all resume strategies."""
        for strategy in self.resume_strategies:
            try:
                strategy.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up strategy {strategy.__class__.__name__}: {e}")
    
    def __del__(self):
        """Cleanup on object destruction."""
        try:
            self._cleanup_strategies()
        except Exception:
            pass 