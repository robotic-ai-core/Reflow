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
    
    **Progress Bar Management**: LightningReflow automatically disables Lightning's 
    default progress bar (`enable_progress_bar=False`) and replaces it with 
    FlowProgressBarCallback, which provides dual progress bars (global + interval)
    with enhanced metrics display. This prevents UI conflicts between progress systems.
    
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
                WandbArtifactResumeStrategy(),  # Check W&B artifacts first
                LocalPathResumeStrategy()        # Then fall back to local paths
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
            import traceback
            logger.error(f"❌ Training failed: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
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
            
            # Store checkpoint path for potential fallback use
            self._resume_checkpoint_path = str(checkpoint_path)
            
            # Merge additional config if available
            if additional_config:
                logger.info("Applying additional configuration from checkpoint source")
                merged_config = self.config_loader._apply_overrides(self.config, additional_config)
                self.config_loader.config = merged_config
                self.config = merged_config
            
            # Run training with checkpoint
            return self.fit(ckpt_path=str(checkpoint_path))
            
        except Exception as e:
            import traceback
            logger.error(f"❌ Resume failed: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            raise
    
    def _create_model(self) -> pl.LightningModule:
        """Create model from configuration with enhanced resume support."""
        if self.model_class:
            model_args = {**self.model_init_args}
            
            # PRIMARY: Use Lightning's proven hyper_parameters during resume
            if hasattr(self, '_resume_checkpoint_path'):
                logger.info("⚡ Attempting to use Lightning's hyper_parameters (primary)")
                try:
                    lightning_args = self._extract_model_args_from_checkpoint(self._resume_checkpoint_path)
                    if lightning_args:
                        model_args.update(lightning_args)
                        logger.info(f"✅ Using Lightning's hyper_parameters for model creation")
                except Exception as e:
                    logger.warning(f"⚠️ Lightning hyper_parameters extraction failed: {e}")
            
            # FALLBACK: Use embedded config if Lightning's approach didn't provide args OR for fresh training
            if not model_args or not hasattr(self, '_resume_checkpoint_path'):
                if not hasattr(self, '_resume_checkpoint_path'):
                    logger.info("🆕 Fresh training - using config files")
                else:
                    logger.info("🔄 Falling back to embedded config extraction")
                    
                config_model_section = self.config_loader.get_section("model", {})
                
                if config_model_section:
                    # The primary source of arguments should be the 'init_args' subsection
                    config_model_args = config_model_section.get("init_args", {})
                    if isinstance(config_model_args, dict):
                        model_args.update(config_model_args)
                    
                    # Also include top-level args from the 'model' section, for convenience
                    for key, value in config_model_section.items():
                        if key not in ["class_path", "init_args"]:
                            model_args[key] = value
                    
                    if model_args:
                        logger.info(f"✅ Using config for model creation")

            # Convert any nested dictionaries (like 'backbone') to their proper dataclass types
            try:
                from ..utils.config.config_synthesis import convert_config_dict_to_dataclasses
                model_args = convert_config_dict_to_dataclasses(model_args)
            except ImportError:
                logger.warning("Config synthesis not available, continuing with dict args")
            except Exception as e:
                logger.warning(f"Config synthesis failed: {e}, continuing with dict args")
            
            logger.info(f"Creating model: {self.model_class.__name__} with args: {list(model_args.keys()) if model_args else 'EMPTY'}")
            
            # Debug logging for troubleshooting
            if not model_args:
                logger.error("❌ CRITICAL: Model args are empty! This will cause process_sampler=None error")
                logger.error("   Config sections available: %s", list(self.config.keys()) if self.config else "No config")
                if config_model_section:
                    logger.error("   Model section keys: %s", list(config_model_section.keys()))
            
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
            
            # Convert init_args if needed
            try:
                from ..utils.config.config_synthesis import convert_config_dict_to_dataclasses
                init_args = convert_config_dict_to_dataclasses(init_args)
            except Exception as e:
                logger.warning(f"Config synthesis failed for init_args: {e}")
            
            # Import and instantiate the model class
            model_class = self._import_class(class_path)
            logger.info(f"Creating model from config: {class_path}")
            return model_class(**init_args)
    
    def _extract_model_args_from_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Extract model arguments from Lightning's native hyper_parameters.
        This serves as a robust fallback when embedded config fails.
        """
        try:
            import torch
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            if 'hyper_parameters' not in checkpoint:
                logger.warning("No hyper_parameters found in checkpoint")
                return {}
            
            hparams = checkpoint['hyper_parameters']
            
            # Extract model parameters (excluding Lightning internal fields)
            model_params = {}
            for key, value in hparams.items():
                if not key.startswith('_'):  # Skip Lightning internal fields
                    model_params[key] = value
            
            logger.info(f"📦 Extracted {len(model_params)} model parameters from Lightning checkpoint")
            return model_params
            
        except Exception as e:
            logger.error(f"Failed to extract model args from checkpoint: {e}")
            return {}
    
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
        from .shared_config import get_trainer_defaults
        
        # Start with shared defaults + user defaults
        trainer_config = get_trainer_defaults(self.trainer_defaults)
        logger.info(f"🔧 Initial trainer config (shared + user defaults): enable_progress_bar={trainer_config.get('enable_progress_bar')}")
        
        # Merge with config file trainer settings
        config_trainer = self.config_loader.get_section("trainer", {})
        if config_trainer:
            # Log before merging to help debug overrides
            config_progress_bar = config_trainer.get('enable_progress_bar')
            if config_progress_bar is True:
                logger.warning(f"⚠️ Config file is trying to enable Lightning's default progress bar!")
                logger.warning("⚠️ This will conflict with Reflow's custom progress bar (PauseCallback).")
                logger.warning("⚠️ Ignoring enable_progress_bar=true from config to prevent UI conflicts.")
                logger.warning("⚠️ If you need to disable progress bars entirely, set enable_pause=False in PauseCallback config.")
                
                # Apply config settings but keep our progress bar setting
                config_trainer_safe = config_trainer.copy()
                config_trainer_safe.pop('enable_progress_bar', None)
                trainer_config.update(config_trainer_safe)
            else:
                trainer_config.update(config_trainer)
            
            logger.info(f"🔧 Final trainer config (after config file merge): enable_progress_bar={trainer_config.get('enable_progress_bar')}")
        
        # Handle callbacks
        callbacks = self._prepare_callbacks(trainer_config.get("callbacks", []))
        trainer_config["callbacks"] = callbacks
        
        # Handle logger configuration - ensure it's not a string
        if "logger" in trainer_config:
            logger_cfg = trainer_config["logger"]
            
            # Convert OmegaConf to dict if needed
            try:
                from omegaconf import DictConfig, OmegaConf
                if isinstance(logger_cfg, DictConfig):
                    logger_cfg = OmegaConf.to_container(logger_cfg, resolve=True)
                    trainer_config["logger"] = logger_cfg
            except ImportError:
                pass
            
            if isinstance(logger_cfg, str):
                # If logger is a string, remove it or convert to proper logger config
                logger.warning(f"Logger configuration is a string: {logger_cfg}. Removing it.")
                trainer_config.pop("logger")
            elif isinstance(logger_cfg, list):
                # Filter out any string loggers in the list
                valid_loggers = []
                for l in logger_cfg:
                    if isinstance(l, str):
                        logger.warning(f"Skipping string logger: {l}")
                    else:
                        valid_loggers.append(l)
                trainer_config["logger"] = valid_loggers if valid_loggers else None
            elif isinstance(logger_cfg, dict):
                # Check if it's a dict config for a logger
                if "_target_" in logger_cfg:
                    # This is likely a hydra/omegaconf instantiation config
                    logger.info(f"Logger config has _target_: {logger_cfg.get('_target_')}")
                    # For now, remove it to avoid issues
                    logger.warning("Removing logger config with _target_ to avoid instantiation issues")
                    trainer_config.pop("logger")
                elif "class_path" in logger_cfg:
                    # This is a Lightning CLI-style logger config
                    try:
                        logger_class = self._import_class(logger_cfg["class_path"])
                        init_args = logger_cfg.get("init_args", {})
                        logger_instance = logger_class(**init_args)
                        trainer_config["logger"] = logger_instance
                        logger.info(f"Created logger instance: {logger_class.__name__}")
                    except Exception as e:
                        logger.warning(f"Failed to create logger from config: {e}")
                        trainer_config.pop("logger", None)
        
        
        logger.info(f"Creating trainer with {len(callbacks)} callbacks")
        return pl.Trainer(**trainer_config)
    
    def _prepare_callbacks(self, config_callbacks: List[Any]) -> List[Callback]:
        """Prepare the full list of callbacks."""
        from .shared_config import ensure_essential_callbacks
        
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
        
        # Ensure essential callbacks are present using shared logic
        all_callbacks = ensure_essential_callbacks(all_callbacks)
        
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
        # Prioritize local path if it exists, as W&B artifacts can look like paths
        if Path(resume_source).exists():
            logger.info("Resume source is a local path, selecting LocalPathResumeStrategy")
            return LocalPathResumeStrategy()
        
        for strategy in self.resume_strategies:
            if strategy.validate_source(resume_source):
                logger.info(f"Selected resume strategy: {strategy.__class__.__name__}")
                return strategy
        
        # Default to local path strategy if no specific strategy matches
        # This can happen if the path does not exist yet (e.g. will be created by another process)
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