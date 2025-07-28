import lightning.pytorch as pl
import warnings
from lightning.pytorch.utilities import rank_zero_only
from typing import Union, List
from lightning_reflow.utils.logging.logging_config import get_logger

class WandbWatchCallback(pl.Callback):
    """
    Handles the setup of wandb.watch() for logging model gradients and parameters.
    
    This callback automatically calls wandb.watch() on your model during training,
    providing insights into gradient flow, parameter evolution, and training dynamics.
    
    Examples:
        # Default usage (checks for 'backbone' attribute)
        WandbWatchCallback()
        
        # For models with different architecture attribute names
        WandbWatchCallback(model_attribute="encoder")  # For encoder-decoder models
        WandbWatchCallback(model_attribute="model")    # For wrapped models
        WandbWatchCallback(model_attribute="net")      # For network-style naming
        
        # For models that might use different naming conventions (tries in order)
        WandbWatchCallback(model_attribute=["backbone", "encoder", "model", "net"])
        WandbWatchCallback(model_attribute=["model", "network", "core"])  # Custom priority order
        
        # For simple models without sub-components
        WandbWatchCallback(model_attribute=None)
        
        # Full configuration
        WandbWatchCallback(
            log_level="parameters",                    # gradients, parameters, all, or None
            log_freq=500,                             # Log every 500 steps
            log_graph=False,                          # Don't log computational graph
            model_attribute=["backbone", "encoder"]   # Check multiple attributes in order
        )
    """
    def __init__(self, log_level: str = "gradients", log_freq: int = 1000, log_graph: bool = False, 
                 model_attribute: Union[str, List[str], None] = "backbone"):
        """
        Initialize WandbWatchCallback.
        
        Args:
            log_level: What to log - "gradients", "parameters", "all", or None
            log_freq: How often to log (every N steps)
            log_graph: Whether to log model computational graph
            model_attribute: Name(s) of model attribute(s) to check for compilation.
                           Can be:
                           - str: Single attribute name (e.g., "backbone")
                           - List[str]: Multiple attribute names to try in order (e.g., ["backbone", "encoder", "model"])
                           - None: Only check the main module
        """
        super().__init__()
        self.log_level = log_level
        self.log_freq = log_freq
        self.log_graph = log_graph
        self.model_attribute = model_attribute
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
                is_compiled = hasattr(pl_module, '_orig_mod')
                
                # Check configurable model attribute(s) for compilation (if specified and exists)
                if self.model_attribute is not None:
                    # Handle both single string and list of strings
                    attributes_to_check = [self.model_attribute] if isinstance(self.model_attribute, str) else self.model_attribute
                    
                    for attr_name in attributes_to_check:
                        if hasattr(pl_module, attr_name):
                            model_component = getattr(pl_module, attr_name)
                            is_compiled = is_compiled or hasattr(model_component, '_orig_mod')
                            self.logger.debug(f"WandbWatchCallback: Found and checked '{attr_name}' attribute for compilation")
                            break  # Use the first attribute that exists
                    else:
                        # No attributes were found
                        attrs_str = f"'{self.model_attribute}'" if isinstance(self.model_attribute, str) else str(self.model_attribute)
                        self.logger.debug(f"WandbWatchCallback: None of the model attributes {attrs_str} found, skipping sub-component compilation check")
                
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