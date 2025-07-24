import lightning.pytorch as pl
import logging
from lightning.pytorch.utilities import rank_zero_only
from lightning_reflow.utils.config import get_config_raw

# Imports for type hinting that were previously global, now relative
try:
    from modules.training.process_samplers import DiffusionProcessSampler, BaseProcessSampler
except ImportError:
    DiffusionProcessSampler = None 
    BaseProcessSampler = None

class ConfigSummaryLogger(pl.Callback):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
    
    @rank_zero_only
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if not trainer.logger: return
        project_name = getattr(trainer.logger.experiment, "project", "Unknown")
        run_name = getattr(trainer.logger.experiment, "name", "Unknown")
        
        self.logger.info("=" * 50)
        self.logger.info("ConfigSummaryLogger: Logging Model and System Configuration")
        self.logger.info("Project: %s, Run: %s", project_name, run_name)

        if hasattr(pl_module, 'config') and pl_module.config is not None:
            model_c = pl_module.config
            logging_c = model_c.logging # Assumed to exist by Pydantic with defaults

            if model_c.condition_encoder:
                self.logger.info("Condition Encoder: %s", model_c.condition_encoder.type)
                if hasattr(model_c.condition_encoder, 'num_classes') and model_c.condition_encoder.num_classes is not None:
                    self.logger.info("  Num Classes: %s", model_c.condition_encoder.num_classes)
                if hasattr(model_c.condition_encoder, 'embed_dim') and model_c.condition_encoder.embed_dim is not None:
                    self.logger.info("  Embedding Dim: %s", model_c.condition_encoder.embed_dim)
                self.logger.info("  CFG Unconditional Strategy: %s", model_c.cfg_unconditional_strategy)
            else:
                self.logger.info("Condition Encoder: None (Unconditional Model)")

            if pl_module.process_sampler:
                sampler = pl_module.process_sampler
                self.logger.info("Process Sampler: %s", sampler.__class__.__name__)
                if DiffusionProcessSampler and isinstance(sampler, DiffusionProcessSampler):
                    self.logger.info("  Num Timesteps: %s, Schedule: %s", sampler.num_train_timesteps, sampler.schedule_type)
                elif BaseProcessSampler and hasattr(sampler, 'path_type'): # FlowMatching
                    self.logger.info("  Path Type: %s", sampler.path_type)
                    if hasattr(sampler, 'ode_solver') and sampler.ode_solver:
                        self.logger.info("  ODE Solver: %s", sampler.ode_solver.method)
            self.logger.info("Val Sample Gen Steps (from model.logging): %s", logging_c.generation_num_inference_steps)
        else:
            self.logger.info("Model configuration (pl_module.config) not found.")
        self.logger.info("=" * 50) 