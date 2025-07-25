import torch
import warnings
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
import sys
import os
import argparse
import json
import subprocess
import time

from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

# Assuming these modules are in the same 'modules' directory or accessible
from lightning_reflow.utils.logging.logging_config import setup_logging, configure_logging
from lightning_reflow.utils.wandb.resume_command_handler import ResumeCommandHandler
from lightning_reflow.utils.logging.environment_manager import EnvironmentManager

class LightningReflowCLI(LightningCLI):
    """
    Extended LightningCLI with advanced resume capabilities and pause functionality.
    """
    
    def __init__(self, *args, **kwargs):
        self._wandb_checkpoint_path: Optional[str] = None
        self._detected_wandb_run_id: Optional[str] = None  # Store detected run ID
        
        # Store trainer_defaults so we can modify them
        self._trainer_defaults = kwargs.get('trainer_defaults', {})
        
        # Store save_config_kwargs for validation by callbacks
        self.save_config_kwargs = kwargs.get('save_config_kwargs', {})
        
        # Initialize resume handler
        self._resume_handler = ResumeCommandHandler(self)
        
        # Handle resume subcommand ONLY if it's not handled by Lightning CLI
        if self._resume_handler.should_handle_resume_manually():
            self._resume_handler.execute_resume_subcommand()
            return  # Resume subcommand handled, exit early
        
        # Configure wandb resume BEFORE calling super().__init__
        self._configure_wandb_resume_early(kwargs)
        
        super().__init__(*args, **kwargs)
        
        # Post-initialization: fix W&B logger if needed
        self._fix_wandb_logger_post_init()

    def before_instantiate_classes(self) -> None:
        """Handle logging setup, wandb config, and convert multimodal configs."""
        super().before_instantiate_classes()
        
        # Add PauseCallback unless disabled
        self._configure_pause_exit_callback()
        
        # Fix Lightning's broken fast_dev_run + ModelCheckpoint validation
        # Note: This was removed as the method is not defined in this class
        # self._fix_lightning_checkpoint_validation()
        
        # Setup logging
        self._setup_logging()
        
        # Handle W&B config path
        self._handle_wandb_config_path()
        
        # Convert multimodal configs (REQUIRED FOR OmegaConf compatibility)
        self._convert_multimodal_configs()
        
        # Setup environment
        self._setup_environment()
        
    def instantiate_trainer(self, **kwargs):
        """Override to set CLI reference in trainer."""
        # Call parent implementation to create trainer
        trainer = super().instantiate_trainer(**kwargs)
        
        # Store CLI reference for compatibility (some callbacks might use it)
        trainer.cli = self
        logger.info("✅ Stored CLI reference in trainer")
        
        return trainer
    
    def after_instantiate_classes(self) -> None:
        """Called after trainer is instantiated - CLI reference should already be set."""
        super().after_instantiate_classes()
        
        # Handle wandb resume configuration
        resume_from_wandb = None
        if hasattr(self.config, 'resume_from_wandb') and self.config.resume_from_wandb:
            resume_from_wandb = self.config.resume_from_wandb
        elif hasattr(self.config, 'fit') and hasattr(self.config.fit, 'resume_from_wandb') and self.config.fit.resume_from_wandb:
            resume_from_wandb = self.config.fit.resume_from_wandb
        
        use_wandb_config = False
        if hasattr(self.config, 'use_wandb_config') and self.config.use_wandb_config:
            use_wandb_config = True
        elif hasattr(self.config, 'fit') and hasattr(self.config.fit, 'use_wandb_config') and self.config.fit.use_wandb_config:
            use_wandb_config = True
        
        if use_wandb_config and not resume_from_wandb:
            raise ValueError("--use_wandb_config can only be used together with --resume_from_wandb")
        
        if resume_from_wandb:
            self._wandb_checkpoint_path, wandb_config_path = self._download_wandb_checkpoint(resume_from_wandb, use_wandb_config)
            print(f"[INFO] Downloaded W&B checkpoint to: {self._wandb_checkpoint_path}")
            
            auto_resume_wandb = True
            if '--no_auto_resume_wandb' in sys.argv:
                auto_resume_wandb = False
                
            if auto_resume_wandb and self._wandb_checkpoint_path:
                detected_run_id = self._extract_wandb_run_id_from_path(self._wandb_checkpoint_path)
                if detected_run_id:
                    print(f"[INFO] Auto-detected wandb run ID from W&B checkpoint: {detected_run_id}")
                    self._detected_wandb_run_id = detected_run_id
                    
                    if hasattr(self.config, 'trainer') and hasattr(self.config.trainer, 'logger'):
                        if hasattr(self.config.trainer.logger, 'init_args'):
                            self.config.trainer.logger.init_args.id = detected_run_id
                            self.config.trainer.logger.init_args.resume = "allow"
                            print(f"[INFO] Configured W&B logger to resume run {detected_run_id}")
                    else:
                        print(f"[WARNING] Could not configure W&B logger - trainer.logger not found in config")
            
            if use_wandb_config and wandb_config_path:
                print(f"[INFO] Using W&B config file: {wandb_config_path}")
                if '--config' not in sys.argv:
                    subcommands = ['fit', 'validate', 'test', 'predict']
                    insert_idx = 2 
                    for i, arg in enumerate(sys.argv):
                        if arg in subcommands:
                            insert_idx = i + 1
                            break
                    sys.argv.insert(insert_idx, '--config')
                    sys.argv.insert(insert_idx + 1, wandb_config_path)
                    print(f"[INFO] Added W&B config to command line: --config {wandb_config_path}")
                else:
                    print(f"[WARNING] Config already specified in command line, will use W&B config as additional override")
                    insert_idx = len(sys.argv)
                    sys.argv.insert(insert_idx, '--config')
                    sys.argv.insert(insert_idx + 1, wandb_config_path)
            elif use_wandb_config and not wandb_config_path:
                print(f"[WARNING] --use_wandb_config specified but no config found in W&B artifact")
        
    def _configure_wandb_resume_early(self, kwargs):
        """Configure wandb resume before trainer instantiation using Lightning CLI's native mechanism."""
        wandb_run_id = self._extract_wandb_run_from_args()
        if wandb_run_id:
            print(f"[INFO] W&B run ID detected: {wandb_run_id}")
            return
        
        if self._should_auto_resume_wandb():
            checkpoint_path = self._extract_checkpoint_path_from_args()
            if checkpoint_path:
                detected_run_id = self._extract_wandb_run_id_from_path(checkpoint_path)
                if detected_run_id:
                    print(f"[INFO] Auto-detected W&B run ID from checkpoint: {detected_run_id}")

    def _fix_wandb_logger_post_init(self):
        """Validate W&B logger configuration after CLI initialization."""
        wandb_run_id = self._extract_wandb_run_from_args()
        if wandb_run_id and hasattr(self, 'trainer') and self.trainer:
            if hasattr(self.trainer, 'logger') and self.trainer.logger:
                from lightning.pytorch.loggers import WandbLogger
                if isinstance(self.trainer.logger, WandbLogger):
                    print(f"[INFO] Validating W&B logger configuration for run: {wandb_run_id}")

    def _process_environment_variables_from_config(self) -> None:
        """Process environment variables from original config files using Lightning's config precedence."""
        try:
            from lightning_reflow.utils.logging.environment_manager import EnvironmentManager
            import sys
            
            config_files = []
            i = 0
            while i < len(sys.argv):
                if sys.argv[i] == '--config' and i + 1 < len(sys.argv):
                    config_files.append(Path(sys.argv[i + 1]))
                    i += 2
                elif sys.argv[i].startswith('--config='):
                    config_files.append(Path(sys.argv[i].split('=', 1)[1]))
                    i += 1
                else:
                    i += 1
            
            if config_files:
                print(f"[INFO] Processing environment variables from {len(config_files)} config files in Lightning precedence order:")
                for i, config_file in enumerate(config_files):
                    print(f"[INFO]   {i+1}. {config_file}")
                
                env_vars, _ = EnvironmentManager.extract_environment_from_configs(config_files)
                
                if env_vars:
                    print(f"[INFO] Merged {len(env_vars)} environment variables from configs:")
                    for var_name, value in env_vars.items():
                        print(f"[INFO]   {var_name}={value}")
                    
                    config_sources = [str(f) for f in config_files]
                    EnvironmentManager.set_environment_variables(env_vars, config_sources)
                    
                    EnvironmentManager.register_for_checkpoint_persistence()
                    
                    print(f"[INFO] Environment variables set via Lightning CLI with proper precedence")
                else:
                    print(f"[INFO] No environment variables found in config files")
            else:
                print(f"[INFO] No config files specified - no environment variables to process")
                
        except Exception as e:
            print(f"[WARNING] Failed to process environment variables from config: {e}")

    def _process_environment_callback_config(self) -> None:
        """Process environment variables from EnvironmentCallback configuration."""
        try:
            import sys
            import yaml
            
            config_files = []
            i = 0
            while i < len(sys.argv):
                if sys.argv[i] == '--config' and i + 1 < len(sys.argv):
                    config_files.append(Path(sys.argv[i + 1]))
                    i += 2
                elif sys.argv[i].startswith('--config='):
                    config_files.append(Path(sys.argv[i].split('=', 1)[1]))
                    i += 1
                else:
                    i += 1
            
            env_vars = {}
            for config_file in config_files:
                if not config_file.exists():
                    continue
                    
                try:
                    with open(config_file, 'r') as f:
                        config_data = yaml.safe_load(f)
                    
                    trainer_config = config_data.get('trainer', {})
                    callbacks = trainer_config.get('callbacks', [])
                    
                    for callback in callbacks:
                        if isinstance(callback, dict):
                            class_path = callback.get('class_path', '')
                            # Check for both possible paths for EnvironmentCallback
                            if (class_path == 'lightning_reflow.callbacks.EnvironmentCallback' or
                                class_path == 'modules.callbacks.EnvironmentCallback' or
                                class_path.endswith('.EnvironmentCallback')):
                                init_args = callback.get('init_args', {})
                                callback_env_vars = init_args.get('env_vars', {})
                                env_vars.update(callback_env_vars)
                            
                except Exception as e:
                    print(f"[WARNING] Error parsing {config_file}: {e}")
            
            try:
                if hasattr(self, '_trainer_defaults') and self._trainer_defaults:
                    callbacks = self._trainer_defaults.get('callbacks', [])
                    for callback in callbacks:
                        if hasattr(callback, '__class__') and callback.__class__.__name__ == 'EnvironmentCallback':
                            if hasattr(callback, 'env_vars'):
                                env_vars.update(callback.env_vars)
            except Exception as e:
                print(f"[WARNING] Error extracting EnvironmentCallback from trainer defaults: {e}")
            
            if not env_vars:
                return
            
            print(f"[INFO] Setting {len(env_vars)} environment variables from EnvironmentCallback config")
            
            for var_name, value in env_vars.items():
                existing_value = os.environ.get(var_name)
                value_str = str(value)
                
                if existing_value and existing_value != value_str:
                    print(f"[INFO] ⚠️  Overriding existing {var_name}:")
                    print(f"[INFO]      Existing (checkpoint): {existing_value}")
                    print(f"[INFO]      New (config file):     {value_str}")
                    print(f"[INFO]      Config file takes precedence during resume")
                    
                os.environ[var_name] = value_str
                print(f"[INFO]   ✅ {var_name}={value_str}")
            
            try:
                from lightning_reflow.utils.logging.environment_manager import EnvironmentManager
                config_source = ["EnvironmentCallback configuration"]
                EnvironmentManager.set_environment_variables(env_vars, config_source)
                EnvironmentManager.register_for_checkpoint_persistence()
            except Exception as e:
                print(f"[INFO] Environment variables set (without state persistence): {e}")
                            
        except Exception as e:
            print(f"[WARNING] Failed to process EnvironmentCallback config: {e}")

    def add_arguments_to_parser(self, parser):
        parser.add_argument(
            '--wandb_project',
            type=str,
            default="",  # Changed from None to empty string to satisfy type checking
            help='W&B project name. Overrides project name in logger config.'
        )
        parser.add_argument(
            '--wandb_log_model',
            type=Optional[bool],
            default=None, 
            help='Custom alias to control wandb model logging. Links to trainer.logger.init_args.log_model.'
        )
        parser.add_argument(
            '--resume_from_wandb',
            type=Optional[str],
            default=None,
            help='Resume training from a W&B checkpoint artifact. Format: entity/project/artifact_name:version or artifact_name:version'
        )
        parser.add_argument(
            '--weights_only',
            action='store_true',
            default=False,
            help='Load only model weights from checkpoint, not training state (global_step, epoch, optimizer).'
        )
        parser.add_argument(
            '--use_wandb_config',
            action='store_true',
            default=False,
            help='Use the config file saved in the W&B artifact. Must be used with --resume_from_wandb.'
        )
        parser.add_argument(
            '--auto_resume_wandb',
            action='store_true',
            default=True,
            help='Automatically detect wandb run ID from checkpoint path and resume the same wandb run.'
        )
        parser.add_argument(
            '--no_auto_resume_wandb',
            dest='auto_resume_wandb',
            action='store_false',
            help='Disable automatic wandb run resumption detection.'
        )
        parser.add_argument(
            '--disable_pause_exit',
            action='store_true',
            default=False,
            help='Disable validation-boundary pause functionality.'
        )
        parser.add_argument(
            '--wandb_run', '--wandb-run',
            type=lambda x: x if x else None,
            default=None,
            help='W&B run ID to resume (cleaner alternative to --trainer.logger.init_args.id)'
        )
        parser.link_arguments("wandb_log_model", "trainer.logger.init_args.log_model")
        parser.link_arguments("seed_everything", "data.init_args.seed", apply_on="parse")
        # Only link wandb_project if it's not empty
        parser.link_arguments("wandb_project", "trainer.logger.init_args.project", 
                            compute_fn=lambda x: x if x else None)
        # Link wandb_run to logger id for run ID continuation
        parser.link_arguments("wandb_run", "trainer.logger.init_args.id",
                            compute_fn=lambda x: x if x else None)
        # Also set resume="allow" when wandb_run is provided
        parser.link_arguments("wandb_run", "trainer.logger.init_args.resume",
                            compute_fn=lambda x: "allow" if x else None)

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        configure_logging()
        log_level = os.getenv("VIBE_LOG_LEVEL", "INFO").upper()
        logging_config = {
            "level": log_level,
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "dataset_logging": {
                "enabled": True, 
                "level": log_level, 
                "log_worker_details": True, 
                "log_shard_assignment": True, 
                "log_item_counts": True, 
            }
        }
        setup_logging(logging_config)
        
    def _handle_wandb_config_path(self) -> None:
        """Handle W&B config path if needed."""
        # This is a placeholder - implement if needed
        pass
        
    def _convert_multimodal_configs(self) -> None:
        """Convert multimodal configs for OmegaConf compatibility."""
        # This is a placeholder - implement if needed
        pass
        
    def _setup_environment(self) -> None:
        """Setup environment variables and settings."""
        # Process environment variables from EnvironmentCallback config
        self._process_environment_callback_config()

    def _download_wandb_checkpoint(self, artifact_reference: str, use_wandb_config: bool = False) -> tuple[str, Optional[str]]:
        try:
            import wandb
        except ImportError:
            raise ImportError("wandb is required. Please install it with 'pip install wandb'")
        
        print(f"[INFO] Downloading checkpoint from W&B artifact: {artifact_reference}")
        if use_wandb_config:
            print(f"[INFO] Will also look for config file in the artifact")
        
        try:
            api = wandb.Api()
            artifact = api.artifact(artifact_reference, type="model")
            artifact_dir = artifact.download()
            print(f"[INFO] Downloaded artifact to: {artifact_dir}")
            
            artifact_path = Path(artifact_dir)
            checkpoint_files = list(artifact_path.glob("*.ckpt")) + list(artifact_path.glob("**/*.ckpt"))
            compressed_files = list(artifact_path.glob("*.ckpt.gz")) + list(artifact_path.glob("**/*.ckpt.gz"))
            
            if not checkpoint_files and not compressed_files:
                raise FileNotFoundError(f"No .ckpt or .ckpt.gz files found in downloaded artifact at {artifact_dir}")
            
            if checkpoint_files:
                if len(checkpoint_files) > 1:
                    print(f"[WARNING] Multiple checkpoint files found: {[str(f) for f in checkpoint_files]}")
                    print(f"[INFO] Using the first one: {checkpoint_files[0]}")
                checkpoint_path = str(checkpoint_files[0])
            else:
                if len(compressed_files) > 1:
                    print(f"[WARNING] Multiple compressed checkpoint files found: {[str(f) for f in compressed_files]}")
                    print(f"[INFO] Using the first one: {compressed_files[0]}")
                
                compressed_path = str(compressed_files[0])
                checkpoint_path = self._decompress_checkpoint_file(compressed_path)
            
            config_path = None
            if use_wandb_config:
                config_files = list(artifact_path.glob("*.yaml")) + list(artifact_path.glob("**/*.yaml"))
                config_files.extend(list(artifact_path.glob("*.yml")) + list(artifact_path.glob("**/*.yml")))
                
                if config_files:
                    config_candidates = [f for f in config_files if f.name == "config.yaml"]
                    config_path = str(config_candidates[0]) if config_candidates else str(config_files[0])
                    print(f"[INFO] Found config file: {config_path}")
                else:
                    print(f"[WARNING] --use_wandb_config specified but no config files found in artifact")
            
            try:
                checkpoint_data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                from lightning_reflow.utils.checkpoint.checkpoint_utils import validate_checkpoint_structure
                validate_checkpoint_structure(checkpoint_data, checkpoint_path)
            except Exception as e:
                raise ValueError(f"Failed to load or validate checkpoint file {checkpoint_path}: {e}")
            
            return checkpoint_path, config_path
            
        except Exception as e:
            raise RuntimeError(f"Failed to download or process W&B checkpoint artifact '{artifact_reference}': {e}")
    
    def _decompress_checkpoint_file(self, compressed_path: str) -> str:
        """Decompress a gzipped checkpoint file to a temporary location."""
        import gzip
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix='.ckpt', delete=False) as tmp:
            decompressed_path = tmp.name
        
        try:
            with gzip.open(compressed_path, 'rb') as f_in:
                with open(decompressed_path, 'wb') as f_out:
                    f_out.write(f_in.read())
            return decompressed_path
        except Exception as e:
            if os.path.exists(decompressed_path):
                os.unlink(decompressed_path)
            raise RuntimeError(f"Failed to decompress checkpoint file {compressed_path}: {e}")
    
    def _validate_and_warn_artifact_relationships(self, artifact_path: str) -> None:
        """Validate artifact relationships and warn about potential issues."""
        try:
            from lightning_reflow.utils.wandb.wandb_artifact_manager import WandbArtifactManager
            manager = WandbArtifactManager(verbose=False)
            validation_result = manager.validate_artifact_relationships(artifact_path)
            
            if validation_result['issues'] or not validation_result.get('has_embedded_config', True):
                print("\n🔍 ARTIFACT RELATIONSHIP VALIDATION:")
                print("=" * 60)
                
                status = {
                    'checkpoint_exists': "✅ Found" if validation_result['checkpoint_exists'] else "❌ Missing",
                    'config_exists': "✅ Found" if validation_result['config_exists'] else "❌ Missing",
                    'has_embedded_config': "✅ Found" if validation_result.get('has_embedded_config') else "⚠️ Missing"
                }
                print(f"{status['checkpoint_exists']} Checkpoint artifact")
                print(f"{status['config_exists']} Config artifact")
                print(f"{status['has_embedded_config']} Embedded config")
                
                if validation_result['issues']:
                    print("\n⚠️  ISSUES DETECTED:")
                    for issue in validation_result['issues']: print(f"   • {issue}")
                
                if validation_result['recommendations']:
                    print("\n💡 RECOMMENDATIONS:")
                    for rec in validation_result['recommendations']: print(f"   • {rec}")
                
                print("=" * 60, "\n")
            
        except Exception as e:
            print(f"⚠️  Could not validate artifact relationships: {e}")
    
    def _extract_run_id_from_checkpoint(self, checkpoint_path: Path) -> Optional[str]:
        """Extract W&B run ID from checkpoint metadata."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            pause_metadata = checkpoint.get('pause_callback_metadata', {})
            if pause_metadata and 'wandb_run_id' in pause_metadata:
                run_id = pause_metadata['wandb_run_id']
                if run_id and run_id.strip():
                    return run_id.strip()
            return None
        except Exception as e:
            print(f"❌ Failed to load checkpoint for W&B run ID extraction: {e}")
            return None

    def before_fit(self):
        """Log config to W&B and apply model compilation."""
        if self.trainer.logger and isinstance(self.trainer.logger, WandbLogger):
            config_dict = self.config.as_dict()
            self.trainer.logger.log_hyperparams(config_dict)
            print("[INFO] Logged full configuration to Wandb.")
        self._apply_model_compilation()

    def _apply_model_compilation(self):
        """Apply model compilation if configured."""
        compile_settings = getattr(self.model.hparams, 'torch_compile_settings', None)
        if not compile_settings or not compile_settings.get('enabled', False):
            print("[INFO] Model compilation not requested.")
            return

        final_compile_args = {k: v for k, v in compile_settings.items() if k != 'enabled'}
        print(f"[INFO] Model compilation requested with args: {final_compile_args}")
        
        try:
            self.model = torch.compile(self.model, **final_compile_args)
            print("[INFO] Model compiled successfully.")
        except Exception as e:
            warnings.warn(f"torch.compile failed with error: {e}. Proceeding without compilation.", UserWarning)

    def _load_weights_only(self, checkpoint_path: str, model):
        print(f"[INFO] Loading weights only from: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            from lightning_reflow.utils.checkpoint.checkpoint_utils import validate_checkpoint_structure
            validate_checkpoint_structure(checkpoint, checkpoint_path)
            
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            print(f"[INFO] Successfully loaded weights.")
        except Exception as e:
            raise RuntimeError(f"Failed to load weights from checkpoint {checkpoint_path}: {e}")

    def fit(self, model=None, datamodule=None, ckpt_path=None):
        weights_only = self.config.fit.get('weights_only', False)
        
        final_ckpt_path = ckpt_path or self._wandb_checkpoint_path
            
        if final_ckpt_path:
            if weights_only:
                self._load_weights_only(final_ckpt_path, self.model)
                final_ckpt_path = None
                print(f"[INFO] Loaded weights only. Starting training fresh.")
            else:
                print(f"[INFO] Resuming full training state from: {final_ckpt_path}")
        else:
            print("[INFO] Starting training from scratch.")
        
        self.trainer.fit(self.model, datamodule, ckpt_path=final_ckpt_path)

    def _extract_wandb_run_id_from_path(self, checkpoint_path: str) -> Optional[str]:
        """Extract wandb run ID from checkpoint path patterns or content."""
        try:
            import torch
            from lightning_reflow.utils.checkpoint.checkpoint_utils import extract_wandb_run_id
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            run_id = extract_wandb_run_id(checkpoint)
            if run_id:
                print(f"[INFO] Extracted W&B run ID from checkpoint metadata: {run_id}")
                return run_id
        except Exception as e:
            print(f"[DEBUG] Could not read checkpoint for run ID extraction: {e}")
        
        # Fallback to path parsing when checkpoint loading fails
        try:
            import re
            # Extract run ID from patterns like "run-abc123def-latest" or directory paths  
            pattern = r'run-([a-z0-9]{8})-[^/]*'
            match = re.search(pattern, checkpoint_path)
            if match:
                return match.group(1)
            
            # Only extract from directory paths (must contain at least one slash)
            if '/' not in checkpoint_path:
                return None
                
            # Try extracting from checkpoint path patterns (8-char hex IDs in path)
            # Prefer the rightmost (closest to filename) valid hex ID that contains letters
            path_parts = checkpoint_path.split('/')
            valid_ids = []
            for part in path_parts:
                if (len(part) == 8 and 
                    re.match(r'^[a-z0-9]{8}$', part) and 
                    re.search(r'[a-z]', part)):  # Must contain at least one letter (not all digits)
                    valid_ids.append(part)
            
            # Return the last (rightmost) valid ID
            if valid_ids:
                return valid_ids[-1]
                
        except Exception as e:
            print(f"[DEBUG] Failed to extract run ID from path: {e}")
        
        return None

    def _configure_pause_exit_callback(self) -> None:
        """Configure pause/exit functionality based on CLI flags."""
        if self.config.get('disable_pause_exit', False):
            print("[INFO] Pause functionality disabled via --disable_pause_exit flag")
            return

        print("[INFO] Pause functionality enabled. Press 'p' to pause.")
        from lightning_reflow.callbacks.pause.pause_callback import PauseCallback
        
        has_pause_callback = any(isinstance(cb, PauseCallback) for cb in self._trainer_defaults.get('callbacks', []))
        
        if not has_pause_callback:
            pause_callback = PauseCallback()
            if 'callbacks' in self._trainer_defaults:
                self._trainer_defaults['callbacks'].append(pause_callback)
            else:
                self._trainer_defaults['callbacks'] = [pause_callback]

    def _verify_wandb_run_exists(self, run_id: str) -> bool:
        """Verify that a wandb run exists."""
        try:
            import wandb
            api = wandb.Api()
            project = self.config.trainer.logger.init_args.get('project', 'default_project')
            run_path = f"{project}/{run_id}"
            api.run(run_path)
            print(f"[INFO] Found existing wandb run: {run_path}")
            return True
        except Exception:
            print(f"[INFO] Wandb run {run_id} not found in project {project}")
            return False
    
    def _extract_wandb_run_from_args(self) -> Optional[str]:
        """Extract W&B run ID from command line arguments."""
        for i, arg in enumerate(sys.argv):
            if arg in ['--wandb_run', '--wandb-run'] and i + 1 < len(sys.argv):
                return sys.argv[i + 1]
            elif arg == '--trainer.logger.init_args.id' and i + 1 < len(sys.argv):
                return sys.argv[i + 1]
        return None
    
    def _should_auto_resume_wandb(self) -> bool:
        """Check if auto-resume W&B is enabled."""
        return '--no_auto_resume_wandb' not in sys.argv
    
    def _extract_checkpoint_path_from_args(self) -> Optional[str]:
        """Extract checkpoint path from command line arguments."""
        for i, arg in enumerate(sys.argv):
            if arg == '--ckpt_path' and i + 1 < len(sys.argv):
                return sys.argv[i + 1]
        return None

    # Missing methods required by tests
    def _extract_run_id_from_artifact(self, artifact_path: str) -> Optional[str]:
        """
        Extract W&B run ID from artifact path.
        
        Handles W&B artifact paths like:
        - "myuser/myproject/abc123xyz-pause:v0" → "abc123xyz"
        - "entity/project/run-with-dashes-pause:latest" → "run-with-dashes"
        """
        try:
            import re
            
            # W&B artifact paths follow format: entity/project/artifact_name:version
            # The artifact name typically contains the run ID followed by a suffix like "-pause", "-best", "-latest"
            if '/' in artifact_path and ':' in artifact_path:
                # Split by '/' to get the artifact name part
                parts = artifact_path.split('/')
                if len(parts) >= 3:
                    # Get the artifact name (last part before version)
                    artifact_name_with_version = parts[-1]
                    # Remove version part (after ':')
                    artifact_name = artifact_name_with_version.split(':')[0]
                    
                    # Return None if artifact name is empty
                    if not artifact_name or artifact_name.strip() == '':
                        return None
                    
                    # Extract run ID by removing common suffixes
                    suffixes = ['-pause', '-best', '-latest', '-checkpoint']
                    for suffix in suffixes:
                        if artifact_name.endswith(suffix):
                            run_id = artifact_name[:-len(suffix)]
                            if run_id:  # Ensure we have something left
                                return run_id
                    
                    # If no known suffix, try to find the pattern before the last dash
                    if '-' in artifact_name:
                        # Find the last dash and take everything before it as potential run ID
                        last_dash_index = artifact_name.rfind('-')
                        potential_run_id = artifact_name[:last_dash_index]
                        if potential_run_id:
                            return potential_run_id
                    
                    # Fallback: return the entire artifact name if no patterns match and it's valid
                    if artifact_name and artifact_name.strip():
                        return artifact_name
            
            # Fallback for non-standard formats - try directory structure patterns
            path_parts = artifact_path.split('/')
            for part in path_parts:
                if len(part) == 8 and re.match(r'^[a-f0-9]{8}$', part):
                    return part
                    
            return None
            
        except Exception as e:
            print(f"❌ Failed to extract run ID from artifact path: {e}")
            return None

    def _execute_resume_from_checkpoint(self, checkpoint_path, config_overrides: Optional[List] = None, 
                                       wandb_run_override: Optional[str] = None, dry_run: bool = False,
                                       extra_cli_args: Optional[List[str]] = None) -> bool:
        """Execute resume from checkpoint with embedded config."""
        try:
            import subprocess
            from pathlib import Path
            
            # Handle both str and Path types
            checkpoint_path = str(checkpoint_path) if checkpoint_path else None
            if not checkpoint_path or not Path(checkpoint_path).exists():
                print(f"❌ Checkpoint file not found: {checkpoint_path}")
                return False
            
            # Validate config override files exist
            if config_overrides:
                for override_path in config_overrides:
                    override_path = Path(override_path)
                    if not override_path.exists():
                        print(f"❌ Config override file not found: {override_path}")
                        return False
            
            # Extract embedded config
            config_path = self._extract_embedded_config_from_checkpoint(checkpoint_path)
            if not config_path and not config_overrides:
                print(f"❌ No embedded config found and no config overrides provided")
                return False
            
            # Display config merge order for transparency
            if dry_run and (config_path or config_overrides):
                print("📋 Config merge order:")
                if config_path:
                    print(f"  Base: {config_path} (embedded config)")
                if config_overrides:
                    for i, override in enumerate(config_overrides, 1):
                        print(f"  Override {i}: {override}")
                if extra_cli_args:
                    print(f"  Final: CLI arguments ({' '.join(extra_cli_args)})")
            
            # Build resume command using Lightning's native config merging
            cmd = ["python", "train_lightning.py", "fit"]
            
            # Add embedded config as base (if available)
            if config_path:
                cmd.extend(["--config", config_path])
            
            # Add config overrides - Lightning will merge these automatically
            # Following Lightning's precedence: base -> override1 -> override2 -> CLI args
            if config_overrides:
                for override in config_overrides:
                    cmd.extend(["--config", str(override)])
            
            # Add checkpoint path for resume
            cmd.extend(["--ckpt_path", checkpoint_path])
            
            # Add W&B run configuration if provided  
            if wandb_run_override:
                # Use the --wandb_run argument which will be properly linked
                cmd.extend(["--wandb_run", wandb_run_override])
            
            # Skip sanity validation on resume (Lightning best practice)
            cmd.extend(["--trainer.num_sanity_val_steps", "0"])
            
            # Add extra CLI arguments - these have highest precedence in Lightning
            if extra_cli_args:
                cmd.extend(extra_cli_args)
            
            if dry_run:
                print(f"[DRY RUN] Would execute: {' '.join(cmd)}")
                print("📋 Lightning config precedence: embedded → overrides → CLI args")
                return True
            
            # Debug logging
            print(f"[DEBUG] Executing resume command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd)
            return result.returncode == 0
            
        except Exception as e:
            print(f"❌ Failed to execute resume from checkpoint: {e}")
            return False

    def _extract_embedded_config_from_checkpoint(self, checkpoint_path: str) -> Optional[str]:
        """Extract embedded config from checkpoint and save to temporary file."""
        try:
            import tempfile
            import torch
            
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            pause_metadata = checkpoint.get('pause_callback_metadata', {})
            
            embedded_config = pause_metadata.get('embedded_config_content')
            if not embedded_config:
                return None
            
            # Create temporary config file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write(embedded_config)
                return f.name
                
        except Exception as e:
            print(f"❌ Failed to extract embedded config: {e}")
            return None

    def _extract_config_from_checkpoint_metadata(self, checkpoint_path: str) -> Optional[str]:
        """Alias for _extract_embedded_config_from_checkpoint for compatibility."""
        return self._extract_embedded_config_from_checkpoint(checkpoint_path)

    def _execute_resume_from_artifact(self, artifact_reference: str, wandb_run_override: str = None,
                                    config_overrides: List[Path] = None, dry_run: bool = False,
                                    extra_cli_args: List[str] = None) -> bool:
        """Execute resume from W&B artifact."""
        try:
            # Download artifact
            checkpoint_path, config_path = self._download_wandb_checkpoint(artifact_reference, use_wandb_config=True)
            
            # Extract W&B run ID from checkpoint if not explicitly provided
            if not wandb_run_override:
                wandb_run_override = self._extract_wandb_run_id_from_path(checkpoint_path)
                if wandb_run_override:
                    print(f"[INFO] Auto-detected W&B run ID from checkpoint: {wandb_run_override}")
                else:
                    print(f"[WARNING] Could not extract W&B run ID from checkpoint - will create new run")
            
            # Merge config overrides
            all_config_overrides = []
            if config_path:
                all_config_overrides.append(config_path)
            if config_overrides:
                all_config_overrides.extend(config_overrides)
            
            # Execute resume
            return self._execute_resume_from_checkpoint(
                checkpoint_path, 
                config_overrides=all_config_overrides if all_config_overrides else None,
                wandb_run_override=wandb_run_override,
                dry_run=dry_run,
                extra_cli_args=extra_cli_args
            )
            
        except Exception as e:
            print(f"❌ Failed to execute resume from artifact: {e}")
            return False

    def _execute_resume_with_paths(self, checkpoint_path, config_overrides: Optional[List] = None,
                                  wandb_run_id: str = None, dry_run: bool = False,
                                  extra_cli_args: Optional[List[str]] = None) -> bool:
        """Execute resume with given paths."""
        # Convert Path objects to strings if needed
        checkpoint_path = str(checkpoint_path) if checkpoint_path else None
        if config_overrides:
            config_overrides = [str(cfg) for cfg in config_overrides]
        
        return self._execute_resume_from_checkpoint(
            checkpoint_path, 
            config_overrides=config_overrides, 
            wandb_run_override=wandb_run_id,
            dry_run=dry_run,
            extra_cli_args=extra_cli_args
        )

    def _validate_checkpoint_path(self, checkpoint_path: str) -> bool:
        """Validate that checkpoint path exists and is readable."""
        try:
            from pathlib import Path
            return Path(checkpoint_path).exists()
        except Exception:
            return False

    def _validate_checkpoint_artifact(self, artifact_reference: str) -> bool:
        """Validate that W&B artifact exists and is accessible."""
        try:
            import wandb
            api = wandb.Api()
            _ = api.artifact(artifact_reference, type="model")
            return True
        except Exception:
            return False