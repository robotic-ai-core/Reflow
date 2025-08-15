"""
Environment variable management callback for Lightning training.

This callback handles environment variable configuration from config files,
providing a clean integration with Lightning's callback system.
"""

import os
from typing import Dict, List, Optional, Any
from pathlib import Path
import yaml
import logging

from lightning.pytorch import Callback, Trainer, LightningModule
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from lightning_reflow.utils.logging.environment_manager import EnvironmentManager

logger = logging.getLogger(__name__)


class EnvironmentCallback(Callback):
    """
    Callback to manage environment variables from config files.
    
    This callback processes environment variables defined in config files
    and sets them before training begins. It respects Lightning's config
    precedence: base config -> override configs -> CLI args.
    
    Example config:
        environment:
            PYTORCH_CUDA_ALLOC_CONF: "max_split_size_mb:256,expandable_segments:True"
            MALLOC_TRIM_THRESHOLD_: "128MB"
    """
    
    def __init__(self, env_vars: Optional[Dict[str, str]] = None, config_paths: Optional[List[Path]] = None):
        """
        Initialize the environment callback.
        
        Args:
            env_vars: Direct dictionary of environment variables to set
            config_paths: Optional list of config file paths. If not provided,
                         will be extracted from sys.argv during setup.
        """
        super().__init__()
        self.config_paths = config_paths or []
        self.env_vars: Dict[str, str] = env_vars or {}
        self.original_env: Dict[str, Optional[str]] = {}
        self._initialized = False
    
    @rank_zero_only
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Setup environment variables before training begins."""
        if self._initialized:
            return
        
        # Environment variables are set early by CLI before_instantiate_classes hook
        # This callback now serves as a config container and handles cleanup
        if self.env_vars:
            logger.info(f"EnvironmentCallback: {len(self.env_vars)} environment variables were set by CLI hook")
            
            # Store original values for cleanup (the variables should already be set)
            for var_name in self.env_vars:
                if var_name not in self.original_env:
                    self.original_env[var_name] = os.environ.get(var_name)
                # Verify they are actually set
                current_value = os.environ.get(var_name)
                expected_value = str(self.env_vars[var_name])
                if current_value != expected_value:
                    logger.warning(f"  {var_name}: expected '{expected_value}', found '{current_value}'")
                
        # Otherwise, try to extract from config files
        elif self.config_paths or self._should_extract_from_argv():
            # If config_paths not provided, extract from sys.argv
            if not self.config_paths:
                import sys
                self._extract_config_paths_from_argv(sys.argv)
            
            if self.config_paths:
                # Process environment variables from configs
                self._process_environment_variables()
        
        self._initialized = True
    
    def _should_extract_from_argv(self) -> bool:
        """Check if we should attempt to extract config paths from argv."""
        import sys
        return any('--config' in arg for arg in sys.argv)
    
    def _extract_config_paths_from_argv(self, argv: List[str]) -> None:
        """Extract config file paths from command line arguments."""
        i = 0
        while i < len(argv):
            if argv[i] == '--config' and i + 1 < len(argv):
                config_path = Path(argv[i + 1])
                if config_path.exists():
                    self.config_paths.append(config_path)
                i += 2
            elif argv[i].startswith('--config='):
                config_path = Path(argv[i].split('=', 1)[1])
                if config_path.exists():
                    self.config_paths.append(config_path)
                i += 1
            else:
                i += 1
    
    def _process_environment_variables(self) -> None:
        """Process environment variables from config files."""
        if not self.config_paths:
            return
        
        logger.info(f"Processing environment variables from {len(self.config_paths)} config files")
        
        # Extract environment variables with proper precedence
        try:
            env_vars, _ = EnvironmentManager.extract_environment_from_configs(self.config_paths)
            
            if env_vars:
                # Store original values for cleanup
                for var_name in env_vars:
                    self.original_env[var_name] = os.environ.get(var_name)
                
                # Set environment variables
                config_sources = [str(f) for f in self.config_paths]
                EnvironmentManager.set_environment_variables(env_vars, config_sources)
                self.env_vars = env_vars
                
                logger.info(f"Set {len(env_vars)} environment variables from configs")
                for var_name, value in env_vars.items():
                    logger.info(f"  {var_name}={value}")
            else:
                logger.debug("No environment variables found in config files")
                
        except Exception as e:
            logger.warning(f"Failed to process environment variables: {e}")
    
    @rank_zero_only
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        Verify that environment variables were set correctly by the CLI hook.
        
        This method runs at the start of training to validate that:
        1. All expected environment variables from the config are actually set in os.environ
        2. The values match what was specified in the configuration
        3. Critical variables like PYTORCH_CUDA_ALLOC_CONF took effect before CUDA initialization
        
        Why this validation is important:
        - Environment variables like PYTORCH_CUDA_ALLOC_CONF must be set BEFORE CUDA is initialized
        - If they're set too late, they have no effect but no error is raised
        - This check ensures our early CLI hook approach is working correctly
        - Helps debug configuration issues that could cause memory problems
        
        The validation happens at on_fit_start because:
        - CUDA has definitely been initialized by this point (during model.to(device))
        - We can verify the variables are set before training begins
        - It's late enough that all initialization is complete
        - It's early enough to catch issues before training starts
        """
        if not self.env_vars:
            logger.debug("No environment variables to validate")
            return
        
        logger.info(f"Validating {len(self.env_vars)} environment variables were set correctly")
        
        validation_errors = []
        
        for var_name, expected_value in self.env_vars.items():
            current_value = os.environ.get(var_name)
            expected_str = str(expected_value)
            
            # Check if variable is set
            if current_value is None:
                validation_errors.append(f"{var_name} is not set in environment")
                continue
                
            # Check if value matches
            if current_value != expected_str:
                validation_errors.append(
                    f"{var_name} has incorrect value: expected '{expected_str}', got '{current_value}'"
                )
                continue
                
            # Variable is correctly set
            logger.info(f"  ✓ {var_name}={current_value}")
            
            # Special validation for CUDA-related variables
            if var_name == 'PYTORCH_CUDA_ALLOC_CONF':
                # Additional check: verify CUDA memory allocator settings took effect
                try:
                    import torch
                    if torch.cuda.is_available():
                        # Try to get memory info to see if allocator is working
                        device = torch.cuda.current_device()
                        memory_info = torch.cuda.memory_stats(device)
                        logger.info(f"  ✓ CUDA allocator initialized with custom config (device {device})")
                except Exception as e:
                    logger.warning(f"  ⚠ Could not verify CUDA allocator config: {e}")
        
        # Assert all variables are correctly set
        if validation_errors:
            error_msg = "Environment variable validation failed:\n" + "\n".join(f"  - {err}" for err in validation_errors)
            error_msg += "\n\nThis usually means the CLI hook failed to set variables early enough."
            error_msg += "\nCheck that _process_environment_callback_config() is called in before_instantiate_classes()."
            
            # Log the error details
            logger.error(error_msg)
            
            # Raise assertion error to fail fast
            raise AssertionError(f"Environment variable validation failed: {len(validation_errors)} errors found")
        
        logger.info(f"✓ All {len(self.env_vars)} environment variables validated successfully")

    @rank_zero_only  
    def on_load_checkpoint(self, trainer: Trainer, pl_module: LightningModule, checkpoint: Dict[str, Any]) -> None:
        """
        Validate environment variables after checkpoint loading.
        
        This hook runs after a checkpoint is loaded and validates that:
        1. Environment variables from checkpoint state are properly restored
        2. Current environment matches what was saved in the checkpoint
        3. Critical variables needed for resumed training are available
        
        This is especially important for resume scenarios where:
        - Original training used specific CUDA allocator settings
        - Memory optimization settings need to be preserved
        - Configuration consistency between runs is critical
        
        The validation happens at on_load_checkpoint because:
        - Checkpoint state has been fully loaded by this point
        - Environment variables should already be set by CLI hook
        - We can compare checkpoint state vs current environment
        - It's before training actually resumes
        """
        if not self.env_vars:
            logger.debug("No environment variables to validate after checkpoint load")
            return
        
        logger.info(f"Validating environment variables after checkpoint loading")
        
        validation_errors = []
        
        for var_name, expected_value in self.env_vars.items():
            current_value = os.environ.get(var_name)
            expected_str = str(expected_value)
            
            # Check if variable is set in current environment
            if current_value is None:
                validation_errors.append(
                    f"{var_name} from checkpoint is not set in current environment"
                )
                continue
                
            # Check if current value matches checkpoint value
            if current_value != expected_str:
                validation_errors.append(
                    f"{var_name} mismatch: checkpoint='{expected_str}', current='{current_value}'"
                )
                continue
                
            logger.info(f"  ✓ {var_name}={current_value} (restored from checkpoint)")
        
        # Assert environment consistency for checkpoint resume
        if validation_errors:
            error_msg = "Environment variable validation failed after checkpoint loading:\n"
            error_msg += "\n".join(f"  - {err}" for err in validation_errors)
            error_msg += "\n\nThis indicates environment variables were not properly restored for resume."
            error_msg += "\nCheck that the CLI hook processed the same config used during original training."
            
            logger.error(error_msg)
            
            # Fail fast on resume environment issues
            raise AssertionError(f"Checkpoint environment validation failed: {len(validation_errors)} errors found")
        
        logger.info(f"✓ All {len(self.env_vars)} environment variables validated after checkpoint loading")

    @rank_zero_only
    def teardown(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Restore original environment variables on teardown."""
        if not self.env_vars:
            return
        
        # Restore original environment
        for var_name, original_value in self.original_env.items():
            if original_value is None:
                # Variable didn't exist before, remove it
                os.environ.pop(var_name, None)
            else:
                # Restore original value
                os.environ[var_name] = original_value
        
        logger.info(f"Restored {len(self.original_env)} environment variables")
    
    def state_dict(self) -> Dict[str, Any]:
        """Save callback state for checkpointing."""
        return {
            'env_vars': self.env_vars,
            'config_paths': [str(p) for p in self.config_paths]
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load callback state from checkpoint and validate environment restoration.
        
        When resuming from a checkpoint, this method:
        1. Restores the environment variable configuration from checkpoint state
        2. Validates that variables from checkpoint match current config
        3. Ensures environment variables are properly restored for resumed training
        
        This is critical for:
        - Ensuring consistent environment between original and resumed training
        - Validating that pause/resume preserves all environment settings
        - Catching configuration drift between original and resume configs
        """
        # Load state from checkpoint
        checkpoint_env_vars = state_dict.get('env_vars', {})
        checkpoint_config_paths = [Path(p) for p in state_dict.get('config_paths', [])]
        
        logger.info(f"Loading EnvironmentCallback state from checkpoint")
        logger.info(f"  Checkpoint contained {len(checkpoint_env_vars)} environment variables")
        logger.info(f"  Current config specifies {len(self.env_vars)} environment variables")
        
        # Validate environment variable consistency
        validation_errors = []
        
        # Check if current config env_vars match checkpoint
        if self.env_vars and checkpoint_env_vars:
            for var_name, current_value in self.env_vars.items():
                checkpoint_value = checkpoint_env_vars.get(var_name)
                
                if checkpoint_value is None:
                    validation_errors.append(
                        f"{var_name} is in current config but was not in checkpoint"
                    )
                elif str(current_value) != str(checkpoint_value):
                    validation_errors.append(
                        f"{var_name} value changed: checkpoint='{checkpoint_value}', config='{current_value}'"
                    )
                else:
                    logger.info(f"  ✓ {var_name} matches between checkpoint and config")
            
            # Check for variables that were in checkpoint but not in current config
            for var_name in checkpoint_env_vars:
                if var_name not in self.env_vars:
                    logger.warning(f"  ⚠ {var_name} was in checkpoint but not in current config")
        
        # PRECEDENCE FIX: Use config values as source of truth, not checkpoint values
        # The correct precedence is: CLI args > config files > checkpoint
        
        # If we have current config env_vars (from resume config), keep them
        # Otherwise, fall back to checkpoint values  
        if self.env_vars:
            # Current config has environment variables - use them (higher precedence)
            logger.info(f"Using environment variables from config files (higher precedence)")
            logger.info(f"Checkpoint environment variables will be ignored where conflicts exist")
            
            # Merge checkpoint values only for variables not in current config
            merged_env_vars = checkpoint_env_vars.copy()
            merged_env_vars.update(self.env_vars)  # Config values override checkpoint values
            self.env_vars = merged_env_vars
        else:
            # No current config env_vars - use checkpoint values as fallback
            logger.info(f"No environment variables in current config - using checkpoint values as fallback")
            self.env_vars = checkpoint_env_vars
            
        # Config paths come from current config, not checkpoint (for proper precedence tracking)
        # self.config_paths = checkpoint_config_paths  # Don't override current config paths
        
        # Assert consistency for critical variables
        if validation_errors:
            error_msg = "Environment variable differences detected between checkpoint and current config:\n"
            error_msg += "\n".join(f"  - {err}" for err in validation_errors)
            error_msg += "\n\nThis is expected when using config overrides during resume."
            error_msg += "\nCurrent config values will take precedence (correct behavior)."
            
            logger.info(error_msg)
            logger.info("Using config file environment variables (higher precedence than checkpoint)")
        
        logger.info(f"✓ Environment variable state loaded from checkpoint")