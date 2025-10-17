import time
import torch
import warnings
from pathlib import Path
from typing import Optional, Any, Dict

from lightning.pytorch import Trainer, LightningModule
from lightning.pytorch.callbacks import Callback

from .improved_keyboard_handler import ImprovedKeyboardHandler
from .pause_state_machine import PauseState, PauseStateMachine
from ..core.config_embedding_mixin import ConfigEmbeddingMixin
from ..monitoring.flow_progress_bar_callback import FlowProgressBarCallback
from ...utils.wandb.wandb_artifact_manager import WandbArtifactManager

class PauseCallback(FlowProgressBarCallback, ConfigEmbeddingMixin):
    """
    Interactive pause callback with production-grade validation boundary pausing.
    
    Features:
    - Interactive pause via keyboard (default: 'p' key)
    - Optional W&B upload during pause (default: 'w' key)  
    - Validation boundary pausing for robustness
    - Production-grade error handling and fallbacks
    - Configurable progress bar pause countdown (disabled by default)
    
    Args:
        checkpoint_dir: Directory for pause checkpoints
        enable_pause: Enable interactive pause functionality
        pause_key: Key to trigger pause (default: 'p')
        upload_key: Key to toggle W&B upload (default: 'w')
        show_pause_countdown: Show "⏸️ Pause in: X steps" on progress bar (default: False)
        ... (other FlowProgressBarCallback args)
    
    Example:
        # Default usage (countdown disabled to save screen space)
        callback = PauseCallback(checkpoint_dir="my_checkpoints")
        
        # Enable countdown display if desired
        callback = PauseCallback(
            checkpoint_dir="my_checkpoints", 
            show_pause_countdown=True
        )
    """
    def __init__(
        self,
        checkpoint_dir: str = "pause_checkpoints",
        enable_pause: bool = True,
        pause_key: str = 'p',
        upload_key: str = 'w',
        debounce_interval: float = 0.3,
        startup_grace_period: float = 2.0,  # New: grace period to ignore automated input
        refresh_rate: int = 1,
        bar_colour: str = "#fcac17",
        global_bar_metrics: list = None,
        interval_bar_metrics: list = None,
        logging_interval: str = "step",
        enable_pause_context_management: bool = True,
        skip_dependency_check: bool = False,  # Added for test compatibility
        show_pause_countdown: bool = False,  # Disable pause countdown by default to save screen space
        save_rng_states: bool = True,  # Enable scientific reproducibility by default
    ):
        super().__init__(
            refresh_rate=refresh_rate,
            bar_colour=bar_colour,
            global_bar_metrics=global_bar_metrics,
            interval_bar_metrics=interval_bar_metrics,
            logging_interval=logging_interval,
        )
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.enable_pause = enable_pause
        self.pause_key = pause_key
        self.upload_key = upload_key
        self.debounce_interval = debounce_interval
        self.startup_grace_period = startup_grace_period  # New: store grace period
        self.enable_pause_context_management = enable_pause_context_management
        self.show_pause_countdown = show_pause_countdown
        
        # State management
        self._state_machine = PauseStateMachine()
        self._keyboard_handler: Optional[ImprovedKeyboardHandler] = None
        self._last_key_time = 0.0
        # Initialize shared WandB artifact manager
        self._wandb_manager = WandbArtifactManager(verbose=True)
        
        # Debug hook flag - when True, pauses are scheduled but not auto-executed
        self._debug_hooks_enabled = False
        
        # Initialize config embedding mixin (also stores sys.argv for resume commands)
        ConfigEmbeddingMixin.__init__(self)

        # Track last checkpoint path for HPO integration
        self.last_checkpoint_path = None

        # Register scientific reproducibility manager if enabled
        self.save_rng_states = save_rng_states
        if self.save_rng_states:
            from ...utils.checkpoint.scientific_reproducibility_state import ScientificReproducibilityState
            from ...utils.checkpoint.manager_state import register_manager
            self._reproducibility_manager = ScientificReproducibilityState()
            register_manager(self._reproducibility_manager)
            print("🔬 Scientific reproducibility enabled (RNG states will be saved)")

    @property
    def _pause_state(self) -> PauseState:
        return self._state_machine.state

    def is_pause_scheduled(self) -> bool:
        return self._state_machine.is_pause_scheduled()
    
    def is_pausing(self) -> bool:
        """Check if pause is currently being executed (for WandbArtifactCheckpoint compatibility)."""
        return self._state_machine.is_pause_scheduled()
    
    def is_upload_requested(self) -> bool:
        """Check if upload is requested during pause."""
        return self._state_machine.is_upload_requested()
    
    def is_upload_all_requested(self) -> bool:
        """Check if upload all checkpoints is requested during pause."""
        return self._state_machine.is_upload_all_requested()
    
    def get_last_checkpoint(self) -> Optional[Path]:
        """Get the last saved checkpoint path (for HPO integration)."""
        return self.last_checkpoint_path

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule):
        super().on_train_start(trainer, pl_module)
        
        # Check for debug hooks and trigger appropriate states
        self._check_debug_hooks()
        
        if self.enable_pause and trainer.is_global_zero:
            try:
                # Pass startup_grace_period to keyboard handler
                self._keyboard_handler = ImprovedKeyboardHandler(
                    debounce_interval=self.debounce_interval,
                    startup_grace_period=self.startup_grace_period
                )
                # Start monitoring instead of registering keys
                self._keyboard_handler.start_monitoring()
            except Exception as e:
                warnings.warn(f"Could not initialize keyboard handler: {e}. Disabling pause functionality.")
                self.enable_pause = False

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        
        # Check for keyboard input
        if self.enable_pause:
            self._check_keyboard_input()
        
        # Only execute pause immediately if not configured to use validation intervals
        # If val_check_interval is set, prefer to pause at validation boundaries
        # Also don't execute if debug hooks are enabled (for testing)
        if (self._state_machine.is_pause_scheduled() and 
            getattr(trainer, 'val_check_interval', None) is None and
            not self._debug_hooks_enabled):
            self._execute_immediate_pause(trainer, pl_module)

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule):
        super().on_validation_end(trainer, pl_module)
        
        # Production-grade validation boundary pause with comprehensive safety checks
        try:
            self._execute_validation_boundary_pause_if_needed(trainer, pl_module)
        except Exception as e:
            # Critical: Never let pause logic crash training
            print(f"❌ CRITICAL: Validation boundary pause failed: {e}")
            print(f"   Training will continue without pause. Please investigate.")
            # Reset pause state to prevent repeated failures
            self._state_machine.reset()
            
    def _execute_validation_boundary_pause_if_needed(self, trainer: Trainer, pl_module: LightningModule):
        """Execute validation boundary pause with comprehensive production safety checks."""
        
        # Safety check 1: Must have pause scheduled
        if not self._state_machine.is_pause_scheduled():
            return
            
        # Safety check 2: Skip during debug mode (for testing)
        if self._debug_hooks_enabled:
            return
            
        # Safety check 3: Skip during sanity validation (global_step == 0)
        if trainer.global_step <= 0:
            print(f"🔄 Pause scheduled but skipping during sanity validation (global_step={trainer.global_step})")
            return
            
        # Safety check 4: Verify trainer state is valid for checkpointing
        if not self._validate_trainer_state_for_pause(trainer, pl_module):
            print(f"❌ Trainer state invalid for pause - skipping pause at validation boundary")
            # Reset pause state to prevent repeated validation failures
            self._state_machine.reset()
            return
            
        # Safety check 5: Verify training is actually active (not finished/stopping)
        if hasattr(trainer, 'interrupted') and trainer.interrupted:
            print(f"🔄 Training already interrupted - skipping pause execution")
            return
            
        if hasattr(trainer, 'should_stop') and trainer.should_stop:
            print(f"🔄 Training should_stop=True - skipping pause execution")
            return
        
        print(f"🔄 Executing pause at validation boundary (global_step={trainer.global_step}, epoch={trainer.current_epoch})")
        
        # Execute the actual pause
        self._execute_validation_boundary_pause(trainer, pl_module)
        
    def _validate_trainer_state_for_pause(self, trainer: Trainer, pl_module: LightningModule) -> bool:
        """Validate that trainer and module state is safe for pause checkpoint creation."""
        
        # Check trainer has required attributes for checkpointing
        required_trainer_attrs = ['global_step', 'current_epoch', 'logger']
        for attr in required_trainer_attrs:
            if not hasattr(trainer, attr):
                print(f"❌ Trainer missing required attribute for pause: {attr}")
                return False
                
        # Check pl_module is valid
        if pl_module is None:
            print(f"❌ LightningModule is None - cannot create pause checkpoint")
            return False
            
        # Check checkpoint directory is accessible
        try:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"❌ Cannot access checkpoint directory {self.checkpoint_dir}: {e}")
            return False
            
        # Check we have disk space (basic check)
        try:
            import shutil
            free_space = shutil.disk_usage(self.checkpoint_dir).free
            if free_space < 100 * 1024 * 1024:  # Less than 100MB
                print(f"❌ Low disk space for pause checkpoint: {free_space / (1024*1024):.1f} MB")
                return False
        except Exception as e:
            print(f"⚠️ Could not check disk space: {e}")
            # Continue anyway - disk space check is optional
            
        return True
            
    def _is_end_of_epoch_validation(self, trainer: Trainer) -> bool:
        """Check if this validation is happening at the end of an epoch."""
        # If no val_check_interval is set, validation happens at epoch boundaries
        val_check_interval = getattr(trainer, 'val_check_interval', None)
        if val_check_interval is None:
            return True
        
        # If val_check_interval is set, check if we're at an epoch boundary
        # This is a heuristic - if global_step is divisible by total batches per epoch
        try:
            steps_per_epoch = len(trainer.train_dataloader)
            return (trainer.global_step % steps_per_epoch) == 0
        except:
            return False
            
    def on_train_end(self, trainer: Trainer, pl_module: LightningModule):
        if self._keyboard_handler:
            self._keyboard_handler.stop_monitoring()

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        super().on_train_epoch_end(trainer, pl_module)
        # DISABLED: Epoch boundary pausing removed for production robustness
        # Validation boundary pausing is more reliable and sufficient
        # if self._state_machine.is_pause_scheduled() and not self._debug_hooks_enabled:
        #     self._execute_epoch_boundary_pause(trainer, pl_module)
        pass
            
    def _execute_epoch_boundary_pause(self, trainer: Trainer, pl_module: LightningModule):
        should_upload = self._state_machine.is_upload_requested()
        checkpoint_path = self._get_checkpoint_path(trainer, upload=should_upload)
        
        # Save checkpoint normally first
        self._save_checkpoint(trainer, pl_module, checkpoint_path)
        
        # For epoch boundary pauses, we need to manually increment the epoch in the checkpoint
        # because Lightning hasn't incremented it yet when validation ends
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            checkpoint['epoch'] = trainer.current_epoch + 1
            torch.save(checkpoint, checkpoint_path)
        except Exception as e:
            print(f"Warning: Could not update epoch in checkpoint: {e}")
            
        # Handle upload and print resume commands with proper error handling
        artifact_path = None
        if should_upload:
            try:
                artifact_path = self._handle_wandb_upload(trainer, pl_module, str(checkpoint_path))
            except (ValueError, RuntimeError) as e:
                print(f"⚠️  Upload failed but pause will continue: {e}")
                artifact_path = None
        
        try:
            self._print_resume_commands(trainer, str(checkpoint_path), artifact_path)
        except ValueError as e:
            print(f"⚠️  Could not generate resume commands: {e}")
            print(f"💾 Checkpoint saved at: {checkpoint_path}")
        
        trainer.should_stop = True
        self._state_machine.reset()

    def _handle_pause_key(self):
        if self._debounce(): return
        if self._state_machine.toggle_pause():
            # State changed, print status message
            if self._state_machine.is_pause_scheduled():
                print("\n🔄 PAUSE scheduled - will pause at next validation boundary")
            else:
                print("\n❌ PAUSE cancelled - training will continue")

    def _handle_upload_key(self):
        if self._debounce(): return
        if self._state_machine.toggle_upload():
            # State changed, print status message
            if self._state_machine.is_upload_all_requested():
                print("\n📤📤 Upload ALL ENABLED - all checkpoints will be uploaded to W&B")
            elif self._state_machine.is_upload_requested():
                print("\n📤 Upload ENABLED - pause checkpoint will be uploaded to W&B")
            else:
                print("\n📤 Upload DISABLED - checkpoint will not be uploaded to W&B")

    # Backward compatibility methods for tests
    def handle_pause_key(self):
        """Alias for _handle_pause_key for backward compatibility."""
        return self._handle_pause_key()
    
    def _check_keyboard_input(self):
        """Check keyboard input for test compatibility."""
        if self._keyboard_handler:
            key = self._keyboard_handler.get_key()
            if key == self.pause_key:
                self._handle_pause_key()
            elif key == self.upload_key:
                self._handle_upload_key()

    def _debounce(self) -> bool:
        current_time = time.time()
        if current_time - self._last_key_time < self.debounce_interval:
            return True
        self._last_key_time = current_time
        return False

    def _get_checkpoint_path(self, trainer: Trainer, upload: bool = False) -> Path:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time())
        tag = "upload" if upload else "pause"
        filename = f"{tag}_epoch={trainer.current_epoch}_step={trainer.global_step}_{timestamp}.ckpt"
        return self.checkpoint_dir / filename
    
    def _save_checkpoint(self, trainer: Trainer, pl_module: LightningModule, checkpoint_path: Path):
        # Save checkpoint normally
        trainer.save_checkpoint(checkpoint_path)
        
        # Track the checkpoint path for HPO integration
        self.last_checkpoint_path = checkpoint_path
        
        # Add config metadata if the mixin is available
        # try:
        #     # Load the saved checkpoint to add metadata
        #     checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
        #     # Add config metadata using the mixin
        #     self.add_config_metadata(trainer, pl_module, checkpoint)
            
        #     # Save the checkpoint back with metadata
        #     torch.save(checkpoint, checkpoint_path)
        # except Exception as e:
        #     print(f"Warning: Could not add config metadata to checkpoint: {e}")

    def _execute_immediate_pause(self, trainer: Trainer, pl_module: LightningModule):
        should_upload = self._state_machine.is_upload_requested()
        checkpoint_path = self._get_checkpoint_path(trainer, upload=should_upload)
        self._save_checkpoint(trainer, pl_module, checkpoint_path)
        
        # Handle upload and print resume commands with proper error handling
        artifact_path = None
        if should_upload:
            try:
                artifact_path = self._handle_wandb_upload(trainer, pl_module, str(checkpoint_path))
            except (ValueError, RuntimeError) as e:
                print(f"⚠️  Upload failed but pause will continue: {e}")
                artifact_path = None
        
        try:
            self._print_resume_commands(trainer, str(checkpoint_path), artifact_path)
        except ValueError as e:
            print(f"⚠️  Could not generate resume commands: {e}")
            print(f"💾 Checkpoint saved at: {checkpoint_path}")
        
        trainer.should_stop = True
        self._state_machine.reset()

    def _execute_validation_boundary_pause(self, trainer: Trainer, pl_module: LightningModule):
        """Execute pause at validation boundary with production-grade error handling."""
        
        should_upload = self._state_machine.is_upload_requested()
        checkpoint_path = None
        artifact_path = None
        
        try:
            # Step 1: Generate checkpoint path (fail fast if issues)
            checkpoint_path = self._get_checkpoint_path(trainer, upload=should_upload)
            print(f"💾 Creating pause checkpoint at: {checkpoint_path}")
            
            # Step 2: Save checkpoint (critical operation) - no additional print needed
            self._save_checkpoint_with_validation(trainer, pl_module, checkpoint_path)
            print(f"✅ Pause checkpoint saved successfully")
            
            # Step 3: Handle upload (optional operation - should not fail pause)
            if should_upload:
                artifact_path = self._handle_upload_with_fallback(trainer, pl_module, str(checkpoint_path))
            
            # Step 4: Set trainer stop and reset state
            trainer.should_stop = True
            self._state_machine.reset()
            
            print(f"🔄 Training paused successfully at validation boundary")
            
            # Step 5: Print resume commands AFTER setting should_stop
            # This allows any pending checkpoint operations to complete first
            self._print_resume_commands_with_fallback(trainer, str(checkpoint_path), artifact_path)
            
        except Exception as e:
            # Critical failure handling
            print(f"❌ CRITICAL: Pause execution failed: {e}")
            
            # Try to clean up partial checkpoint if it exists
            if checkpoint_path and checkpoint_path.exists():
                try:
                    checkpoint_path.unlink()
                    print(f"🧹 Cleaned up partial checkpoint: {checkpoint_path}")
                except Exception as cleanup_error:
                    print(f"⚠️ Could not clean up partial checkpoint: {cleanup_error}")
            
            # Reset state to prevent repeated failures
            self._state_machine.reset()
            
            # Re-raise to trigger the outer exception handler
            raise
            
    def _save_checkpoint_with_validation(self, trainer: Trainer, pl_module: LightningModule, checkpoint_path: Path):
        """Save checkpoint with validation and atomic operation."""
        
        # Create temporary checkpoint path for atomic operation
        temp_checkpoint_path = checkpoint_path.with_suffix('.tmp')
        
        try:
            # Save to temporary file first
            trainer.save_checkpoint(temp_checkpoint_path)
            
            # Validate the checkpoint was created and is readable
            if not temp_checkpoint_path.exists():
                raise RuntimeError(f"Checkpoint was not created at {temp_checkpoint_path}")
                
            checkpoint_size = temp_checkpoint_path.stat().st_size
            if checkpoint_size < 1024:  # Less than 1KB is suspicious
                raise RuntimeError(f"Checkpoint file too small ({checkpoint_size} bytes) - likely corrupted")
            
            # Try to load and validate the checkpoint structure
            try:
                checkpoint_dict = torch.load(temp_checkpoint_path, map_location='cpu', weights_only=False)
                required_keys = ['state_dict', 'epoch', 'global_step']
                missing_keys = [key for key in required_keys if key not in checkpoint_dict]
                if missing_keys:
                    raise RuntimeError(f"Checkpoint missing required keys: {missing_keys}")
            except Exception as e:
                raise RuntimeError(f"Checkpoint validation failed: {e}")
            
            # Add config metadata if the mixin is available
            try:
                # Load the saved checkpoint to add metadata
                checkpoint = torch.load(temp_checkpoint_path, map_location='cpu', weights_only=False)
                
                # Add config metadata using the mixin
                self.add_config_metadata(trainer, pl_module, checkpoint)
                
                # Save back to temporary file
                torch.save(checkpoint, temp_checkpoint_path)
                print(f"✅ Added config metadata to pause checkpoint")
                
            except Exception as e:
                print(f"⚠️ Could not add config metadata to checkpoint: {e}")
                # Continue without metadata - not critical for pause functionality
            
            # Atomic move from temporary to final location
            temp_checkpoint_path.rename(checkpoint_path)
            print(f"✅ Checkpoint atomically saved to {checkpoint_path} ({checkpoint_size:,} bytes)")
            
        except Exception as e:
            # Clean up temporary file on any failure
            if temp_checkpoint_path.exists():
                try:
                    temp_checkpoint_path.unlink()
                except Exception:
                    pass
            raise RuntimeError(f"Failed to save pause checkpoint: {e}")
            
    def _handle_upload_with_fallback(self, trainer: Trainer, pl_module: LightningModule, checkpoint_path: str) -> Optional[str]:
        """Handle W&B upload with comprehensive fallback."""
        
        try:
            artifact_path = self._handle_wandb_upload(trainer, pl_module, checkpoint_path)
            if artifact_path:
                print(f"✅ Pause checkpoint uploaded to W&B: {artifact_path}")
                return artifact_path
            else:
                print(f"⚠️ W&B upload returned None - checkpoint saved locally only")
                return None
                
        except (ValueError, RuntimeError) as e:
            print(f"⚠️ W&B upload failed but pause will continue: {e}")
            print(f"💾 Checkpoint available locally at: {checkpoint_path}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error during W&B upload: {e}")
            print(f"💾 Checkpoint available locally at: {checkpoint_path}")
            return None
            
    def _print_resume_commands_with_fallback(self, trainer: Trainer, checkpoint_path: str, artifact_path: Optional[str] = None):
        """Print resume commands with fallback for errors."""
        
        try:
            self._print_resume_commands(trainer, checkpoint_path, artifact_path)
        except ValueError as e:
            print(f"⚠️ Could not generate resume commands: {e}")
            # Provide basic fallback information
            print(f"💾 Checkpoint saved at: {checkpoint_path}")
            if artifact_path:
                print(f"☁️ W&B artifact: {artifact_path}")
            print(f"📝 Use standard Lightning resume: --ckpt_path {checkpoint_path}")
        except Exception as e:
            print(f"❌ Unexpected error generating resume commands: {e}")
            print(f"💾 Checkpoint saved at: {checkpoint_path}")
            print(f"📝 Manually resume with: --ckpt_path {checkpoint_path}")
            
    def _upload_pause_checkpoint_artifact(self, wandb_callback, trainer: Trainer, checkpoint_path: str) -> Optional[str]:
        """
        Upload pause checkpoint artifact using shared WandB artifact manager.
        
        Args:
            wandb_callback: WandbArtifactCheckpoint instance for compatibility
            trainer: PyTorch Lightning trainer
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Full artifact path if upload successful (e.g. "entity/project/artifact:version"), None otherwise
            
        Raises:
            ValueError: If required inputs are invalid
            RuntimeError: If W&B manager is not available or configured incorrectly
        """
        # Fail early: Validate critical inputs
        if trainer is None:
            raise ValueError("Trainer cannot be None for artifact upload")
        if not checkpoint_path or not Path(checkpoint_path).exists():
            raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
        if not hasattr(self, '_wandb_manager') or self._wandb_manager is None:
            raise RuntimeError("W&B artifact manager is not initialized")
        
        # Fail early: Check W&B run availability
        wandb_run = self._wandb_manager.get_wandb_run(trainer)
        if not wandb_run:
            raise RuntimeError("No active W&B run found - cannot upload artifacts")
        
        # Fail early: Validate trainer state
        if not hasattr(trainer, 'lightning_module') or trainer.lightning_module is None:
            raise ValueError("Trainer must have a valid lightning_module for upload")
        
        try:
            # Create pause-specific metadata
            extra_metadata = {
                "pause_type": "manual_pause",
                "checkpoint_type": "pause_checkpoint",
                "pause_callback_version": "2.1"
            }
            
            # Upload using shared artifact manager - returns full artifact path
            artifact_path = self._wandb_manager.upload_checkpoint_artifact(
                trainer=trainer,
                pl_module=trainer.lightning_module,
                filepath=checkpoint_path,
                ckpt_type="pause",
                aliases=["pause", "latest"],
                score=None,  # Pause checkpoints don't have scores
                epoch=trainer.current_epoch,
                step=trainer.global_step,
                wandb_run=wandb_run,
                extra_metadata=extra_metadata
            )
            
            if not artifact_path:
                raise RuntimeError("Artifact upload returned None - upload failed")
                
            return artifact_path  # Returns full path like "entity/project/artifact:version"
            
        except (AttributeError, KeyError) as e:
            # Specific exceptions for missing attributes/keys
            raise RuntimeError(f"Missing required attribute for artifact upload: {e}") from e
        except Exception as e:
            # Re-raise with more context
            raise RuntimeError(f"Artifact upload failed: {e}") from e

    def _handle_wandb_upload(self, trainer: Trainer, pl_module: LightningModule, checkpoint_path: str) -> Optional[str]:
        """
        Handle W&B upload and return artifact path if successful.
        
        Args:
            trainer: PyTorch Lightning trainer
            pl_module: Lightning module 
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Full artifact path if upload successful, None if W&B not available (graceful degradation)
            
        Raises:
            ValueError: If required inputs are invalid
        """
        # Fail early: Validate critical inputs
        if trainer is None:
            raise ValueError("Trainer cannot be None for W&B upload")
        if not hasattr(trainer, 'callbacks') or trainer.callbacks is None:
            raise ValueError("Trainer must have callbacks list for W&B upload")
        
        # Find WandbArtifactCheckpoint callback
        wandb_callback = None
        for callback in trainer.callbacks:
            # Check for WandbArtifactCheckpoint callback with upload_pause_checkpoint method
            if hasattr(callback, 'upload_pause_checkpoint'):
                wandb_callback = callback
                break
            # Fallback to old method name for backward compatibility
            elif hasattr(callback, '_upload_pause_checkpoint_artifact'):
                wandb_callback = callback
                break
        
        # Graceful degradation: No W&B callback is not an error, just no upload
        if not wandb_callback:
            print(f"⚠️  No W&B callback found - checkpoint saved locally only")
            return None
        
        # Attempt upload with proper error handling
        try:
            # Use new method if available, otherwise use fallback
            if hasattr(wandb_callback, 'upload_pause_checkpoint'):
                artifact_path = wandb_callback.upload_pause_checkpoint(trainer, trainer.lightning_module, checkpoint_path)
            else:
                artifact_path = self._upload_pause_checkpoint_artifact(wandb_callback, trainer, checkpoint_path)
            print(f"✅ Pause checkpoint uploaded to W&B successfully")
            return artifact_path
            
        except (ValueError, RuntimeError) as e:
            # Expected errors from upload method - log and gracefully degrade
            print(f"❌ Failed to upload pause checkpoint to W&B: {e}")
            return None
        except Exception as e:
            # Unexpected errors - re-raise with context
            raise RuntimeError(f"Unexpected error during W&B upload: {e}") from e

    def _print_resume_commands(self, trainer: Trainer, checkpoint_path: str, artifact_path: Optional[str] = None) -> None:
        """
        Print resume commands based on what's available.
        
        Args:
            trainer: PyTorch Lightning trainer (unused but kept for interface consistency)
            checkpoint_path: Path to local checkpoint file
            artifact_path: Optional W&B artifact path
            
        Raises:
            ValueError: If required inputs are invalid
        """
        # Fail early: Validate critical inputs
        if not checkpoint_path:
            raise ValueError("Checkpoint path cannot be empty")
        if not hasattr(self, '_original_argv') or not self._original_argv:
            raise ValueError("Original argv not stored - cannot generate resume commands")
        
        # Extract script name and ensure "python" is included for copy-paste convenience
        script_name = self._original_argv[0] if self._original_argv else "train_lightning.py"

        # Detect if running via module and provide user-friendly command
        if ("__main__.py" in script_name or
            script_name.endswith("/lightning_reflow/cli/__main__.py") or
            "lightning_reflow" in script_name and "__main__" in script_name):
            # Running via python -m lightning_reflow.cli - suggest user-friendly command
            # Use the actual script that invoked the CLI rather than hardcoding train_lightning.py
            import sys
            if len(sys.argv) > 0 and sys.argv[0].endswith('.py'):
                script_command = f"python {sys.argv[0]}"
            else:
                # Fallback to a generic command if we can't determine the actual script
                script_command = "python train_lightning.py"
        elif not script_name.startswith("python"):
            script_command = f"python {script_name}"
        else:
            script_command = script_name
        
        print(f"\n🔄 Training paused. Resume options:")
        print(f"📁 Local resume:    {script_command} resume --checkpoint-path {checkpoint_path}")
        
        if artifact_path:
            print(f"☁️  W&B resume:     {script_command} resume --checkpoint-artifact {artifact_path}")
        
        # Also show the legacy method for backward compatibility
        # Filter out any existing --ckpt_path arguments to avoid duplicates/conflicts
        filtered_argv = []
        i = 0
        while i < len(self._original_argv):
            if self._original_argv[i] == '--ckpt_path':
                # Skip both the flag and its value
                i += 2
            elif self._original_argv[i].startswith('--ckpt_path='):
                # Skip combined flag=value format
                i += 1
            else:
                filtered_argv.append(self._original_argv[i])
                i += 1

        legacy_command = f"python {' '.join(filtered_argv)}" if not filtered_argv[0].startswith("python") else ' '.join(filtered_argv)
        print(f"📁 Legacy method:   {legacy_command} --ckpt_path {checkpoint_path}")
        if artifact_path:
            # For legacy method, we can use --resume_from_wandb flag
            print(f"☁️  Legacy W&B:     {legacy_command} --resume_from_wandb {artifact_path}")

    def _check_debug_hooks(self):
        """Check for debug hook environment variables and trigger appropriate states."""
        import os
        
        debug_pause = os.getenv("DEBUG_PAUSE_HOOK", "false").lower() == "true"
        debug_upload = os.getenv("DEBUG_UPLOAD_HOOK", "false").lower() == "true"
        
        if debug_pause:
            self._debug_hooks_enabled = True  # Mark that debug hooks are active
            if debug_upload:
                # Schedule pause with upload
                self._state_machine.toggle_pause()  # First toggle to pause
                self._state_machine.toggle_upload()  # Then toggle upload
            else:
                # Schedule pause without upload
                self._state_machine.toggle_pause()
                
    def _get_interval_pause_status_suffix(self) -> str:
        """Get pause status suffix for interval progress bar based on state machine."""
        from .pause_state_machine import PauseState
        
        if self._pause_state == PauseState.RUNNING:
            return f" - Press '{self.pause_key}' to pause"
        elif self._pause_state == PauseState.PAUSE_SCHEDULED_NO_UPLOAD:
            return f" - {self.pause_key}: to unpause, {self.upload_key}: upload pause ckpt"
        elif self._pause_state == PauseState.PAUSE_SCHEDULED_WITH_UPLOAD:
            return f" - {self.pause_key}: to unpause, {self.upload_key}: upload all (✓)"
        elif self._pause_state == PauseState.PAUSE_SCHEDULED_UPLOAD_ALL:
            return f" - {self.pause_key}: to unpause, {self.upload_key}: cancel upload (✓✓)"
        else:
            return f" - Press '{self.pause_key}' to pause"  # Fallback
    
    def _get_global_pause_status_suffix(self) -> str:
        """Get pause status suffix for global progress bar."""
        return ""  # Keep global bar clean
    
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Set model reference for the reproducibility manager and capture config."""
        super().on_fit_start(trainer, pl_module)

        # Capture and cache config at training start (eliminates false stale config warnings)
        self.capture_and_cache_config(trainer)

        if self.save_rng_states and hasattr(self, '_reproducibility_manager'):
            self._reproducibility_manager.set_references(model=pl_module, trainer=trainer)

    def on_load_checkpoint(self, trainer: Trainer, pl_module: LightningModule, checkpoint: Dict[str, Any]) -> None:
        """Trigger post-restoration hooks for scientific reproducibility."""
        super().on_load_checkpoint(trainer, pl_module, checkpoint)

        # Update _original_argv to reflect current command line when resuming
        # This ensures the legacy resume command shows the correct checkpoint path
        import sys
        self._original_argv = sys.argv.copy()

        if self.save_rng_states and hasattr(self, '_reproducibility_manager'):
            # Update references and trigger post-restoration
            self._reproducibility_manager.set_references(model=pl_module, trainer=trainer)
            self._reproducibility_manager.post_restoration_hook()

    def on_save_checkpoint(self, trainer: Trainer, pl_module: LightningModule, checkpoint: Dict[str, Any]) -> None:
        """Save pause callback metadata to checkpoint."""
        # Call parent to handle any base class logic
        super().on_save_checkpoint(trainer, pl_module, checkpoint)
        
        # Add config metadata from the mixin with the specific key
        self.add_config_metadata(trainer, pl_module, checkpoint, metadata_key='pause_callback_metadata')
        
        # Capture manager states
        try:
            from lightning_reflow.utils.checkpoint.manager_state import capture_all_manager_states
            manager_states = capture_all_manager_states()
            if 'pause_callback_metadata' not in checkpoint:
                checkpoint['pause_callback_metadata'] = {}
            checkpoint['pause_callback_metadata']['manager_states'] = manager_states
            print(f"📦 Captured {len(manager_states)} manager states")
        except Exception as e:
            print(f"⚠️ Failed to capture manager states: {e}")
        
        # Add pause-specific metadata
        pause_metadata = {
            'pause_timestamp': time.time(),
            'pause_state': self._state_machine.state.value,
            'checkpoint_dir': str(self.checkpoint_dir),
            'enable_pause': self.enable_pause,
            'pause_point': 'manual',  # Could be 'manual', 'validation', 'epoch_end', etc.
            'upload_requested': self._state_machine.is_upload_requested(),
        }
        
        # Update the metadata with pause-specific fields
        if 'pause_callback_metadata' in checkpoint:
            checkpoint['pause_callback_metadata'].update(pause_metadata)
        else:
            checkpoint['pause_callback_metadata'] = pause_metadata

    def _update_interval_bar_postfix(self) -> None:
        """Override to add validation timing info when pause is scheduled."""
        if self.current_interval_bar is None or not self._interval_metrics:
            return
        
        # Get the base postfix with metrics
        base_postfix = self._format_metrics_postfix(self._interval_metrics)
        
        # Add steps until next validation if pause is scheduled AND countdown is enabled
        if (self._state_machine.is_pause_scheduled() and 
            self._trainer and 
            self.show_pause_countdown):
            # Import the method from parent if needed
            if hasattr(super(), '_get_steps_until_next_validation'):
                steps_until_val = super()._get_steps_until_next_validation(self._trainer, self._current_batch_idx or 0)
            else:
                # Calculate it here if parent doesn't have the method
                steps_until_val = self._calculate_steps_until_validation(self._trainer, self._current_batch_idx or 0)
            
            if steps_until_val is not None:
                base_postfix += f" | ⏸️ Pause in: {steps_until_val} steps"
        
        self.current_interval_bar.set_postfix_str(base_postfix)
    
    def _calculate_steps_until_validation(self, trainer: Trainer, batch_idx: int) -> Optional[int]:
        """Calculate actual steps remaining until the next validation."""
        check_val_every_n_epoch = getattr(trainer, 'check_val_every_n_epoch', None)
        
        if check_val_every_n_epoch:
            # For epoch-based validation
            current_epoch = trainer.current_epoch
            next_val_epoch = ((current_epoch // check_val_every_n_epoch) + 1) * check_val_every_n_epoch
            epochs_until_val = next_val_epoch - current_epoch
            
            # Get steps per epoch
            num_training_batches = None
            if hasattr(trainer, 'num_training_batches') and trainer.num_training_batches != float('inf'):
                num_training_batches = trainer.num_training_batches
            
            if num_training_batches:
                # Steps remaining in current epoch
                steps_left_in_epoch = num_training_batches - (batch_idx + 1)
                # Plus steps in remaining full epochs
                steps_in_full_epochs = (epochs_until_val - 1) * num_training_batches
                return steps_left_in_epoch + steps_in_full_epochs
        
        elif trainer.val_check_interval:
            # For step-based validation
            val_interval = trainer.val_check_interval
            if isinstance(val_interval, int):
                # Convert to training batches for consistency
                accumulate_grad_batches = getattr(trainer, 'accumulate_grad_batches', 1)
                current_training_batch = trainer.global_step * accumulate_grad_batches
                batches_since_last_val = current_training_batch - self._last_validation_batch
                return val_interval - batches_since_last_val
            elif isinstance(val_interval, float) and val_interval > 1.0:
                # Convert to training batches for consistency
                accumulate_grad_batches = getattr(trainer, 'accumulate_grad_batches', 1)
                current_training_batch = trainer.global_step * accumulate_grad_batches
                batches_since_last_val = current_training_batch - self._last_validation_batch
                return int(val_interval) - batches_since_last_val
        
        return None