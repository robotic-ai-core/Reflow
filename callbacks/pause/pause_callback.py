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
    def __init__(
        self,
        checkpoint_dir: str = "pause_checkpoints",
        enable_pause: bool = True,
        pause_key: str = 'p',
        upload_key: str = 'w',
        debounce_interval: float = 0.3,
        refresh_rate: int = 1,
        bar_colour: str = "#fcac17",
        global_bar_metrics: list = None,
        interval_bar_metrics: list = None,
        logging_interval: str = "step",
        enable_pause_context_management: bool = True,
        skip_dependency_check: bool = False,  # Added for test compatibility
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
        self.enable_pause_context_management = enable_pause_context_management
        
        # State management
        self._state_machine = PauseStateMachine()
        self._keyboard_handler: Optional[ImprovedKeyboardHandler] = None
        self._last_key_time = 0.0
        # Initialize shared WandB artifact manager
        self._wandb_manager = WandbArtifactManager(verbose=True)
        
        # Debug hook flag - when True, pauses are scheduled but not auto-executed
        self._debug_hooks_enabled = False
        
        # Initialize config embedding mixin
        ConfigEmbeddingMixin.__init__(self)
        
        # Store original command line arguments for resume commands
        import sys
        self._original_argv = sys.argv.copy()

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

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule):
        super().on_train_start(trainer, pl_module)
        
        # Check for debug hooks and trigger appropriate states
        self._check_debug_hooks()
        
        if self.enable_pause and trainer.is_global_zero:
            try:
                self._keyboard_handler = ImprovedKeyboardHandler()
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
        # Only execute pause if training has actually progressed (not during sanity validation)
        # and if debug hooks are not enabled (for testing purposes)
        if (self._state_machine.is_pause_scheduled() and 
            trainer.global_step > 0 and 
            not self._debug_hooks_enabled):
            # Check if this is an end-of-epoch validation
            is_end_of_epoch = self._is_end_of_epoch_validation(trainer)
            if is_end_of_epoch:
                self._execute_epoch_boundary_pause(trainer, pl_module)
            else:
                self._execute_validation_boundary_pause(trainer, pl_module)
            
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
        # Execute pause at epoch boundaries when scheduled
        # but not when debug hooks are enabled (for testing)
        if self._state_machine.is_pause_scheduled() and not self._debug_hooks_enabled:
            self._execute_epoch_boundary_pause(trainer, pl_module)
            
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

    def _get_checkpoint_path(self, trainer: Trainer, upload: bool) -> Path:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time())
        tag = "upload" if upload else "pause"
        filename = f"{tag}_epoch={trainer.current_epoch}_step={trainer.global_step}_{timestamp}.ckpt"
        return self.checkpoint_dir / filename
    
    def _save_checkpoint(self, trainer: Trainer, pl_module: LightningModule, checkpoint_path: Path):
        # Save checkpoint normally
        trainer.save_checkpoint(checkpoint_path)
        
        # Add config metadata if the mixin is available
        try:
            # Load the saved checkpoint to add metadata
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # Add config metadata using the mixin
            self.add_config_metadata(trainer, pl_module, checkpoint)
            
            # Save the checkpoint back with metadata
            torch.save(checkpoint, checkpoint_path)
        except Exception as e:
            print(f"Warning: Could not add config metadata to checkpoint: {e}")

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
            if hasattr(callback, '_upload_pause_checkpoint_artifact'):
                wandb_callback = callback
                break
        
        # Graceful degradation: No W&B callback is not an error, just no upload
        if not wandb_callback:
            print(f"⚠️  No W&B callback found - checkpoint saved locally only")
            return None
        
        # Attempt upload with proper error handling
        try:
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
        
        # Extract script name from original argv (usually first element)
        script_name = self._original_argv[0] if self._original_argv else "python train_lightning.py"
        
        print(f"\n🔄 Training paused. Resume options:")
        print(f"📁 Local resume:    {script_name} resume --checkpoint-path {checkpoint_path}")
        
        if artifact_path:
            print(f"☁️  W&B resume:     {script_name} resume --checkpoint-artifact {artifact_path}")
        
        # Also show the legacy method for backward compatibility
        print(f"📁 Legacy method:   {' '.join(self._original_argv)} --ckpt_path {checkpoint_path}")
        if artifact_path:
            # For legacy method, we can use --resume_from_wandb flag
            print(f"☁️  Legacy W&B:     {' '.join(self._original_argv)} --resume_from_wandb {artifact_path}")

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
            print(f"📦 Captured {manager_states.get('manager_count', 0)} manager states")
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