"""Interactive pause callback for PyTorch Lightning training.

Pauses training cleanly at the next validation boundary in response to a keyboard
press or programmatic `request_pause()` call. Saves an atomic checkpoint, optionally
uploads it to W&B, prints resume commands, and sets `trainer.should_stop`.

Internal helpers (checkpoint save, upload, resume-command formatting) used to live
in separate files; they were folded back here as private methods so the full pause
flow can be read in one place.
"""

import logging
import shutil
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from lightning.pytorch import LightningModule, Trainer

from ..core.config_embedding_mixin import ConfigEmbeddingMixin
from ..monitoring.flow_progress_bar_callback import FlowProgressBarCallback
from ...utils.wandb.wandb_artifact_manager import WandbArtifactManager
from .keyboard_handler import KeyboardHandler
from .pause_state_machine import PauseState, PauseStateMachine

logger = logging.getLogger(__name__)


class PauseCallback(FlowProgressBarCallback, ConfigEmbeddingMixin):
    """Interactive pause callback with validation-boundary checkpointing.

    Features:
    - Interactive pause via keyboard (default: 'p' key)
    - Optional W&B upload during pause (default: 'w' key)
    - Validation boundary pausing for robustness
    - Atomic checkpoint save with validation
    - Configurable progress-bar pause countdown

    Args:
        checkpoint_dir: Directory for pause checkpoints.
        enable_pause: Enable interactive (keyboard) pause functionality.
        pause_key: Key to trigger pause (default: 'p').
        upload_key: Key to toggle W&B upload (default: 'w').
        show_pause_countdown: Show "Pause in: X steps" on progress bar.
        ... (other FlowProgressBarCallback args)
    """

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
        skip_dependency_check: bool = False,  # Kept for test compatibility
        show_pause_countdown: bool = False,
        save_rng_states: bool = True,
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
        self.show_pause_countdown = show_pause_countdown

        # State
        self._state_machine = PauseStateMachine()
        self._keyboard_handler: Optional[KeyboardHandler] = None
        self._last_key_time = 0.0
        self._wandb_manager = WandbArtifactManager(verbose=True)
        self.last_checkpoint_path: Optional[Path] = None
        self._original_argv: List[str] = sys.argv.copy()

        # When True, pauses are scheduled but not auto-executed (testing hook).
        self._debug_hooks_enabled = False

        # Initialize config embedding mixin (also stores sys.argv for resume commands)
        ConfigEmbeddingMixin.__init__(self)

        # Register scientific reproducibility manager if enabled
        self.save_rng_states = save_rng_states
        if self.save_rng_states:
            from ...utils.checkpoint.manager_state import register_manager
            from ...utils.checkpoint.scientific_reproducibility_state import (
                ScientificReproducibilityState,
            )
            self._reproducibility_manager = ScientificReproducibilityState()
            register_manager(self._reproducibility_manager)
            logger.info("Scientific reproducibility enabled (RNG states will be saved)")

    # ------------------------------------------------------------------
    # State predicates
    # ------------------------------------------------------------------

    @property
    def _pause_state(self) -> PauseState:
        return self._state_machine.state

    def is_pause_pending(self) -> bool:
        """True if a pause has been scheduled and not yet executed."""
        return self._state_machine.is_pause_scheduled()

    # Aliases kept for callers that already use these names.
    is_pause_scheduled = is_pause_pending
    is_pausing = is_pause_pending

    def is_upload_requested(self) -> bool:
        return self._state_machine.is_upload_requested()

    def is_upload_all_requested(self) -> bool:
        return self._state_machine.is_upload_all_requested()

    def get_last_checkpoint(self) -> Optional[Path]:
        return self.last_checkpoint_path

    # ------------------------------------------------------------------
    # Public API for external callbacks (e.g. EarlyPauseCallback)
    # ------------------------------------------------------------------

    def request_pause(self, upload: bool = False, reason: str | None = None) -> bool:
        """Schedule a pause from an external source.

        The pause executes at the next validation boundary. Returns True if a
        new pause was scheduled, False if one was already pending.
        """
        if self._state_machine.is_pause_scheduled():
            if upload and not self._state_machine.is_upload_requested():
                self._state_machine.toggle_upload()
            if reason:
                logger.info("Pause already scheduled: %s", reason)
            return False

        self._state_machine.toggle_pause()
        if upload:
            self._state_machine.toggle_upload()

        if reason:
            logger.info("Pause requested: %s", reason)
        else:
            logger.info("Pause requested by external callback")
        return True

    def cancel_pause(self) -> bool:
        """Cancel a scheduled pause. Returns True if one was cancelled."""
        if not self._state_machine.is_pause_scheduled():
            return False
        self._state_machine.reset()
        logger.info("Pause cancelled")
        return True

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule):
        super().on_train_start(trainer, pl_module)
        self._check_debug_hooks()

        if self.enable_pause and trainer.is_global_zero:
            try:
                self._keyboard_handler = KeyboardHandler()
                self._keyboard_handler.start_monitoring()
            except Exception as e:
                warnings.warn(
                    f"Could not initialize keyboard handler: {e}. Disabling pause functionality."
                )
                self.enable_pause = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

        if self.enable_pause:
            self._check_keyboard_input()

        # Immediate pause only when val_check_interval isn't set; otherwise we
        # wait for the validation boundary.
        if (
            self._state_machine.is_pause_scheduled()
            and getattr(trainer, 'val_check_interval', None) is None
            and not self._debug_hooks_enabled
        ):
            self._execute_pause(trainer, pl_module, atomic=False)

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule):
        super().on_validation_end(trainer, pl_module)
        try:
            self._execute_validation_boundary_pause_if_needed(trainer, pl_module)
        except Exception as e:
            logger.critical(
                "Validation boundary pause failed: %s. Training continues without pause.", e
            )
            self._state_machine.reset()

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule):
        if self._keyboard_handler:
            self._keyboard_handler.stop_monitoring()

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_fit_start(trainer, pl_module)
        self.capture_and_cache_config(trainer)

        if self.save_rng_states and hasattr(self, '_reproducibility_manager'):
            self._reproducibility_manager.set_references(model=pl_module, trainer=trainer)

    def on_load_checkpoint(self, trainer, pl_module, checkpoint: Dict[str, Any]) -> None:
        super().on_load_checkpoint(trainer, pl_module, checkpoint)

        # Update stored argv so resume command output reflects the current invocation.
        self._original_argv = sys.argv.copy()

        # Restore manager states (RNG, DataModule, TrainerConfig, Environment, etc.)
        metadata = checkpoint.get('pause_callback_metadata', {})
        manager_states = metadata.get('manager_states', {})
        if manager_states:
            from ...utils.checkpoint.manager_state import restore_all_manager_states
            results = restore_all_manager_states(manager_states)
            restored = sum(1 for v in results.values() if v)
            failed = sum(1 for v in results.values() if not v)
            if failed:
                logger.warning(
                    "Manager state restore: %d restored, %d failed (%s)",
                    restored, failed, results,
                )
            else:
                logger.info("Restored %d manager states: %s", restored, list(results.keys()))

        if self.save_rng_states and hasattr(self, '_reproducibility_manager'):
            self._reproducibility_manager.set_references(model=pl_module, trainer=trainer)
            self._reproducibility_manager.post_restoration_hook()

    def on_save_checkpoint(self, trainer, pl_module, checkpoint: Dict[str, Any]) -> None:
        super().on_save_checkpoint(trainer, pl_module, checkpoint)
        self.add_config_metadata(trainer, pl_module, checkpoint, metadata_key='pause_callback_metadata')

        try:
            from ...utils.checkpoint.manager_state import capture_all_manager_states
            manager_states = capture_all_manager_states()
            checkpoint.setdefault('pause_callback_metadata', {})['manager_states'] = manager_states
            logger.debug("Captured %d manager states", len(manager_states))
        except Exception as e:
            logger.warning("Failed to capture manager states: %s", e)

        pause_metadata = {
            'pause_timestamp': time.time(),
            'pause_state': self._state_machine.state.value,
            'checkpoint_dir': str(self.checkpoint_dir),
            'enable_pause': self.enable_pause,
            'pause_point': 'manual',
            'upload_requested': self._state_machine.is_upload_requested(),
        }
        checkpoint.setdefault('pause_callback_metadata', {}).update(pause_metadata)

    # ------------------------------------------------------------------
    # Pause execution
    # ------------------------------------------------------------------

    def _execute_validation_boundary_pause_if_needed(self, trainer, pl_module):
        """Run safety checks, then execute a validation-boundary pause."""
        if not self._state_machine.is_pause_scheduled():
            return
        if self._debug_hooks_enabled:
            return
        if trainer.global_step <= 0:
            logger.info(
                "Pause scheduled but skipping during sanity validation (global_step=%d)",
                trainer.global_step,
            )
            return
        if not self._validate_trainer_state_for_pause(trainer, pl_module):
            logger.error("Trainer state invalid for pause — skipping pause at validation boundary")
            self._state_machine.reset()
            return
        if getattr(trainer, 'interrupted', False):
            logger.info("Training already interrupted — skipping pause execution")
            return
        if getattr(trainer, 'should_stop', False):
            logger.info("Training should_stop=True — skipping pause execution")
            return

        logger.info(
            "Executing pause at validation boundary (global_step=%d, epoch=%d)",
            trainer.global_step, trainer.current_epoch,
        )
        self._execute_pause(trainer, pl_module, atomic=True)

    def _execute_pause(self, trainer: Trainer, pl_module: LightningModule, atomic: bool):
        """Common pause-execution flow.

        Args:
            atomic: When True, save to a temp file and atomically rename;
                also embed config metadata. When False, do a plain save (used
                for immediate-pause from on_train_batch_end).
        """
        should_upload = self._state_machine.is_upload_requested()
        checkpoint_path = self._get_checkpoint_path(trainer, upload=should_upload)
        artifact_path: Optional[str] = None

        try:
            if atomic:
                logger.info("Creating pause checkpoint at: %s", checkpoint_path)
                self._save_checkpoint_with_validation(trainer, pl_module, checkpoint_path)
                logger.info("Pause checkpoint saved successfully")
            else:
                self._save_checkpoint(trainer, pl_module, checkpoint_path)

            if should_upload:
                if atomic:
                    artifact_path = self._handle_upload_with_fallback(
                        trainer, pl_module, str(checkpoint_path),
                    )
                else:
                    try:
                        artifact_path = self._handle_wandb_upload(
                            trainer, pl_module, str(checkpoint_path),
                        )
                    except (ValueError, RuntimeError) as e:
                        logger.warning("Upload failed but pause will continue: %s", e)

            trainer.should_stop = True
            self._state_machine.reset()

            if atomic:
                logger.info("Training paused successfully at validation boundary")
                self._print_resume_commands_with_fallback(
                    trainer, str(checkpoint_path), artifact_path,
                )
            else:
                try:
                    self._print_resume_commands(trainer, str(checkpoint_path), artifact_path)
                except ValueError as e:
                    logger.warning("Could not generate resume commands: %s", e)
                    logger.info("Checkpoint saved at: %s", checkpoint_path)

        except Exception:
            if atomic and checkpoint_path and checkpoint_path.exists():
                try:
                    checkpoint_path.unlink()
                    logger.info("Cleaned up partial checkpoint: %s", checkpoint_path)
                except Exception as cleanup_error:
                    logger.warning("Could not clean up partial checkpoint: %s", cleanup_error)
            self._state_machine.reset()
            raise

    # ------------------------------------------------------------------
    # Checkpoint helpers (formerly PauseCheckpointManager)
    # ------------------------------------------------------------------

    def _get_checkpoint_path(self, trainer: Trainer, upload: bool = False) -> Path:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time())
        tag = "upload" if upload else "pause"
        filename = (
            f"{tag}_epoch={trainer.current_epoch}_step={trainer.global_step}_{timestamp}.ckpt"
        )
        return self.checkpoint_dir / filename

    def _save_checkpoint(self, trainer: Trainer, pl_module: LightningModule, checkpoint_path: Path):
        """Plain save via trainer.save_checkpoint."""
        trainer.save_checkpoint(checkpoint_path)
        self.last_checkpoint_path = checkpoint_path

    def _save_checkpoint_with_validation(self, trainer, pl_module, checkpoint_path: Path):
        """Atomic save with size + structure validation and embedded config metadata."""
        temp_path = checkpoint_path.with_suffix('.tmp')
        try:
            trainer.save_checkpoint(temp_path)

            if not temp_path.exists():
                raise RuntimeError(f"Checkpoint was not created at {temp_path}")

            size = temp_path.stat().st_size
            if size < 1024:
                raise RuntimeError(f"Checkpoint file too small ({size} bytes) — likely corrupted")

            try:
                checkpoint = torch.load(temp_path, map_location='cpu', weights_only=False)
                missing = [k for k in ('state_dict', 'epoch', 'global_step') if k not in checkpoint]
                if missing:
                    raise RuntimeError(f"Checkpoint missing required keys: {missing}")
            except Exception as e:
                raise RuntimeError(f"Checkpoint validation failed: {e}")

            try:
                self.add_config_metadata(trainer, pl_module, checkpoint)
                torch.save(checkpoint, temp_path)
                logger.debug("Added config metadata to pause checkpoint")
            except Exception as e:
                logger.warning("Could not add config metadata to checkpoint: %s", e)

            temp_path.rename(checkpoint_path)
            logger.info(
                "Checkpoint atomically saved to %s (%s bytes)", checkpoint_path, f"{size:,}"
            )
            self.last_checkpoint_path = checkpoint_path

        except Exception as e:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            raise RuntimeError(f"Failed to save pause checkpoint: {e}")

    def _validate_trainer_state_for_pause(self, trainer, pl_module) -> bool:
        """Return False if state is unsafe for checkpointing."""
        for attr in ('global_step', 'current_epoch', 'logger'):
            if not hasattr(trainer, attr):
                logger.error("Trainer missing required attribute for pause: %s", attr)
                return False
        if pl_module is None:
            logger.error("LightningModule is None — cannot create pause checkpoint")
            return False
        try:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error("Cannot access checkpoint directory %s: %s", self.checkpoint_dir, e)
            return False
        try:
            free = shutil.disk_usage(self.checkpoint_dir).free
            if free < 100 * 1024 * 1024:
                logger.error(
                    "Low disk space for pause checkpoint: %.1f MB", free / (1024 * 1024)
                )
                return False
        except Exception as e:
            logger.warning("Could not check disk space: %s", e)
        return True

    # ------------------------------------------------------------------
    # W&B upload helpers (formerly PauseUploadHandler)
    # ------------------------------------------------------------------

    def _handle_wandb_upload(
        self, trainer: Trainer, pl_module: LightningModule, checkpoint_path: str,
    ) -> Optional[str]:
        """Upload pause checkpoint via the in-trainer WandbArtifactCheckpoint callback."""
        if trainer is None:
            raise ValueError("Trainer cannot be None for W&B upload")
        if not hasattr(trainer, 'callbacks') or trainer.callbacks is None:
            raise ValueError("Trainer must have callbacks list for W&B upload")

        wandb_callback = None
        for cb in trainer.callbacks:
            if hasattr(cb, 'upload_pause_checkpoint'):
                wandb_callback = cb
                break
            if hasattr(cb, '_upload_pause_checkpoint_artifact'):
                wandb_callback = cb
                break

        if wandb_callback is None:
            logger.info("No W&B callback found — checkpoint saved locally only")
            return None

        try:
            if hasattr(wandb_callback, 'upload_pause_checkpoint'):
                artifact_path = wandb_callback.upload_pause_checkpoint(
                    trainer, trainer.lightning_module, checkpoint_path,
                )
            else:
                artifact_path = self._upload_pause_checkpoint_artifact(
                    wandb_callback, trainer, checkpoint_path,
                )
            logger.info("Pause checkpoint uploaded to W&B successfully")
            return artifact_path
        except (ValueError, RuntimeError) as e:
            logger.warning("Failed to upload pause checkpoint to W&B: %s", e)
            return None
        except Exception as e:
            raise RuntimeError(f"Unexpected error during W&B upload: {e}") from e

    def _upload_pause_checkpoint_artifact(
        self, wandb_callback: Any, trainer: Trainer, checkpoint_path: str,
    ) -> Optional[str]:
        """Upload pause checkpoint via the shared WandbArtifactManager."""
        if trainer is None:
            raise ValueError("Trainer cannot be None for artifact upload")
        if not checkpoint_path or not Path(checkpoint_path).exists():
            raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
        if self._wandb_manager is None:
            raise RuntimeError("W&B artifact manager is not initialized")

        wandb_run = self._wandb_manager.get_wandb_run(trainer)
        if not wandb_run:
            raise RuntimeError("No active W&B run found — cannot upload artifacts")

        if not hasattr(trainer, 'lightning_module') or trainer.lightning_module is None:
            raise ValueError("Trainer must have a valid lightning_module for upload")

        try:
            artifact_path = self._wandb_manager.upload_checkpoint_artifact(
                trainer=trainer,
                pl_module=trainer.lightning_module,
                filepath=checkpoint_path,
                ckpt_type="pause",
                aliases=["pause", "latest"],
                score=None,
                epoch=trainer.current_epoch,
                step=trainer.global_step,
                wandb_run=wandb_run,
                extra_metadata={
                    "pause_type": "manual_pause",
                    "checkpoint_type": "pause_checkpoint",
                    "pause_callback_version": "2.1",
                },
            )
            if not artifact_path:
                raise RuntimeError("Artifact upload returned None — upload failed")
            return artifact_path
        except (AttributeError, KeyError) as e:
            raise RuntimeError(f"Missing required attribute for artifact upload: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Artifact upload failed: {e}") from e

    def _handle_upload_with_fallback(
        self, trainer, pl_module, checkpoint_path: str,
    ) -> Optional[str]:
        try:
            artifact_path = self._handle_wandb_upload(trainer, pl_module, checkpoint_path)
            if artifact_path:
                logger.info("Pause checkpoint uploaded to W&B: %s", artifact_path)
                return artifact_path
            logger.info("W&B upload returned None — checkpoint saved locally only")
            return None
        except (ValueError, RuntimeError) as e:
            logger.warning("W&B upload failed but pause will continue: %s", e)
            logger.info("Checkpoint available locally at: %s", checkpoint_path)
            return None
        except Exception as e:
            logger.error("Unexpected error during W&B upload: %s", e)
            logger.info("Checkpoint available locally at: %s", checkpoint_path)
            return None

    # ------------------------------------------------------------------
    # Resume-command output (formerly ResumeCommandPrinter)
    # ------------------------------------------------------------------

    def _print_resume_commands(
        self, trainer: Trainer, checkpoint_path: str, artifact_path: Optional[str] = None,
    ) -> None:
        """Print resume commands. Uses raw print() so the message reaches the
        operator's terminal regardless of log level."""
        if not checkpoint_path:
            raise ValueError("Checkpoint path cannot be empty")
        if not self._original_argv:
            raise ValueError("Original argv not stored — cannot generate resume commands")

        script_command = self._detect_script_command()
        print("\nTraining paused. Resume options:")
        print(f"Local resume:    {script_command} resume --checkpoint-path {checkpoint_path}")
        if artifact_path:
            print(f"W&B resume:      {script_command} resume --checkpoint-artifact {artifact_path}")

        legacy_command = self._build_legacy_command()
        print(f"Legacy method:   {legacy_command} --ckpt_path {checkpoint_path}")
        if artifact_path:
            print(f"Legacy W&B:      {legacy_command} --resume_from_wandb {artifact_path}")

    def _print_resume_commands_with_fallback(
        self, trainer: Trainer, checkpoint_path: str, artifact_path: Optional[str] = None,
    ) -> None:
        try:
            self._print_resume_commands(trainer, checkpoint_path, artifact_path)
        except ValueError as e:
            print(f"Could not generate resume commands: {e}")
            print(f"Checkpoint saved at: {checkpoint_path}")
            if artifact_path:
                print(f"W&B artifact: {artifact_path}")
            print(f"Use standard Lightning resume: --ckpt_path {checkpoint_path}")
        except Exception as e:
            print(f"Unexpected error generating resume commands: {e}")
            print(f"Checkpoint saved at: {checkpoint_path}")
            print(f"Manually resume with: --ckpt_path {checkpoint_path}")

    def _detect_script_command(self) -> str:
        script_name = self._original_argv[0] if self._original_argv else "train_lightning.py"

        if (
            "__main__.py" in script_name
            or script_name.endswith("/lightning_reflow/cli/__main__.py")
            or ("lightning_reflow" in script_name and "__main__" in script_name)
        ):
            if sys.argv and sys.argv[0].endswith('.py'):
                return f"python {sys.argv[0]}"
            return "python train_lightning.py"
        if not script_name.startswith("python"):
            return f"python {script_name}"
        return script_name

    def _build_legacy_command(self) -> str:
        filtered: List[str] = []
        i = 0
        while i < len(self._original_argv):
            arg = self._original_argv[i]
            if arg == '--ckpt_path':
                i += 2
            elif arg.startswith('--ckpt_path='):
                i += 1
            else:
                filtered.append(arg)
                i += 1
        if not filtered:
            return "python train_lightning.py"
        if not filtered[0].startswith("python"):
            return f"python {' '.join(filtered)}"
        return ' '.join(filtered)

    # ------------------------------------------------------------------
    # Keyboard handling
    # ------------------------------------------------------------------

    def _handle_pause_key(self):
        if self._debounce():
            return
        if self._state_machine.toggle_pause():
            if self._state_machine.is_pause_scheduled():
                print("\n🔄 PAUSE scheduled - will pause at next validation boundary")
            else:
                print("\n❌ PAUSE cancelled - training will continue")

    def _handle_upload_key(self):
        if self._debounce():
            return
        if self._state_machine.toggle_upload():
            if self._state_machine.is_upload_all_requested():
                print("\n📤📤 Upload ALL ENABLED - all checkpoints will be uploaded to W&B")
            elif self._state_machine.is_upload_requested():
                print("\n📤 Upload ENABLED - pause checkpoint will be uploaded to W&B")
            else:
                print("\n📤 Upload DISABLED - checkpoint will not be uploaded to W&B")

    def handle_pause_key(self):
        """Backward-compatible alias for tests."""
        return self._handle_pause_key()

    def _check_keyboard_input(self):
        if self._keyboard_handler:
            key = self._keyboard_handler.get_key()
            if key == self.pause_key:
                self._handle_pause_key()
            elif key == self.upload_key:
                self._handle_upload_key()

    def _debounce(self) -> bool:
        now = time.time()
        if now - self._last_key_time < self.debounce_interval:
            return True
        self._last_key_time = now
        return False

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def _check_debug_hooks(self):
        """DEBUG_PAUSE_HOOK / DEBUG_UPLOAD_HOOK env hooks for testing."""
        import os
        debug_pause = os.getenv("DEBUG_PAUSE_HOOK", "false").lower() == "true"
        debug_upload = os.getenv("DEBUG_UPLOAD_HOOK", "false").lower() == "true"
        if debug_pause:
            self._debug_hooks_enabled = True
            self._state_machine.toggle_pause()
            if debug_upload:
                self._state_machine.toggle_upload()

    def _get_interval_pause_status_suffix(self) -> str:
        if self._pause_state == PauseState.RUNNING:
            return f" - Press '{self.pause_key}' to pause"
        if self._pause_state == PauseState.PAUSE_SCHEDULED_NO_UPLOAD:
            return f" - {self.pause_key}: to unpause, {self.upload_key}: upload pause ckpt"
        if self._pause_state == PauseState.PAUSE_SCHEDULED_WITH_UPLOAD:
            return f" - {self.pause_key}: to unpause, {self.upload_key}: upload all (✓)"
        if self._pause_state == PauseState.PAUSE_SCHEDULED_UPLOAD_ALL:
            return f" - {self.pause_key}: to unpause, {self.upload_key}: cancel upload (✓✓)"
        return f" - Press '{self.pause_key}' to pause"

    def _get_global_pause_status_suffix(self) -> str:
        return ""

    def _update_interval_bar_postfix(self) -> None:
        if self.current_interval_bar is None or not self._interval_metrics:
            return
        base_postfix = self._format_metrics_postfix(self._interval_metrics)

        if (
            self._state_machine.is_pause_scheduled()
            and self._trainer
            and self.show_pause_countdown
        ):
            if hasattr(super(), '_get_steps_until_next_validation'):
                steps_until_val = super()._get_steps_until_next_validation(
                    self._trainer, self._current_batch_idx or 0,
                )
            else:
                steps_until_val = self._calculate_steps_until_validation(
                    self._trainer, self._current_batch_idx or 0,
                )
            if steps_until_val is not None:
                base_postfix += f" | ⏸️ Pause in: {steps_until_val} steps"

        self.current_interval_bar.set_postfix_str(base_postfix)

    def _calculate_steps_until_validation(self, trainer: Trainer, batch_idx: int) -> Optional[int]:
        check_val_every_n_epoch = getattr(trainer, 'check_val_every_n_epoch', None)
        if check_val_every_n_epoch:
            current_epoch = trainer.current_epoch
            next_val_epoch = ((current_epoch // check_val_every_n_epoch) + 1) * check_val_every_n_epoch
            epochs_until_val = next_val_epoch - current_epoch
            num_batches = getattr(trainer, 'num_training_batches', None)
            if num_batches and num_batches != float('inf'):
                steps_left_in_epoch = num_batches - (batch_idx + 1)
                steps_in_full_epochs = (epochs_until_val - 1) * num_batches
                return steps_left_in_epoch + steps_in_full_epochs
        elif trainer.val_check_interval:
            val_interval = trainer.val_check_interval
            accumulate_grad_batches = getattr(trainer, 'accumulate_grad_batches', 1)
            current_training_batch = trainer.global_step * accumulate_grad_batches
            batches_since_last_val = current_training_batch - self._last_validation_batch
            if isinstance(val_interval, int):
                return val_interval - batches_since_last_val
            if isinstance(val_interval, float) and val_interval > 1.0:
                return int(val_interval) - batches_since_last_val
        return None
