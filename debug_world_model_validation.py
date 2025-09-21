#!/usr/bin/env python
"""
Add debug logging to the WorldModel validation_step to understand metric divergence.
This script will patch the existing world model to add detailed logging.
"""

import sys
import os

# Add parent directory to path to import world model
sys.path.insert(0, '/home/neil/code/modelling/ProtoWorld')

import torch
import torch.nn.functional as F


def create_debug_validation_step():
    """Create a patched validation_step with extensive debug logging."""

    def validation_step_with_debug(self, batch, batch_idx):
        """
        Validation step with debug logging for metric divergence investigation.
        """
        # Debug: Check VAE state
        print(f"\n{'='*60}")
        print(f"DEBUG validation_step - batch_idx={batch_idx}")
        print(f"VAE training mode: {self.vae.training}")
        print(f"VAE encoder training mode: {self.vae.encoder.training}")
        print(f"VAE decoder training mode: {self.vae.decoder.training}")
        print(f"Model training mode: {self.training}")

        # Check if we were restored from checkpoint
        if hasattr(self, '_restored_from_checkpoint'):
            print(f"_restored_from_checkpoint flag: {self._restored_from_checkpoint}")
        if hasattr(self.dynamics_model, '_restored_from_checkpoint'):
            print(f"dynamics_model._restored_from_checkpoint: {self.dynamics_model._restored_from_checkpoint}")

        # Notify CUDA graphs that a new step is beginning.
        torch.compiler.cudagraph_mark_step_begin()

        # Extract data from batch - handle both old and new formats
        if "observation.image" in batch:
            obs_images = batch["observation.image"]
        else:
            obs_images = batch["observation"]["image"]

        actions = batch["action"]

        # Get the actual batch size from the tensor
        batch_size = actions.shape[0]
        print(f"Batch size: {batch_size}")

        # 1. Encode all frames to get the VAE latent sequence `z_sequence`
        _, seq_len, C, H, W = obs_images.shape
        images_flat = obs_images.reshape(-1, C, H, W)

        # LeRobotDataset gives CHW float32 [0,1], VAE needs [-1,1]
        images_for_vae = (images_flat * 2.0 - 1.0).float()

        print(f"Input images shape: {images_flat.shape}")
        print(f"Images for VAE - mean: {images_for_vae.mean():.6f}, std: {images_for_vae.std():.6f}")

        with torch.no_grad():
            z_map_sequence = self.vae.encoder.encode(images_for_vae)
            flat_z_raw = torch.flatten(z_map_sequence, start_dim=1)

        z_sequence = flat_z_raw.reshape(batch_size, seq_len, -1)
        print(f"Z sequence shape: {z_sequence.shape}")
        print(f"Z sequence - mean: {z_sequence.mean():.6f}, std: {z_sequence.std():.6f}")

        # 2. Prepare sequences for one-step prediction
        input_z_sequence = z_sequence[:, :-1, :]
        input_action_sequence = actions[:, :-1, :]

        # 3. Predict next step (one-step)
        torch.compiler.cudagraph_mark_step_begin()
        predictions = self.dynamics_model(input_z_sequence, input_action_sequence)
        pred_next_z = predictions["next_z"]

        print(f"Predicted next_z shape: {pred_next_z.shape}")
        print(f"Predicted next_z - mean: {pred_next_z.mean():.6f}, std: {pred_next_z.std():.6f}")

        # 4. Compute one-step loss
        gt_next_z = z_sequence[:, 1:, :]
        pred_len = pred_next_z.shape[1]
        gt_len = gt_next_z.shape[1]
        min_len = min(pred_len, gt_len)

        one_step_loss = F.mse_loss(pred_next_z[:, :min_len, :], gt_next_z[:, :min_len, :])
        print(f"val_one_step_loss: {one_step_loss:.6f}")

        # Check for the specific jump pattern (0.01 -> 0.07)
        if one_step_loss > 0.05 and one_step_loss < 0.1:
            print("*** WARNING: val_one_step_loss in problematic range (0.05-0.1) ***")

        # --- Multi-Step Prediction Loss for Validation ---
        initial_latent = z_sequence[:, 0, :]
        action_sequence = actions[:, :-1, :]
        target_sequence = z_sequence[:, 1:, :]

        prediction_loss = self._rollout_and_compute_loss(initial_latent, action_sequence, target_sequence)
        print(f"val_prediction_loss: {prediction_loss:.6f}")

        # Store validation metrics for logging outside the compiled context
        val_metrics = {
            "val_one_step_loss": one_step_loss,
            "val_prediction_loss": prediction_loss,
            "val_loss": prediction_loss,  # For ModelCheckpoint compatibility
        }
        self._validation_metrics = val_metrics
        self._validation_batch_size = batch_size

        print(f"{'='*60}\n")

        # Return dict for compatibility with StepOutputLoggerCallback
        return {"val_loss": prediction_loss, **val_metrics}

    return validation_step_with_debug


def test_world_model_validation():
    """Test the world model validation with debug logging."""

    from world_model.models.world_model import WorldModel
    import lightning.pytorch as pl
    from torch.utils.data import DataLoader, TensorDataset

    print("Creating WorldModel with debug validation_step...")

    # Create model with minimal config
    model = WorldModel(
        transformer_hparams={
            'latent_dim': 32,
            'action_dim': 8,
            'context_length': 4,
            'num_heads': 4,
            'num_layers': 2,
        },
        adapter_hparams={
            'vae_latent_dim': 256,
            'internal_latent_dim': 32,
        },
        prediction_horizon=2,
        learning_rate=1e-4,
        torch_compile_settings={'enabled': False},  # Disable for debugging
    )

    # Patch the validation_step with our debug version
    model.validation_step = create_debug_validation_step().__get__(model, WorldModel)

    # Create dummy data
    batch_size = 2
    seq_len = 5
    img_size = 64

    dummy_batch = {
        'observation': {
            'image': torch.randn(batch_size, seq_len, 3, img_size, img_size).clamp(0, 1)
        },
        'action': torch.randn(batch_size, seq_len, 8)
    }

    # Run validation step
    print("\n" + "="*80)
    print("Running validation_step with debug logging...")
    print("="*80)

    with torch.no_grad():
        result = model.validation_step(dummy_batch, 0)

    print(f"\nValidation result keys: {result.keys()}")
    print(f"val_one_step_loss: {result['val_one_step_loss']:.6f}")
    print(f"val_prediction_loss: {result['val_prediction_loss']:.6f}")

    # Simulate checkpoint save/load
    print("\n" + "="*80)
    print("Simulating checkpoint save/load...")
    print("="*80)

    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.ckpt') as tmp:
        # Save checkpoint
        checkpoint = {
            'state_dict': model.state_dict(),
            'transformer_buffer_state': {}
        }
        torch.save(checkpoint, tmp.name)

        # Simulate load
        model.on_load_checkpoint(checkpoint)

        # Run validation again after "resume"
        print("\nRunning validation after simulated resume...")
        with torch.no_grad():
            result2 = model.validation_step(dummy_batch, 1)

        print(f"\nPost-resume validation result:")
        print(f"val_one_step_loss: {result2['val_one_step_loss']:.6f}")
        print(f"val_prediction_loss: {result2['val_prediction_loss']:.6f}")

        # Compare
        loss_diff = abs(result['val_one_step_loss'] - result2['val_one_step_loss'])
        loss_ratio = result2['val_one_step_loss'] / result['val_one_step_loss'] if result['val_one_step_loss'] > 0 else 0

        print(f"\nLoss difference: {loss_diff:.6f}")
        print(f"Loss ratio: {loss_ratio:.2f}x")

        if loss_ratio > 5:
            print("*** ISSUE REPRODUCED: Large metric divergence after resume! ***")


if __name__ == "__main__":
    test_world_model_validation()