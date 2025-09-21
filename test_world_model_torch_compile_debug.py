#!/usr/bin/env python
"""
Debug script to test if torch.compile recompilation after checkpoint resume causes validation metric divergence.

The hypothesis is that torch.compile creates different optimized code after resume,
potentially causing numerical differences in the validation computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import tempfile
import os
import numpy as np


class SimpleDynamicsModel(nn.Module):
    """Simplified dynamics model to test torch.compile behavior."""

    def __init__(self, latent_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 64)
        self.fc2 = nn.Linear(64, latent_dim)
        self.latent_dim = latent_dim

    def forward(self, z, action):
        # Simple dynamics: next_z = f(z, action)
        h = F.relu(self.fc1(z))
        next_z = self.fc2(h)
        return next_z


def test_torch_compile_reproducibility():
    """Test if torch.compile produces identical results after checkpoint resume."""

    print("=" * 80)
    print("Testing torch.compile reproducibility after checkpoint resume")
    print("=" * 80)

    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Create model and compile it
    model = SimpleDynamicsModel(latent_dim=32)

    # Create test data
    batch_size = 4
    seq_len = 10
    latent_dim = 32

    # Fixed test inputs for reproducibility
    torch.manual_seed(123)
    test_z = torch.randn(batch_size, seq_len, latent_dim)
    test_action = torch.randn(batch_size, seq_len, 8)

    print("\n>>> Phase 1: Initial run with torch.compile")

    # Compile the model
    compiled_model = torch.compile(model, mode='default')

    # Run validation
    with torch.no_grad():
        # Run multiple times to warm up compilation
        for i in range(3):
            output1 = compiled_model(test_z[:, i, :], test_action[:, i, :])

        # Get the actual validation result
        val_output1 = compiled_model(test_z[:, 0, :], test_action[:, 0, :])
        val_loss1 = F.mse_loss(val_output1, test_z[:, 1, :])

    print(f"Initial validation loss: {val_loss1.item():.6f}")
    print(f"Initial output sum: {val_output1.sum().item():.6f}")
    print(f"Initial output mean: {val_output1.mean().item():.6f}")

    # Save checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "checkpoint.pt")

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss1.item(),
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"\n>>> Checkpoint saved to {checkpoint_path}")

        # Phase 2: Load checkpoint and recompile
        print("\n>>> Phase 2: Load checkpoint and recompile")

        # Create new model instance
        model2 = SimpleDynamicsModel(latent_dim=32)

        # Load checkpoint
        checkpoint_loaded = torch.load(checkpoint_path)
        model2.load_state_dict(checkpoint_loaded['model_state_dict'])

        # CRITICAL: Recompile the model (simulating what happens after checkpoint resume)
        compiled_model2 = torch.compile(model2, mode='default')

        # Run validation with same data
        with torch.no_grad():
            # Warm up
            for i in range(3):
                output2 = compiled_model2(test_z[:, i, :], test_action[:, i, :])

            # Get validation result
            val_output2 = compiled_model2(test_z[:, 0, :], test_action[:, 0, :])
            val_loss2 = F.mse_loss(val_output2, test_z[:, 1, :])

        print(f"Post-resume validation loss: {val_loss2.item():.6f}")
        print(f"Post-resume output sum: {val_output2.sum().item():.6f}")
        print(f"Post-resume output mean: {val_output2.mean().item():.6f}")

        # Compare results
        print("\n" + "=" * 40)
        print("COMPARISON")
        print("=" * 40)

        loss_diff = abs(val_loss1.item() - val_loss2.item())
        output_diff = torch.abs(val_output1 - val_output2).max().item()

        print(f"Validation loss difference: {loss_diff:.6e}")
        print(f"Max output difference: {output_diff:.6e}")
        print(f"Loss ratio: {val_loss2.item() / val_loss1.item():.6f}")

        # Check if this matches the reported pattern (0.01 -> 0.07 = 7x increase)
        loss_ratio = val_loss2.item() / val_loss1.item()
        if abs(loss_ratio - 7.0) < 1.0:
            print("\n*** POTENTIAL ISSUE PATTERN DETECTED ***")
            print(f"Loss increased by factor of {loss_ratio:.2f}x after recompilation")
        elif loss_diff < 1e-6:
            print("\n✓ Results are identical - torch.compile is not the issue")
        else:
            print(f"\n⚠ Small difference detected but not matching the 7x pattern")

        # Test without compilation for comparison
        print("\n>>> Phase 3: Test WITHOUT torch.compile for comparison")

        # Reset models
        torch.manual_seed(42)
        model3 = SimpleDynamicsModel(latent_dim=32)
        model4 = SimpleDynamicsModel(latent_dim=32)

        # Run without compilation
        with torch.no_grad():
            val_output3 = model3(test_z[:, 0, :], test_action[:, 0, :])
            val_loss3 = F.mse_loss(val_output3, test_z[:, 1, :])

        # Save and load
        torch.save({'model_state_dict': model3.state_dict()}, checkpoint_path)
        checkpoint_loaded = torch.load(checkpoint_path)
        model4.load_state_dict(checkpoint_loaded['model_state_dict'])

        with torch.no_grad():
            val_output4 = model4(test_z[:, 0, :], test_action[:, 0, :])
            val_loss4 = F.mse_loss(val_output4, test_z[:, 1, :])

        print(f"Without compile - before: {val_loss3.item():.6f}")
        print(f"Without compile - after:  {val_loss4.item():.6f}")
        print(f"Without compile - diff:   {abs(val_loss3.item() - val_loss4.item()):.6e}")


def test_vae_state_preservation():
    """Test if VAE state (eval vs train mode) affects validation metrics."""

    print("\n" + "=" * 80)
    print("Testing VAE eval/train mode impact on validation")
    print("=" * 80)

    class SimpleVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(64, 32),
                nn.BatchNorm1d(32),  # BatchNorm behaves differently in train/eval
                nn.ReLU(),
                nn.Linear(32, 16)
            )

        def encode(self, x):
            return self.encoder(x)

    torch.manual_seed(42)
    vae = SimpleVAE()
    test_input = torch.randn(4, 64)

    # Test in eval mode
    vae.eval()
    with torch.no_grad():
        eval_output = vae.encode(test_input)
        eval_mean = eval_output.mean().item()

    # Test in train mode (incorrect after resume)
    vae.train()
    with torch.no_grad():
        train_output = vae.encode(test_input)
        train_mean = train_output.mean().item()

    print(f"VAE output in eval mode:  mean={eval_mean:.6f}")
    print(f"VAE output in train mode: mean={train_mean:.6f}")
    print(f"Difference: {abs(eval_mean - train_mean):.6f}")
    print(f"Ratio: {train_mean / eval_mean if eval_mean != 0 else 'N/A'}")

    if abs(train_mean / eval_mean - 7.0) < 2.0 and eval_mean != 0:
        print("\n*** POTENTIAL VAE MODE ISSUE DETECTED ***")
        print("VAE being in wrong mode could explain the 7x metric jump!")


if __name__ == "__main__":
    # Test torch.compile reproducibility
    test_torch_compile_reproducibility()

    # Test VAE state preservation
    test_vae_state_preservation()

    print("\n" + "=" * 80)
    print("Debug tests completed")
    print("=" * 80)