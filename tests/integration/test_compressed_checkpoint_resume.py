"""
End-to-end integration test for compressed checkpoint resume functionality.

This test verifies that the complete resume flow works correctly with compressed
checkpoints, from artifact download through to training resumption.
"""

import pytest
import tempfile
import gzip
import shutil
import pickle
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch

# Add the project to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lightning_reflow.core import LightningReflow
from lightning_reflow.strategies.wandb_artifact_resume_strategy import WandbArtifactResumeStrategy


class TestCompressedCheckpointE2E:
    """End-to-end tests for compressed checkpoint resume."""
    
    @patch('lightning_reflow.strategies.wandb_artifact_resume_strategy.WandbService')
    def test_full_resume_flow_with_compressed_checkpoint(self, mock_wandb_service_class):
        """Test complete resume flow with a compressed checkpoint in W&B artifact."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup mock W&B service
            mock_service = Mock()
            mock_wandb_service_class.return_value = mock_service
            
            # Create a fake downloaded artifact directory with compressed checkpoint
            artifact_dir = Path(tmpdir) / "artifact"
            artifact_dir.mkdir()
            
            # Create a realistic checkpoint structure
            checkpoint_data = {
                'state_dict': {
                    'model.layer1.weight': torch.randn(10, 10),
                    'model.layer1.bias': torch.randn(10)
                },
                'epoch': 15,
                'global_step': 3000,
                'pytorch-lightning_version': '2.0.0',
                'callbacks': {},
                'optimizer_states': [{}],
                'lr_schedulers': [],
                'hparams': {
                    'learning_rate': 0.001,
                    'batch_size': 32
                }
            }
            
            # Use a complex filename that mimics the actual bug scenario
            compressed_checkpoint = artifact_dir / "tmp9lxui_on.ckpt.gz"
            
            # Compress the checkpoint using torch.save format
            import io
            buffer = io.BytesIO()
            torch.save(checkpoint_data, buffer)
            buffer.seek(0)
            
            with gzip.open(compressed_checkpoint, 'wb') as f:
                f.write(buffer.getvalue())
            
            # Mock the download_artifact method
            mock_service.download_artifact.return_value = (
                artifact_dir,
                {
                    'name': 'test-run-123-pause',
                    'version': 'latest',
                    'entity': 'test-entity',
                    'project': 'test-project',
                    'metadata': {
                        'run_id': 'abc123xyz',
                        'created_at': '2024-01-01T00:00:00Z'
                    }
                }
            )
            
            # Create strategy and test resume preparation
            strategy = WandbArtifactResumeStrategy()
            
            # Mock extract_embedded_config
            with patch('lightning_reflow.utils.checkpoint.checkpoint_utils.extract_embedded_config') as mock_extract:
                mock_extract.return_value = None
                
                checkpoint_path, config = strategy.prepare_resume(
                    resume_source="test-entity/test-project/test-run-123-pause:latest"
                )
            
            # Verify the checkpoint was decompressed correctly
            assert checkpoint_path.exists()
            assert checkpoint_path.suffix == '.ckpt'
            assert not checkpoint_path.name.endswith('.gz')
            assert checkpoint_path.name == "tmp9lxui_on.ckpt"
            
            # Verify we can load the decompressed checkpoint with torch.load
            loaded_checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            assert loaded_checkpoint['epoch'] == 15
            assert loaded_checkpoint['global_step'] == 3000
            assert 'model.layer1.weight' in loaded_checkpoint['state_dict']
            
            # Verify the tensor data is intact
            assert loaded_checkpoint['state_dict']['model.layer1.weight'].shape == (10, 10)
            assert loaded_checkpoint['state_dict']['model.layer1.bias'].shape == (10,)
    
    def test_compression_preserves_checkpoint_integrity(self):
        """Test that compression and decompression preserves checkpoint data integrity."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a complex checkpoint with various data types
            original_checkpoint = {
                'epoch': 42,
                'global_step': 8400,
                'state_dict': {
                    'encoder.weight': torch.randn(256, 512),
                    'decoder.weight': torch.randn(512, 256),
                    'embeddings': torch.randn(1000, 128)
                },
                'optimizer_states': [{
                    'state': {
                        0: {
                            'momentum_buffer': torch.randn(256, 512),
                            'exp_avg': torch.randn(256, 512),
                            'exp_avg_sq': torch.randn(256, 512),
                            'step': 8400
                        }
                    }
                }],
                'random_state': {
                    'python': [1, 2, 3],
                    'numpy': [4, 5, 6],
                    'torch': torch.randn(100)
                },
                'metrics': {
                    'train_loss': 0.123,
                    'val_loss': 0.456,
                    'accuracy': 0.789
                }
            }
            
            # Save and compress the checkpoint
            compressed_path = Path(tmpdir) / "checkpoint.ckpt.gz"
            
            import io
            buffer = io.BytesIO()
            torch.save(original_checkpoint, buffer)
            buffer.seek(0)
            
            with gzip.open(compressed_path, 'wb') as f:
                f.write(buffer.getvalue())
            
            # Use the strategy to find and decompress
            strategy = WandbArtifactResumeStrategy()
            decompressed_path = strategy._find_checkpoint_in_artifact(Path(tmpdir))
            
            # Load the decompressed checkpoint
            loaded_checkpoint = torch.load(decompressed_path, map_location='cpu', weights_only=False)
            
            # Verify all data is intact
            assert loaded_checkpoint['epoch'] == original_checkpoint['epoch']
            assert loaded_checkpoint['global_step'] == original_checkpoint['global_step']
            
            # Check state dict tensors
            for key in original_checkpoint['state_dict']:
                assert torch.allclose(
                    loaded_checkpoint['state_dict'][key],
                    original_checkpoint['state_dict'][key]
                )
            
            # Check optimizer states
            assert len(loaded_checkpoint['optimizer_states']) == 1
            orig_state = original_checkpoint['optimizer_states'][0]['state'][0]
            loaded_state = loaded_checkpoint['optimizer_states'][0]['state'][0]
            assert loaded_state['step'] == orig_state['step']
            
            for buffer_name in ['momentum_buffer', 'exp_avg', 'exp_avg_sq']:
                assert torch.allclose(loaded_state[buffer_name], orig_state[buffer_name])
            
            # Check metrics
            assert loaded_checkpoint['metrics'] == original_checkpoint['metrics']
    
    def test_multiple_compressed_checkpoints_selection(self):
        """Test selection logic when multiple compressed checkpoints are present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple compressed checkpoints with different names
            checkpoints = {
                'best.ckpt.gz': {'epoch': 50, 'metric': 0.95},
                'last.ckpt.gz': {'epoch': 60, 'metric': 0.92},
                'checkpoint_epoch_30.ckpt.gz': {'epoch': 30, 'metric': 0.88},
                'model.pt.gz': {'epoch': 40, 'metric': 0.90}
            }
            
            for name, data in checkpoints.items():
                path = Path(tmpdir) / name
                
                import io
                buffer = io.BytesIO()
                torch.save(data, buffer)
                buffer.seek(0)
                
                with gzip.open(path, 'wb') as f:
                    f.write(buffer.getvalue())
            
            strategy = WandbArtifactResumeStrategy()
            selected_checkpoint = strategy._find_checkpoint_in_artifact(Path(tmpdir))
            
            # Should select and decompress best.ckpt.gz due to priority
            assert selected_checkpoint.name == "best.ckpt"
            assert selected_checkpoint.exists()
            
            # Verify correct checkpoint was selected
            loaded = torch.load(selected_checkpoint, map_location='cpu', weights_only=False)
            assert loaded['epoch'] == 50
            assert loaded['metric'] == 0.95
    
    def test_mixed_compressed_uncompressed_priority(self):
        """Test that uncompressed checkpoints are preferred when both exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create both compressed and uncompressed versions
            
            # Uncompressed checkpoint
            uncompressed_path = Path(tmpdir) / "checkpoint.ckpt"
            torch.save({'epoch': 100, 'type': 'uncompressed'}, uncompressed_path)
            
            # Compressed checkpoint with same base name
            compressed_path = Path(tmpdir) / "other.ckpt.gz"
            import io
            buffer = io.BytesIO()
            torch.save({'epoch': 99, 'type': 'compressed'}, buffer)
            buffer.seek(0)
            
            with gzip.open(compressed_path, 'wb') as f:
                f.write(buffer.getvalue())
            
            strategy = WandbArtifactResumeStrategy()
            selected_checkpoint = strategy._find_checkpoint_in_artifact(Path(tmpdir))
            
            # Should prefer the uncompressed checkpoint
            assert selected_checkpoint == uncompressed_path
            loaded = torch.load(selected_checkpoint, map_location='cpu', weights_only=False)
            assert loaded['type'] == 'uncompressed'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])