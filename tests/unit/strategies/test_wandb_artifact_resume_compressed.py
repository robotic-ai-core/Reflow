"""
Test suite for W&B artifact resume strategy with compressed checkpoint support.

This module tests the functionality for handling compressed checkpoint files (.gz)
in W&B artifacts, ensuring backward compatibility while adding new capabilities.
"""

import pytest
import tempfile
import gzip
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch

# Add the project to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from lightning_reflow.strategies.wandb_artifact_resume_strategy import WandbArtifactResumeStrategy


class TestCompressedCheckpointHandling:
    """Test handling of compressed checkpoint files in W&B artifacts."""
    
    def test_find_regular_checkpoint(self):
        """Test finding a regular uncompressed .ckpt file (backward compatibility)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a regular checkpoint file
            checkpoint_path = Path(tmpdir) / "checkpoint.ckpt"
            checkpoint_path.write_bytes(b"fake checkpoint content")
            
            strategy = WandbArtifactResumeStrategy()
            found_checkpoint = strategy._find_checkpoint_in_artifact(Path(tmpdir))
            
            assert found_checkpoint == checkpoint_path
            assert found_checkpoint.exists()
            assert found_checkpoint.suffix == '.ckpt'
    
    def test_find_compressed_checkpoint(self):
        """Test finding and decompressing a .ckpt.gz file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a compressed checkpoint file
            original_content = b"original checkpoint data"
            compressed_path = Path(tmpdir) / "checkpoint.ckpt.gz"
            
            with gzip.open(compressed_path, 'wb') as f:
                f.write(original_content)
            
            strategy = WandbArtifactResumeStrategy()
            found_checkpoint = strategy._find_checkpoint_in_artifact(Path(tmpdir))
            
            # Should return the decompressed path
            expected_path = Path(tmpdir) / "checkpoint.ckpt"
            assert found_checkpoint == expected_path
            assert found_checkpoint.exists()
            assert found_checkpoint.suffix == '.ckpt'
            
            # Verify content was correctly decompressed
            assert found_checkpoint.read_bytes() == original_content
    
    def test_find_checkpoint_with_complex_name(self):
        """Test finding compressed checkpoint with complex naming like tmp9lxui_on.ckpt.gz."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a compressed checkpoint with complex name (mimicking the bug scenario)
            original_content = b"checkpoint data for complex name"
            compressed_path = Path(tmpdir) / "tmp9lxui_on.ckpt.gz"
            
            with gzip.open(compressed_path, 'wb') as f:
                f.write(original_content)
            
            strategy = WandbArtifactResumeStrategy()
            found_checkpoint = strategy._find_checkpoint_in_artifact(Path(tmpdir))
            
            # Should return the decompressed path
            expected_path = Path(tmpdir) / "tmp9lxui_on.ckpt"
            assert found_checkpoint == expected_path
            assert found_checkpoint.exists()
            assert found_checkpoint.read_bytes() == original_content
    
    def test_mixed_compressed_and_regular_checkpoints(self):
        """Test selecting between compressed and regular checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create both compressed and regular checkpoints
            regular_path = Path(tmpdir) / "model.pt"
            regular_path.write_bytes(b"regular checkpoint")
            
            compressed_path = Path(tmpdir) / "backup.ckpt.gz"
            with gzip.open(compressed_path, 'wb') as f:
                f.write(b"compressed checkpoint")
            
            strategy = WandbArtifactResumeStrategy()
            found_checkpoint = strategy._find_checkpoint_in_artifact(Path(tmpdir))
            
            # Should prefer model.pt based on priority
            assert found_checkpoint == regular_path
    
    def test_priority_selection_with_compressed(self):
        """Test that priority selection works with compressed checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple checkpoints with different priorities
            files = {
                "random.ckpt": b"random checkpoint",
                "last.ckpt.gz": b"last checkpoint compressed",
                "model.pt": b"model checkpoint"
            }
            
            for name, content in files.items():
                path = Path(tmpdir) / name
                if name.endswith('.gz'):
                    with gzip.open(path, 'wb') as f:
                        f.write(content)
                else:
                    path.write_bytes(content)
            
            strategy = WandbArtifactResumeStrategy()
            
            # Test the selection
            checkpoints = list(Path(tmpdir).glob("**/*"))
            selected = strategy._select_best_checkpoint(checkpoints)
            
            # Should select last.ckpt.gz based on priority
            assert selected.name == "last.ckpt.gz"
    
    def test_best_checkpoint_priority_compressed(self):
        """Test that best.ckpt.gz has highest priority."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple checkpoints
            files = {
                "best.ckpt.gz": b"best checkpoint compressed",
                "checkpoint.ckpt": b"regular checkpoint",
                "last.ckpt": b"last checkpoint"
            }
            
            for name, content in files.items():
                path = Path(tmpdir) / name
                if name.endswith('.gz'):
                    with gzip.open(path, 'wb') as f:
                        f.write(content)
                else:
                    path.write_bytes(content)
            
            strategy = WandbArtifactResumeStrategy()
            
            # Test the selection
            checkpoints = list(Path(tmpdir).glob("**/*"))
            selected = strategy._select_best_checkpoint(checkpoints)
            
            # Should select best.ckpt.gz as highest priority
            assert selected.name == "best.ckpt.gz"
    
    def test_reuse_existing_decompressed_file(self):
        """Test that existing decompressed files are reused instead of re-decompressing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a compressed checkpoint
            compressed_path = Path(tmpdir) / "checkpoint.ckpt.gz"
            with gzip.open(compressed_path, 'wb') as f:
                f.write(b"compressed content")
            
            # Create an existing decompressed file
            decompressed_path = Path(tmpdir) / "checkpoint.ckpt"
            decompressed_path.write_bytes(b"existing decompressed content")
            
            strategy = WandbArtifactResumeStrategy()
            found_checkpoint = strategy._find_checkpoint_in_artifact(Path(tmpdir))
            
            # Should return the existing decompressed file
            assert found_checkpoint == decompressed_path
            # Content should be the existing one, not re-decompressed
            assert found_checkpoint.read_bytes() == b"existing decompressed content"
    
    def test_compressed_pytorch_checkpoints(self):
        """Test handling of compressed PyTorch checkpoint formats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test various compressed PyTorch formats
            formats = ["model.pt.gz", "weights.pth.gz", "checkpoint.pkl.gz"]
            
            for format_name in formats:
                # Clear directory
                for f in Path(tmpdir).glob("*"):
                    f.unlink()
                
                # Create compressed file
                compressed_path = Path(tmpdir) / format_name
                with gzip.open(compressed_path, 'wb') as f:
                    f.write(b"pytorch checkpoint data")
                
                strategy = WandbArtifactResumeStrategy()
                found_checkpoint = strategy._find_checkpoint_in_artifact(Path(tmpdir))
                
                # Should decompress and return decompressed path
                expected_name = format_name[:-3]  # Remove .gz
                assert found_checkpoint.name == expected_name
                assert found_checkpoint.exists()
    
    def test_decompression_error_handling(self):
        """Test error handling when decompression fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create an invalid compressed file
            compressed_path = Path(tmpdir) / "checkpoint.ckpt.gz"
            compressed_path.write_bytes(b"invalid gzip data")
            
            strategy = WandbArtifactResumeStrategy()
            
            # Should raise RuntimeError with appropriate message
            with pytest.raises(RuntimeError) as exc_info:
                strategy._find_checkpoint_in_artifact(Path(tmpdir))
            
            assert "Failed to decompress checkpoint" in str(exc_info.value)
    
    def test_no_checkpoint_found_error(self):
        """Test error when no checkpoint files are found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create non-checkpoint files
            (Path(tmpdir) / "config.yaml").write_text("config: data")
            (Path(tmpdir) / "README.md").write_text("readme content")
            
            strategy = WandbArtifactResumeStrategy()
            
            # Should raise ValueError
            with pytest.raises(ValueError) as exc_info:
                strategy._find_checkpoint_in_artifact(Path(tmpdir))
            
            assert "No checkpoint file found" in str(exc_info.value)
    
    def test_nested_directory_compressed_checkpoint(self):
        """Test finding compressed checkpoints in nested directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested directory structure
            nested_dir = Path(tmpdir) / "artifacts" / "checkpoints"
            nested_dir.mkdir(parents=True)
            
            # Create compressed checkpoint in nested directory
            compressed_path = nested_dir / "model.ckpt.gz"
            with gzip.open(compressed_path, 'wb') as f:
                f.write(b"nested checkpoint data")
            
            strategy = WandbArtifactResumeStrategy()
            found_checkpoint = strategy._find_checkpoint_in_artifact(Path(tmpdir))
            
            # Should find and decompress the nested checkpoint
            expected_path = nested_dir / "model.ckpt"
            assert found_checkpoint == expected_path
            assert found_checkpoint.exists()


class TestIntegrationWithWandbService:
    """Integration tests with the W&B service for compressed checkpoints."""
    
    @patch('lightning_reflow.strategies.wandb_artifact_resume_strategy.WandbService')
    def test_prepare_resume_with_compressed_checkpoint(self, mock_wandb_service_class):
        """Test full prepare_resume flow with compressed checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup mock W&B service
            mock_service = Mock()
            mock_wandb_service_class.return_value = mock_service
            
            # Create a fake downloaded artifact directory with compressed checkpoint
            artifact_dir = Path(tmpdir) / "artifact"
            artifact_dir.mkdir()
            
            compressed_checkpoint = artifact_dir / "checkpoint.ckpt.gz"
            # Create a minimal valid checkpoint structure
            checkpoint_data = {
                'state_dict': {},
                'epoch': 10,
                'global_step': 1000
            }
            
            # Compress the checkpoint
            import pickle
            checkpoint_bytes = pickle.dumps(checkpoint_data)
            with gzip.open(compressed_checkpoint, 'wb') as f:
                f.write(checkpoint_bytes)
            
            # Mock the download_artifact method
            mock_service.download_artifact.return_value = (
                artifact_dir,
                {
                    'name': 'test-artifact',
                    'version': 'v1',
                    'entity': 'test-entity',
                    'project': 'test-project'
                }
            )
            
            # Mock extract_embedded_config to avoid actual checkpoint loading
            with patch('lightning_reflow.utils.checkpoint.checkpoint_utils.extract_embedded_config') as mock_extract:
                mock_extract.return_value = "config: test"
                
                strategy = WandbArtifactResumeStrategy()
                checkpoint_path, config = strategy.prepare_resume(
                    resume_source="test-entity/test-project/test-artifact:v1"
                )
            
            # Verify the checkpoint was decompressed
            assert checkpoint_path.suffix == '.ckpt'
            assert checkpoint_path.exists()
            assert not checkpoint_path.name.endswith('.gz')
            
            # Verify we can load the decompressed checkpoint
            loaded_checkpoint = pickle.loads(checkpoint_path.read_bytes())
            assert loaded_checkpoint['epoch'] == 10
            assert loaded_checkpoint['global_step'] == 1000
    
    @patch('lightning_reflow.strategies.wandb_artifact_resume_strategy.WandbService')
    def test_backward_compatibility_regular_checkpoint(self, mock_wandb_service_class):
        """Test that regular checkpoints still work (backward compatibility)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup mock W&B service
            mock_service = Mock()
            mock_wandb_service_class.return_value = mock_service
            
            # Create artifact directory with regular checkpoint
            artifact_dir = Path(tmpdir) / "artifact"
            artifact_dir.mkdir()
            
            regular_checkpoint = artifact_dir / "checkpoint.ckpt"
            checkpoint_data = {
                'state_dict': {},
                'epoch': 5,
                'global_step': 500
            }
            
            import pickle
            regular_checkpoint.write_bytes(pickle.dumps(checkpoint_data))
            
            # Mock the download_artifact method
            mock_service.download_artifact.return_value = (
                artifact_dir,
                {
                    'name': 'test-artifact',
                    'version': 'v1',
                    'entity': 'test-entity',
                    'project': 'test-project'
                }
            )
            
            # Mock extract_embedded_config
            with patch('lightning_reflow.utils.checkpoint.checkpoint_utils.extract_embedded_config') as mock_extract:
                mock_extract.return_value = None
                
                strategy = WandbArtifactResumeStrategy()
                checkpoint_path, config = strategy.prepare_resume(
                    resume_source="test-entity/test-project/test-artifact:v1"
                )
            
            # Should return the regular checkpoint as-is
            assert checkpoint_path == regular_checkpoint
            assert checkpoint_path.exists()
            
            # Verify we can load it
            loaded_checkpoint = pickle.loads(checkpoint_path.read_bytes())
            assert loaded_checkpoint['epoch'] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])