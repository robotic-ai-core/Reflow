"""
Tests for stale config detection in config embedding and validation.

These tests ensure that LightningReflow properly detects and fails fast when:
1. Config files are stale (older than 1 minute)
2. Config files are in parent/fallback directories
3. Config files are missing entirely
"""

import pytest
import time
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import tempfile
import shutil

from lightning.pytorch import Trainer, LightningModule
from lightning_reflow.callbacks.core.config_embedding_mixin import ConfigEmbeddingMixin
from lightning_reflow.callbacks.core.config_validation_callback import ConfigValidationCallback


class MockModule(LightningModule):
    """Minimal Lightning module for testing."""
    def __init__(self):
        super().__init__()

    def training_step(self, batch, batch_idx):
        return {}


@pytest.fixture
def temp_log_dir():
    """Create a temporary log directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_trainer(temp_log_dir):
    """Create a mock trainer with proper log directory."""
    trainer = Mock(spec=Trainer)
    trainer.global_step = 0
    trainer.current_epoch = 0

    # Mock logger with log_dir
    mock_logger = Mock()
    mock_logger.log_dir = str(temp_log_dir)
    trainer.logger = mock_logger

    # Mock CLI context
    mock_cli = Mock()
    mock_cli.save_config_kwargs = {}  # Config saving enabled
    trainer.cli = mock_cli

    return trainer


class TestStaleConfigDetection:
    """Test stale config detection at different stages."""

    def test_fresh_config_passes_validation(self, temp_log_dir, mock_trainer):
        """Test that a fresh config (created recently) passes validation."""
        # Create a fresh config file
        config_path = temp_log_dir / "config.yaml"
        config_path.write_text("model:\n  class_path: test.Model\n")

        # Validation should pass
        mixin = ConfigEmbeddingMixin()
        config = mixin._capture_lightning_auto_config(mock_trainer)

        assert config is not None
        assert "model:" in config

    def test_stale_config_warns_when_old(self, temp_log_dir, mock_trainer, caplog):
        """Test that a config older than 1 minute triggers a warning."""
        # Create a config file and make it stale (2 minutes old)
        config_path = temp_log_dir / "config.yaml"
        config_path.write_text("model:\n  class_path: test.Model\n")

        # Set modification time to 2 minutes ago
        two_minutes_ago = time.time() - 120
        import os
        os.utime(config_path, (two_minutes_ago, two_minutes_ago))

        # Should warn but not fail (unless combined with other issues)
        mixin = ConfigEmbeddingMixin()
        with caplog.at_level("WARNING"):
            config = mixin._capture_lightning_auto_config(mock_trainer)

        # Check that warning was logged
        assert "seconds old" in caplog.text
        assert config is not None  # Still succeeds, just warns

    def test_missing_config_raises_error(self, temp_log_dir, mock_trainer):
        """Test that missing config file raises clear error."""
        # Don't create config file

        mixin = ConfigEmbeddingMixin()
        with pytest.raises(RuntimeError, match="Lightning config not found"):
            mixin._capture_lightning_auto_config(mock_trainer)

    def test_config_in_parent_directory_raises_error(self, temp_log_dir):
        """Test that config in parent directory (stale location) raises error."""
        # Create directory structure: parent/logs/
        parent_dir = temp_log_dir
        run_dir = parent_dir / "logs" / "run_xyz"
        run_dir.mkdir(parents=True)

        # Put config in PARENT instead of run directory (STALE!)
        stale_config = parent_dir / "config.yaml"
        stale_config.write_text("model:\n  class_path: test.Model\n")

        # Mock trainer pointing to run_dir, but config is in parent
        trainer = Mock(spec=Trainer)
        mock_logger = Mock()
        mock_logger.log_dir = str(run_dir)
        trainer.logger = mock_logger
        mock_cli = Mock()
        mock_cli.save_config_kwargs = {}
        trainer.cli = mock_cli

        # Should raise because config is not found in expected location
        # (It looks in run_dir, not parent_dir)
        mixin = ConfigEmbeddingMixin()
        with pytest.raises(RuntimeError, match="Lightning config not found"):
            mixin._capture_lightning_auto_config(trainer)

    def test_empty_config_raises_error(self, temp_log_dir, mock_trainer):
        """Test that empty config file raises error."""
        # Create empty config file
        config_path = temp_log_dir / "config.yaml"
        config_path.write_text("")

        mixin = ConfigEmbeddingMixin()
        with pytest.raises(RuntimeError, match="Lightning config.*is empty"):
            mixin._capture_lightning_auto_config(mock_trainer)

    def test_save_config_callback_disabled_raises_error(self, temp_log_dir):
        """Test that save_config_callback=None raises clear error."""
        # Create trainer with save_config_callback=None
        trainer = Mock(spec=Trainer)
        mock_logger = Mock()
        mock_logger.log_dir = str(temp_log_dir)
        trainer.logger = mock_logger

        # save_config_kwargs=False indicates config saving is disabled
        mock_cli = Mock()
        mock_cli.save_config_kwargs = False
        trainer.cli = mock_cli

        # Should be detected early
        mixin = ConfigEmbeddingMixin()
        can_embed = mixin._can_embed_config(trainer)

        assert not can_embed  # Should detect that config embedding is disabled


class TestConfigValidationCallback:
    """Test the ConfigValidationCallback that runs at training start."""

    def test_validation_callback_passes_with_fresh_config(self, temp_log_dir, mock_trainer):
        """Test that validation callback passes with fresh config."""
        # Create fresh config
        config_path = temp_log_dir / "config.yaml"
        config_path.write_text("model:\n  class_path: test.Model\n")

        callback = ConfigValidationCallback()
        module = MockModule()

        # Should not raise
        callback.on_fit_start(mock_trainer, module)

    def test_validation_callback_fails_with_missing_config(self, temp_log_dir, mock_trainer):
        """Test that validation callback fails fast with missing config."""
        # Don't create config file

        callback = ConfigValidationCallback()
        module = MockModule()

        # Should raise immediately at fit start
        with pytest.raises(RuntimeError, match="Training aborted.*config validation failed"):
            callback.on_fit_start(mock_trainer, module)

    def test_validation_callback_fails_with_stale_config_in_parent(self, temp_log_dir):
        """Test that validation callback detects stale config in parent directory."""
        # Create directory structure with stale config in parent
        parent_dir = temp_log_dir
        run_dir = parent_dir / "logs" / "run_xyz"
        run_dir.mkdir(parents=True)

        # Put config in parent (stale location)
        stale_config = parent_dir / "config.yaml"
        stale_config.write_text("model:\n  class_path: test.Model\n")

        # Mock trainer pointing to run_dir
        trainer = Mock(spec=Trainer)
        mock_logger = Mock()
        mock_logger.log_dir = str(run_dir)
        trainer.logger = mock_logger
        mock_cli = Mock()
        mock_cli.save_config_kwargs = {}
        trainer.cli = mock_cli

        callback = ConfigValidationCallback()
        module = MockModule()

        # Should fail fast at training start
        with pytest.raises(RuntimeError, match="Training aborted"):
            callback.on_fit_start(trainer, module)

    def test_validation_callback_with_no_cli_context(self, temp_log_dir):
        """Test validation callback behavior when CLI context is unavailable."""
        # Create trainer without CLI context
        trainer = Mock(spec=Trainer)
        mock_logger = Mock()
        mock_logger.log_dir = str(temp_log_dir)
        trainer.logger = mock_logger
        trainer.cli = None  # No CLI context

        callback = ConfigValidationCallback()
        module = MockModule()

        # Should warn but not fail
        callback.on_fit_start(trainer, module)  # No exception


class TestConfigEmbeddingWithStaleDetection:
    """Test config embedding behavior with stale config detection."""

    def test_checkpoint_save_warns_with_stale_config(self, temp_log_dir, mock_trainer, caplog):
        """Test that checkpoint save warns if config is stale."""
        # Create stale config (2 hours old)
        config_path = temp_log_dir / "config.yaml"
        config_path.write_text("model:\n  class_path: test.Model\n")

        # Make it very old
        very_old_time = time.time() - 7200  # 2 hours old
        import os
        os.utime(config_path, (very_old_time, very_old_time))

        mixin = ConfigEmbeddingMixin()
        checkpoint = {}
        module = MockModule()

        # Checkpoint save should warn about stale config
        with caplog.at_level("WARNING"):
            mixin.add_config_metadata(mock_trainer, module, checkpoint)

        # Check that warning was logged
        assert "seconds old" in caplog.text
        # Checkpoint still succeeds (just warns)
        assert 'self_contained_metadata' in checkpoint

    def test_checkpoint_with_fresh_config_succeeds(self, temp_log_dir, mock_trainer):
        """Test that checkpoint save succeeds with fresh config."""
        # Create fresh config
        config_path = temp_log_dir / "config.yaml"
        config_path.write_text("model:\n  class_path: test.Model\n")

        mixin = ConfigEmbeddingMixin()
        checkpoint = {}
        module = MockModule()

        # Should succeed
        mixin.add_config_metadata(mock_trainer, module, checkpoint)

        # Check metadata was added
        assert 'self_contained_metadata' in checkpoint
        assert 'embedded_config_content' in checkpoint['self_contained_metadata']
        assert checkpoint['self_contained_metadata']['config_source'] == 'lightning_auto_generated'

    def test_config_caching_eliminates_false_warnings(self, temp_log_dir, mock_trainer, caplog):
        """Test that config caching at training start eliminates false age warnings."""
        # Create fresh config
        config_path = temp_log_dir / "config.yaml"
        config_path.write_text("model:\n  class_path: test.Model\n")

        mixin = ConfigEmbeddingMixin()
        module = MockModule()

        # Capture and cache config at training start (when fresh)
        mixin.capture_and_cache_config(mock_trainer)

        # Simulate time passing (config naturally ages)
        import os
        old_time = time.time() - 120  # Make it 2 minutes old
        os.utime(config_path, (old_time, old_time))

        # Now save checkpoint - should use cached config, no warnings
        checkpoint = {}
        with caplog.at_level("DEBUG"):  # DEBUG level to capture the cached config message
            mixin.add_config_metadata(mock_trainer, module, checkpoint)

        # Should NOT warn about stale config because we're using cached version
        assert "seconds old" not in caplog.text
        # Should see message about using cached config (DEBUG level)
        assert "Using cached config from training start" in caplog.text
        # Checkpoint should succeed
        assert 'self_contained_metadata' in checkpoint
        assert 'embedded_config_content' in checkpoint['self_contained_metadata']

    def test_config_caching_only_captures_once(self, temp_log_dir, mock_trainer, caplog):
        """Test that config is only captured once at training start."""
        # Create fresh config
        config_path = temp_log_dir / "config.yaml"
        config_path.write_text("model:\n  class_path: test.Model\n")

        mixin = ConfigEmbeddingMixin()

        # First capture
        with caplog.at_level("INFO"):
            mixin.capture_and_cache_config(mock_trainer)

        assert "Capturing config at training start" in caplog.text
        assert mixin._config_captured_at_start is True
        assert mixin._cached_config is not None

        # Clear log
        caplog.clear()

        # Second capture attempt - should skip
        with caplog.at_level("DEBUG"):
            mixin.capture_and_cache_config(mock_trainer)

        assert "Config already captured at training start" in caplog.text

    def test_fallback_when_config_not_cached(self, temp_log_dir, mock_trainer, caplog):
        """Test that fallback works when config wasn't cached at training start."""
        # Create fresh config
        config_path = temp_log_dir / "config.yaml"
        config_path.write_text("model:\n  class_path: test.Model\n")

        mixin = ConfigEmbeddingMixin()
        module = MockModule()

        # Skip capture_and_cache_config to simulate backwards compatibility case
        # Directly save checkpoint - should fall back to reading from disk
        checkpoint = {}
        with caplog.at_level("WARNING"):
            mixin.add_config_metadata(mock_trainer, module, checkpoint)

        # Should warn about not using cached config
        assert "Config not cached at training start" in caplog.text
        assert "may show false age warnings" in caplog.text
        # But should still succeed
        assert 'self_contained_metadata' in checkpoint
        assert 'embedded_config_content' in checkpoint['self_contained_metadata']


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
