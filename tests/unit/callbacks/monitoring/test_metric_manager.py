"""
Unit tests for MetricManager.

Tests metric pattern matching, caching, and formatting.
"""

import pytest
from unittest.mock import Mock, MagicMock

from lightning_reflow.callbacks.monitoring.metric_manager import MetricManager


class TestMetricManager:
    """Test MetricManager functionality."""

    @pytest.fixture
    def metric_manager(self):
        """Create a metric manager for testing."""
        return MetricManager(
            global_bar_patterns=['*lr*'],
            interval_bar_patterns=['loss', 'train_*']
        )

    @pytest.fixture
    def mock_trainer(self):
        """Create a mock trainer."""
        trainer = Mock()
        trainer.current_epoch = 5
        trainer.global_step = 100
        trainer.accumulate_grad_batches = 1
        trainer.callback_metrics = {
            'loss': 0.5,
            'train_accuracy': 0.95,
            'val_accuracy': 0.90,
            'lr': 0.001
        }
        return trainer

    def test_initialization(self, metric_manager):
        """Test metric manager initialization."""
        assert metric_manager.global_bar_patterns == ['*lr*']
        assert metric_manager.interval_bar_patterns == ['loss', 'train_*']
        assert metric_manager._prog_bar_metrics == {}

    def test_initialization_defaults(self):
        """Test metric manager with default patterns."""
        manager = MetricManager()
        assert manager.global_bar_patterns == ['*lr*']
        assert manager.interval_bar_patterns == ['loss']

    def test_set_trainer(self, metric_manager, mock_trainer):
        """Test setting trainer reference."""
        metric_manager.set_trainer(mock_trainer)
        assert metric_manager._trainer is mock_trainer

    def test_set_batch_idx(self, metric_manager):
        """Test setting batch index."""
        metric_manager.set_batch_idx(42)
        assert metric_manager._current_batch_idx == 42

    def test_reset_caches(self, metric_manager):
        """Test cache reset."""
        metric_manager._global_metrics = {'key': 'value'}
        metric_manager._interval_metrics = {'key': 'value'}
        metric_manager._global_metric_keys_cache = {'key'}

        metric_manager.reset_caches()

        assert metric_manager._global_metrics is None
        assert metric_manager._interval_metrics is None
        assert metric_manager._global_metric_keys_cache is None

    def test_format_metrics_postfix(self, metric_manager):
        """Test metric formatting."""
        metrics = {'loss': '0.5000', 'lr': '0.0010'}
        result = metric_manager.format_metrics_postfix(metrics)

        assert 'loss=0.5000' in result
        assert 'lr=0.0010' in result

    def test_format_metrics_postfix_empty(self, metric_manager):
        """Test formatting empty metrics."""
        result = metric_manager.format_metrics_postfix({})
        assert result == ""


class TestMetricPatternMatching:
    """Test pattern matching functionality."""

    @pytest.fixture
    def metric_manager(self):
        """Create a metric manager for testing."""
        return MetricManager()

    def test_match_exact_pattern(self, metric_manager):
        """Test exact pattern matching."""
        available = {'loss': '0.5', 'lr': '0.001'}
        result = metric_manager._match_metrics_by_pattern('loss', available)
        assert result == {'loss': '0.5'}

    def test_match_glob_pattern(self, metric_manager):
        """Test glob pattern matching."""
        available = {'train_loss': '0.5', 'val_loss': '0.4', 'lr': '0.001'}
        result = metric_manager._match_metrics_by_pattern('*loss', available)
        assert 'train_loss' in result
        assert 'val_loss' in result
        assert 'lr' not in result

    def test_match_wildcard_in_middle(self, metric_manager):
        """Test wildcard in middle of pattern."""
        available = {'train_loss_step': '0.5', 'val_loss_epoch': '0.4', 'accuracy': '0.9'}
        result = metric_manager._match_metrics_by_pattern('*loss*', available)
        assert 'train_loss_step' in result
        assert 'val_loss_epoch' in result
        assert 'accuracy' not in result

    def test_match_no_match(self, metric_manager):
        """Test pattern with no matches."""
        available = {'loss': '0.5', 'accuracy': '0.9'}
        result = metric_manager._match_metrics_by_pattern('*lr*', available)
        assert result == {}

    def test_get_matched_keys_for_patterns(self, metric_manager):
        """Test matching keys for multiple patterns."""
        available_keys = {'loss', 'train_accuracy', 'val_accuracy', 'lr'}
        special_keys = {'epoch', 'step'}

        result = metric_manager._get_matched_keys_for_patterns(
            ['loss', '*accuracy', 'epoch'],
            available_keys,
            special_keys
        )

        assert 'loss' in result
        assert 'train_accuracy' in result
        assert 'val_accuracy' in result
        assert 'epoch' in result
        assert 'lr' not in result


class TestSpecialMetrics:
    """Test special metrics functionality."""

    @pytest.fixture
    def metric_manager(self):
        """Create a metric manager for testing."""
        return MetricManager()

    def test_get_special_metrics_with_trainer(self, metric_manager):
        """Test getting special metrics with trainer."""
        trainer = Mock()
        trainer.current_epoch = 5
        trainer.global_step = 100

        metric_manager.set_trainer(trainer)
        metric_manager.set_batch_idx(42)

        result = metric_manager._get_special_metrics()

        assert result['epoch'] == '6'  # current_epoch + 1
        assert result['step'] == '100'
        assert result['batch_idx'] == '42'

    def test_get_special_metrics_no_trainer(self, metric_manager):
        """Test getting special metrics without trainer."""
        result = metric_manager._get_special_metrics()
        assert result == {}

    def test_get_special_metrics_no_batch_idx(self, metric_manager):
        """Test getting special metrics without batch index."""
        trainer = Mock()
        trainer.current_epoch = 0
        trainer.global_step = 0

        metric_manager.set_trainer(trainer)

        result = metric_manager._get_special_metrics()

        assert 'epoch' in result
        assert 'step' in result
        assert 'batch_idx' not in result


class TestMetricPopulation:
    """Test metric population and caching."""

    @pytest.fixture
    def metric_manager(self):
        """Create a metric manager for testing."""
        return MetricManager(
            global_bar_patterns=['*lr*', 'step'],
            interval_bar_patterns=['loss']
        )

    def test_populate_metrics_caching(self, metric_manager):
        """Test that metric keys are cached."""
        trainer = Mock()
        trainer.current_epoch = 0
        trainer.global_step = 100

        metric_manager.set_trainer(trainer)
        metric_manager._prog_bar_metrics = {'loss': '0.5', 'lr': '0.001'}

        # First call should populate caches
        metric_manager.populate_metrics_if_needed()

        assert metric_manager._global_metric_keys_cache is not None
        assert metric_manager._interval_metric_keys_cache is not None
        assert metric_manager._available_metric_keys_cache is not None

    def test_populate_metrics_uses_cache(self, metric_manager):
        """Test that subsequent calls use cached keys."""
        trainer = Mock()
        trainer.current_epoch = 0
        trainer.global_step = 100

        metric_manager.set_trainer(trainer)
        metric_manager._prog_bar_metrics = {'loss': '0.5'}

        # First call
        metric_manager.populate_metrics_if_needed()
        first_cache = metric_manager._interval_metric_keys_cache

        # Update value but not keys
        metric_manager._prog_bar_metrics = {'loss': '0.6'}

        # Second call should use cache
        metric_manager.populate_metrics_if_needed()

        assert metric_manager._interval_metric_keys_cache is first_cache

    def test_populate_metrics_force_refresh(self, metric_manager):
        """Test force refresh recalculates keys."""
        trainer = Mock()
        trainer.current_epoch = 0
        trainer.global_step = 100

        metric_manager.set_trainer(trainer)
        metric_manager._prog_bar_metrics = {'loss': '0.5'}

        metric_manager.populate_metrics_if_needed()
        metric_manager._interval_metric_keys_cache = {'old_key'}

        # Force refresh should recalculate
        metric_manager.populate_metrics_if_needed(force_refresh=True)

        assert 'old_key' not in metric_manager._interval_metric_keys_cache
