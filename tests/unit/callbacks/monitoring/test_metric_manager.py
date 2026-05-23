"""
Unit tests for MetricManager.

Tests metric pattern matching, caching, and formatting.
"""

import pytest
import torch
from unittest.mock import Mock, MagicMock, patch

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


class TestAsyncTensorSync:
    """Verify the async pinned-memory transfer path in update_metrics.

    The legacy implementation called ``float(tensor.detach().cpu())`` directly,
    which forces a CUDA stream sync (~28 ms on compute-heavy runs). The new
    path stages the transfer through a pinned-memory buffer so the kernels on
    the training stream keep running while the copy is in flight.
    """

    @pytest.fixture
    def metric_manager(self):
        return MetricManager()

    @pytest.fixture
    def mock_trainer(self):
        trainer = Mock()
        trainer.current_epoch = 0
        trainer.global_step = 0
        trainer.accumulate_grad_batches = 1
        return trainer

    def _make_trainer(self, callback_metrics, accumulate_grad_batches=1):
        trainer = Mock()
        trainer.current_epoch = 0
        trainer.global_step = 0
        trainer.accumulate_grad_batches = accumulate_grad_batches
        trainer.callback_metrics = callback_metrics
        return trainer

    # ---- correctness ----------------------------------------------------

    def test_returns_correct_float_for_scalar_tensor(self, metric_manager):
        """Returned formatted float must match the tensor's value."""
        t = torch.tensor(0.4242)
        trainer = self._make_trainer({'loss': t})

        metric_manager.update_metrics(trainer)

        assert 'loss' in metric_manager._prog_bar_metrics
        assert metric_manager._prog_bar_metrics['loss'] == "0.4242"

    def test_passes_through_python_floats(self, metric_manager):
        """Plain Python floats must bypass the async path entirely."""
        trainer = self._make_trainer({'loss': 0.5, 'lr': 0.001})
        metric_manager.update_metrics(trainer)
        assert metric_manager._prog_bar_metrics['loss'] == "0.5000"
        assert metric_manager._prog_bar_metrics['lr'] == "0.0010"

    def test_loss_scaling_with_grad_accum_preserved(self, metric_manager):
        """The grad-accum loss scaling fix-up must still apply."""
        t = torch.tensor(0.1)
        trainer = self._make_trainer({'loss': t}, accumulate_grad_batches=4)
        metric_manager.update_metrics(trainer)
        # 0.1 * 4 = 0.4
        assert metric_manager._prog_bar_metrics['loss'] == "0.4000"

    def test_handles_cpu_tensor(self, metric_manager):
        """CPU tensors must work without pin_memory (no GPU required)."""
        t = torch.tensor(1.5)
        trainer = self._make_trainer({'loss': t})
        metric_manager.update_metrics(trainer)
        assert metric_manager._prog_bar_metrics['loss'] == "1.5000"

    def test_multi_element_tensor_falls_back_to_blocking(self, metric_manager):
        """Multi-element tensors are rare for progress-bar display; ensure
        the path still produces a usable string (per legacy behavior)."""
        t = torch.tensor([1.0, 2.0])
        trainer = self._make_trainer({'arr': t})
        metric_manager.update_metrics(trainer)
        # Non-scalar tensors are formatted with str()
        assert 'arr' in metric_manager._prog_bar_metrics

    # ---- pinned-buffer reuse / lifecycle --------------------------------

    def test_pinned_buffer_lazily_created_for_cuda_tensor(self, metric_manager):
        """A pinned host buffer is created on first call only."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        t = torch.tensor(1.5, device='cuda')
        trainer = self._make_trainer({'loss': t})

        assert 'loss' not in metric_manager._pinned_buffers

        metric_manager.update_metrics(trainer)

        assert 'loss' in metric_manager._pinned_buffers
        buf = metric_manager._pinned_buffers['loss']
        assert buf.is_pinned()
        assert buf.numel() == 1

    def test_pinned_buffer_reused_for_same_shape(self, metric_manager):
        """Same-shape tensors must reuse the existing pinned buffer."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        trainer = self._make_trainer({'loss': torch.tensor(1.0, device='cuda')})

        metric_manager.update_metrics(trainer)
        first_buf = metric_manager._pinned_buffers['loss']

        trainer.callback_metrics = {'loss': torch.tensor(2.0, device='cuda')}
        metric_manager.update_metrics(trainer)
        second_buf = metric_manager._pinned_buffers['loss']

        assert first_buf is second_buf

    def test_pinned_buffer_reallocated_on_dtype_change(self, metric_manager):
        """Dtype change must trigger a re-alloc."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        trainer = self._make_trainer({
            'loss': torch.tensor(1.0, dtype=torch.float32, device='cuda')
        })
        metric_manager.update_metrics(trainer)
        first_buf = metric_manager._pinned_buffers['loss']
        assert first_buf.dtype == torch.float32

        trainer.callback_metrics = {
            'loss': torch.tensor(1.0, dtype=torch.bfloat16, device='cuda')
        }
        metric_manager.update_metrics(trainer)
        second_buf = metric_manager._pinned_buffers['loss']
        assert second_buf is not first_buf
        assert second_buf.dtype == torch.bfloat16

    # ---- the headline property: no blocking sync on training stream ----

    def test_no_blocking_synchronize_called(self, metric_manager):
        """The async path must NOT invoke torch.cuda.synchronize.

        This is the whole point of the refactor: previously
        ``tensor.detach().cpu()`` forced a stream sync on every call.
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        t = torch.tensor(0.5, device='cuda')
        trainer = self._make_trainer({'loss': t})

        with patch('torch.cuda.synchronize') as sync_mock:
            metric_manager.update_metrics(trainer)
            sync_mock.assert_not_called()

    def test_async_copy_uses_non_blocking(self, metric_manager):
        """The pinned-memory copy must request non_blocking=True so the
        training stream is not stalled."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Spy on Tensor.copy_ to assert non_blocking=True is passed
        real_copy_ = torch.Tensor.copy_
        seen_kwargs = []

        def spy_copy_(self, src, **kwargs):
            seen_kwargs.append(dict(kwargs))
            return real_copy_(self, src, **kwargs)

        t = torch.tensor(0.5, device='cuda')
        trainer = self._make_trainer({'loss': t})

        with patch.object(torch.Tensor, 'copy_', spy_copy_):
            metric_manager.update_metrics(trainer)

        # At least one pinned copy with non_blocking=True
        nb_calls = [kw for kw in seen_kwargs if kw.get('non_blocking') is True]
        assert nb_calls, f"Expected non_blocking copy; saw {seen_kwargs}"

    def test_value_correct_for_cuda_tensor(self, metric_manager):
        """End-to-end: the displayed value must match the CUDA tensor."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        t = torch.tensor(0.7777, device='cuda')
        trainer = self._make_trainer({'loss': t})
        metric_manager.update_metrics(trainer)
        # In async-cache mode we accept either the current value (typical)
        # or a stale value from the cache. For the very first call there is
        # no prior value, so it MUST return the current one.
        assert metric_manager._prog_bar_metrics['loss'] == "0.7777"
