"""
Tests for ConfigMixin - config-based serialization for nn.Module classes.
"""

import pytest
import torch
import torch.nn as nn

from lightning_reflow.utils.config import ConfigMixin, _import_class, _deserialize_value


class SimpleModel(ConfigMixin, nn.Module):
    """Simple model for testing basic config capture."""

    def __init__(self, hidden_dim: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.save_config()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return self.linear(x)


class InnerModel(ConfigMixin, nn.Module):
    """Inner model for testing nested config capture."""

    def __init__(self, dim: int, activation: str = "relu"):
        super().__init__()
        self.save_config()

        self.dim = dim
        self.activation = activation
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear(x)


class OuterModel(ConfigMixin, nn.Module):
    """Outer model wrapping an inner model for testing nested configs."""

    def __init__(self, inner: nn.Module, scale: float, use_bias: bool = True):
        super().__init__()
        self.save_config()

        self.inner = inner
        self.scale = scale
        self.use_bias = use_bias

    def forward(self, x):
        return self.inner(x) * self.scale


class ModelWithCollections(ConfigMixin, nn.Module):
    """Model with list and dict arguments for testing collection serialization."""

    def __init__(self, dims: list, options: dict, factors: tuple):
        super().__init__()
        self.save_config()

        self.dims = dims
        self.options = options
        self.factors = factors

    def forward(self, x):
        return x


class ModelWithIgnore(ConfigMixin, nn.Module):
    """Model that ignores certain arguments."""

    def __init__(self, dim: int, secret: str, debug: bool = False):
        super().__init__()
        self.save_config(ignore=['secret', 'debug'])

        self.dim = dim

    def forward(self, x):
        return x


class Encoder(ConfigMixin, nn.Module):
    """Encoder for testing adapter pattern."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.save_config()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)


class Adapter(ConfigMixin, nn.Module):
    """Adapter wrapping an encoder for testing nested configs."""

    def __init__(self, wrapped: nn.Module, input_dim: int, output_dim: int):
        super().__init__()
        self.save_config()
        self.wrapped = wrapped
        self.input_proj = nn.Linear(input_dim, wrapped.in_dim)
        self.output_proj = nn.Linear(wrapped.out_dim, output_dim)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.wrapped(x)
        return self.output_proj(x)


class TestConfigMixinBasic:
    """Test basic save_config and get_config functionality."""

    def test_save_config_captures_args(self):
        """Test that save_config captures __init__ arguments."""
        model = SimpleModel(hidden_dim=64, num_layers=4, dropout=0.2)
        config = model.get_config()

        assert config['hidden_dim'] == 64
        assert config['num_layers'] == 4
        assert config['dropout'] == 0.2

    def test_save_config_captures_defaults(self):
        """Test that save_config captures default argument values."""
        model = SimpleModel(hidden_dim=32, num_layers=2)
        config = model.get_config()

        assert config['hidden_dim'] == 32
        assert config['num_layers'] == 2
        assert config['dropout'] == 0.1  # Default value

    def test_get_config_returns_copy(self):
        """Test that get_config returns a copy, not the original."""
        model = SimpleModel(hidden_dim=64, num_layers=4)
        config1 = model.get_config()
        config2 = model.get_config()

        assert config1 is not config2
        config1['hidden_dim'] = 999
        assert model.get_config()['hidden_dim'] == 64

    def test_get_config_without_save_raises(self):
        """Test that get_config raises if save_config wasn't called."""

        class BadModel(ConfigMixin, nn.Module):
            def __init__(self, dim: int):
                super().__init__()
                # Forgot to call save_config()
                self.dim = dim

        model = BadModel(dim=64)
        with pytest.raises(RuntimeError, match="save_config.*was not called"):
            model.get_config()


class TestConfigMixinFromConfig:
    """Test from_config reconstruction."""

    def test_from_config_reconstructs_simple_model(self):
        """Test that from_config reconstructs a model correctly."""
        original = SimpleModel(hidden_dim=128, num_layers=6, dropout=0.3)
        config = original.get_config()

        reconstructed = SimpleModel.from_config(config)

        assert reconstructed.hidden_dim == 128
        assert reconstructed.num_layers == 6
        assert reconstructed.dropout == 0.3
        assert isinstance(reconstructed.linear, nn.Linear)

    def test_from_config_creates_new_instance(self):
        """Test that from_config creates a new instance with new weights."""
        original = SimpleModel(hidden_dim=64, num_layers=2)
        config = original.get_config()

        reconstructed = SimpleModel.from_config(config)

        # Should be different instances
        assert original is not reconstructed

        # Weights should be different (newly initialized)
        assert not torch.allclose(
            original.linear.weight,
            reconstructed.linear.weight
        )


class TestConfigMixinNested:
    """Test nested ConfigMixin modules."""

    def test_nested_config_extraction(self):
        """Test that nested ConfigMixin modules have their config extracted."""
        inner = InnerModel(dim=64, activation="gelu")
        outer = OuterModel(inner=inner, scale=2.0)

        config = outer.get_config()

        # Inner model should be serialized as class_path + config
        assert '__class_path__' in config['inner']
        assert '__config__' in config['inner']
        assert 'InnerModel' in config['inner']['__class_path__']
        assert config['inner']['__config__']['dim'] == 64
        assert config['inner']['__config__']['activation'] == "gelu"

        # Other args should be normal
        assert config['scale'] == 2.0
        assert config['use_bias'] is True

    def test_nested_from_config_reconstruction(self):
        """Test that nested modules are reconstructed correctly."""
        inner = InnerModel(dim=32, activation="tanh")
        outer = OuterModel(inner=inner, scale=0.5, use_bias=False)

        config = outer.get_config()
        reconstructed = OuterModel.from_config(config)

        # Outer model attributes
        assert reconstructed.scale == 0.5
        assert reconstructed.use_bias is False

        # Inner model should be reconstructed
        assert isinstance(reconstructed.inner, InnerModel)
        assert reconstructed.inner.dim == 32
        assert reconstructed.inner.activation == "tanh"

    def test_deeply_nested_config(self):
        """Test config extraction with multiple nesting levels."""
        inner = InnerModel(dim=16)
        middle = OuterModel(inner=inner, scale=1.0)
        outer = OuterModel(inner=middle, scale=2.0)

        config = outer.get_config()

        # Navigate to deeply nested config
        middle_config = config['inner']['__config__']
        inner_config = middle_config['inner']['__config__']

        assert inner_config['dim'] == 16

        # Reconstruct and verify
        reconstructed = OuterModel.from_config(config)
        assert isinstance(reconstructed.inner, OuterModel)
        assert isinstance(reconstructed.inner.inner, InnerModel)
        assert reconstructed.inner.inner.dim == 16


class TestConfigMixinCollections:
    """Test serialization of collections (list, dict, tuple)."""

    def test_list_serialization(self):
        """Test that lists are serialized and reconstructed correctly."""
        model = ModelWithCollections(
            dims=[64, 128, 256],
            options={'lr': 0.001},
            factors=(1, 2, 3)
        )
        config = model.get_config()

        assert config['dims'] == [64, 128, 256]

        reconstructed = ModelWithCollections.from_config(config)
        assert reconstructed.dims == [64, 128, 256]

    def test_dict_serialization(self):
        """Test that dicts are serialized and reconstructed correctly."""
        model = ModelWithCollections(
            dims=[32],
            options={'lr': 0.001, 'momentum': 0.9, 'nested': {'a': 1}},
            factors=(1,)
        )
        config = model.get_config()

        assert config['options'] == {'lr': 0.001, 'momentum': 0.9, 'nested': {'a': 1}}

        reconstructed = ModelWithCollections.from_config(config)
        assert reconstructed.options == {'lr': 0.001, 'momentum': 0.9, 'nested': {'a': 1}}

    def test_tuple_serialization(self):
        """Test that tuples are serialized and reconstructed correctly."""
        model = ModelWithCollections(
            dims=[32],
            options={},
            factors=(10, 20, 30)
        )
        config = model.get_config()

        # Tuples are stored with __tuple__ marker
        assert '__tuple__' in config['factors']
        assert config['factors']['__tuple__'] == [10, 20, 30]

        reconstructed = ModelWithCollections.from_config(config)
        assert reconstructed.factors == (10, 20, 30)
        assert isinstance(reconstructed.factors, tuple)


class TestConfigMixinIgnore:
    """Test ignore parameter in save_config."""

    def test_ignore_excludes_args(self):
        """Test that ignored args are not in config."""
        model = ModelWithIgnore(dim=64, secret="password123", debug=True)
        config = model.get_config()

        assert 'dim' in config
        assert config['dim'] == 64
        assert 'secret' not in config
        assert 'debug' not in config

    def test_from_config_with_ignored_args(self):
        """Test that from_config works when args were ignored."""
        model = ModelWithIgnore(dim=128, secret="ignored", debug=True)
        config = model.get_config()

        # Need to provide ignored args manually
        reconstructed = ModelWithIgnore.from_config({
            **config,
            'secret': 'new_secret',
            'debug': False
        })

        assert reconstructed.dim == 128


class TestImportClass:
    """Test _import_class utility function."""

    def test_import_class_standard_library(self):
        """Test importing a standard library class."""
        cls = _import_class('collections.OrderedDict')
        from collections import OrderedDict
        assert cls is OrderedDict

    def test_import_class_torch(self):
        """Test importing a torch class."""
        cls = _import_class('torch.nn.Linear')
        assert cls is nn.Linear

    def test_import_class_invalid_module(self):
        """Test that invalid module raises ImportError."""
        with pytest.raises(ImportError):
            _import_class('nonexistent_module.SomeClass')

    def test_import_class_invalid_class(self):
        """Test that invalid class raises AttributeError."""
        with pytest.raises(AttributeError):
            _import_class('torch.nn.NonexistentClass')


class TestDeserializeValue:
    """Test _deserialize_value utility function."""

    def test_deserialize_primitive(self):
        """Test deserializing primitive values."""
        assert _deserialize_value(42) == 42
        assert _deserialize_value("hello") == "hello"
        assert _deserialize_value(3.14) == 3.14
        assert _deserialize_value(True) is True
        assert _deserialize_value(None) is None

    def test_deserialize_list(self):
        """Test deserializing lists."""
        assert _deserialize_value([1, 2, 3]) == [1, 2, 3]

    def test_deserialize_dict(self):
        """Test deserializing dicts."""
        assert _deserialize_value({'a': 1, 'b': 2}) == {'a': 1, 'b': 2}

    def test_deserialize_tuple(self):
        """Test deserializing tuples."""
        result = _deserialize_value({'__tuple__': [1, 2, 3]})
        assert result == (1, 2, 3)
        assert isinstance(result, tuple)

    def test_deserialize_nested_configmixin(self):
        """Test deserializing nested ConfigMixin config."""
        # Get the actual module path for the test class
        class_path = f"{InnerModel.__module__}.{InnerModel.__name__}"
        config = {
            '__class_path__': class_path,
            '__config__': {'dim': 64, 'activation': 'relu'}
        }
        result = _deserialize_value(config)

        # Check by class name since module paths can vary in test context
        assert result.__class__.__name__ == 'InnerModel'
        assert result.dim == 64
        assert result.activation == 'relu'


class TestConfigMixinIntegration:
    """Integration tests for ConfigMixin with real-world patterns."""

    def test_adapter_pattern(self):
        """Test ConfigMixin with adapter/wrapper pattern (like AdapterDynamicsModel)."""
        # Uses module-level Encoder and Adapter classes

        # Create adapter wrapping encoder
        encoder = Encoder(in_dim=64, out_dim=32)
        adapter = Adapter(wrapped=encoder, input_dim=128, output_dim=256)

        # Get and verify config
        config = adapter.get_config()
        assert config['input_dim'] == 128
        assert config['output_dim'] == 256
        assert config['wrapped']['__config__']['in_dim'] == 64
        assert config['wrapped']['__config__']['out_dim'] == 32

        # Reconstruct and verify
        reconstructed = Adapter.from_config(config)
        assert reconstructed.wrapped.__class__.__name__ == 'Encoder'
        assert reconstructed.wrapped.in_dim == 64
        assert reconstructed.input_proj.in_features == 128
        assert reconstructed.output_proj.out_features == 256

    def test_forward_pass_after_reconstruction(self):
        """Test that reconstructed models can perform forward passes."""
        original = SimpleModel(hidden_dim=64, num_layers=2)
        config = original.get_config()
        reconstructed = SimpleModel.from_config(config)

        # Both should handle forward pass
        x = torch.randn(4, 64)
        out_original = original(x)
        out_reconstructed = reconstructed(x)

        assert out_original.shape == (4, 64)
        assert out_reconstructed.shape == (4, 64)

    def test_config_is_json_serializable(self):
        """Test that config can be serialized to JSON."""
        import json

        inner = InnerModel(dim=64)
        outer = OuterModel(inner=inner, scale=2.0)
        config = outer.get_config()

        # Should not raise
        json_str = json.dumps(config)
        loaded = json.loads(json_str)

        # Should be able to reconstruct from loaded JSON
        reconstructed = OuterModel.from_config(loaded)
        assert isinstance(reconstructed.inner, InnerModel)
