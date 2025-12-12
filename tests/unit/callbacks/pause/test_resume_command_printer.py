"""
Unit tests for ResumeCommandPrinter.

Tests resume command generation, script detection, and fallback handling.
"""

import pytest
import sys
from unittest.mock import Mock, patch
from io import StringIO

from lightning_reflow.callbacks.pause.resume_command_printer import ResumeCommandPrinter


class TestResumeCommandPrinter:
    """Test ResumeCommandPrinter functionality."""

    @pytest.fixture
    def basic_argv(self):
        """Basic argv for testing."""
        return ['train_lightning.py', 'fit', '--model.lr=0.001']

    @pytest.fixture
    def command_printer(self, basic_argv):
        """Create a command printer for testing."""
        return ResumeCommandPrinter(basic_argv)

    @pytest.fixture
    def mock_trainer(self):
        """Create a mock trainer for testing."""
        return Mock()

    def test_initialization(self, basic_argv):
        """Test command printer initialization."""
        printer = ResumeCommandPrinter(basic_argv)
        assert printer._original_argv == basic_argv

    def test_print_resume_commands_basic(self, command_printer, mock_trainer, capsys):
        """Test basic resume command printing."""
        command_printer.print_resume_commands(
            mock_trainer, "/path/to/checkpoint.ckpt"
        )

        captured = capsys.readouterr()
        assert "Training paused" in captured.out
        assert "checkpoint.ckpt" in captured.out
        assert "resume --checkpoint-path" in captured.out

    def test_print_resume_commands_with_artifact(self, command_printer, mock_trainer, capsys):
        """Test resume command printing with artifact path."""
        command_printer.print_resume_commands(
            mock_trainer,
            "/path/to/checkpoint.ckpt",
            artifact_path="entity/project/artifact:v1"
        )

        captured = capsys.readouterr()
        assert "W&B resume" in captured.out
        assert "entity/project/artifact:v1" in captured.out
        assert "--checkpoint-artifact" in captured.out

    def test_print_resume_commands_empty_checkpoint_path(self, command_printer, mock_trainer):
        """Test that empty checkpoint path raises ValueError."""
        with pytest.raises(ValueError, match="Checkpoint path cannot be empty"):
            command_printer.print_resume_commands(mock_trainer, "")

    def test_print_resume_commands_no_argv(self, mock_trainer):
        """Test that missing argv raises ValueError."""
        printer = ResumeCommandPrinter([])

        with pytest.raises(ValueError, match="Original argv not stored"):
            printer.print_resume_commands(mock_trainer, "/path/checkpoint.ckpt")

    def test_print_resume_commands_legacy_format(self, command_printer, mock_trainer, capsys):
        """Test legacy command format is included."""
        command_printer.print_resume_commands(
            mock_trainer, "/path/to/checkpoint.ckpt"
        )

        captured = capsys.readouterr()
        assert "Legacy method" in captured.out
        assert "--ckpt_path" in captured.out


class TestResumeCommandPrinterWithFallback:
    """Test print_resume_commands_with_fallback method."""

    @pytest.fixture
    def command_printer(self):
        """Create a command printer for testing."""
        return ResumeCommandPrinter(['train.py', 'fit'])

    @pytest.fixture
    def mock_trainer(self):
        """Create a mock trainer for testing."""
        return Mock()

    def test_fallback_success(self, command_printer, mock_trainer, capsys):
        """Test fallback method works on success."""
        command_printer.print_resume_commands_with_fallback(
            mock_trainer, "/path/checkpoint.ckpt"
        )

        captured = capsys.readouterr()
        assert "Training paused" in captured.out

    def test_fallback_handles_value_error(self, mock_trainer, capsys):
        """Test fallback handles ValueError gracefully."""
        printer = ResumeCommandPrinter([])  # Empty argv triggers error

        printer.print_resume_commands_with_fallback(
            mock_trainer, "/path/checkpoint.ckpt"
        )

        captured = capsys.readouterr()
        assert "Could not generate resume commands" in captured.out
        assert "checkpoint.ckpt" in captured.out

    def test_fallback_handles_unexpected_error(self, command_printer, mock_trainer, capsys):
        """Test fallback handles unexpected errors gracefully."""
        with patch.object(command_printer, 'print_resume_commands',
                         side_effect=Exception("Unexpected error")):
            command_printer.print_resume_commands_with_fallback(
                mock_trainer, "/path/checkpoint.ckpt"
            )

        captured = capsys.readouterr()
        assert "Unexpected error" in captured.out
        assert "checkpoint.ckpt" in captured.out

    def test_fallback_includes_artifact_in_error_case(self, mock_trainer, capsys):
        """Test fallback includes artifact path even when primary fails."""
        printer = ResumeCommandPrinter([])  # Empty argv triggers error

        printer.print_resume_commands_with_fallback(
            mock_trainer, "/path/checkpoint.ckpt",
            artifact_path="entity/project/artifact:v1"
        )

        captured = capsys.readouterr()
        assert "entity/project/artifact:v1" in captured.out


class TestScriptDetection:
    """Test _detect_script_command method."""

    def test_basic_script(self):
        """Test detection of basic Python script."""
        printer = ResumeCommandPrinter(['train.py', 'fit'])
        result = printer._detect_script_command()
        assert result == "python train.py"

    def test_script_with_python_prefix(self):
        """Test detection when python is already in argv."""
        printer = ResumeCommandPrinter(['python', 'train.py', 'fit'])
        result = printer._detect_script_command()
        assert result == "python"

    def test_module_main_detection(self):
        """Test detection of __main__.py module invocation."""
        printer = ResumeCommandPrinter(['/path/to/__main__.py', 'fit'])
        result = printer._detect_script_command()
        # Should use sys.argv[0] or fallback
        assert "python" in result

    def test_lightning_reflow_module(self):
        """Test detection of lightning_reflow module."""
        printer = ResumeCommandPrinter(
            ['/path/lightning_reflow/cli/__main__.py', 'fit']
        )
        result = printer._detect_script_command()
        assert "python" in result


class TestLegacyCommandBuilding:
    """Test _build_legacy_command method."""

    def test_basic_command(self):
        """Test building basic legacy command."""
        printer = ResumeCommandPrinter(['train.py', 'fit', '--lr=0.001'])
        result = printer._build_legacy_command()
        assert "python train.py fit --lr=0.001" == result

    def test_filters_ckpt_path_flag(self):
        """Test filtering of --ckpt_path flag and value."""
        printer = ResumeCommandPrinter(
            ['train.py', 'fit', '--ckpt_path', '/old/path.ckpt', '--lr=0.001']
        )
        result = printer._build_legacy_command()
        assert "--ckpt_path" not in result
        assert "/old/path.ckpt" not in result
        assert "--lr=0.001" in result

    def test_filters_ckpt_path_equals(self):
        """Test filtering of --ckpt_path=value format."""
        printer = ResumeCommandPrinter(
            ['train.py', 'fit', '--ckpt_path=/old/path.ckpt', '--lr=0.001']
        )
        result = printer._build_legacy_command()
        assert "--ckpt_path" not in result
        assert "--lr=0.001" in result

    def test_preserves_python_prefix(self):
        """Test that existing python prefix is preserved."""
        printer = ResumeCommandPrinter(['python', 'train.py', 'fit'])
        result = printer._build_legacy_command()
        assert result == "python train.py fit"


class TestUpdateOriginalArgv:
    """Test update_original_argv method."""

    def test_update_argv(self):
        """Test updating stored argv."""
        printer = ResumeCommandPrinter(['old.py', 'fit'])
        new_argv = ['new.py', 'resume', '--checkpoint-path', '/path']

        printer.update_original_argv(new_argv)

        assert printer._original_argv == new_argv


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
