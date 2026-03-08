"""
Unit tests for resume fallback sys.path preservation (Fix C).

When LightningReflow's resume CLI can't find a checkpoint, it falls back to
running fit as a subprocess. These tests verify that:

1. The subprocess uses the original invoking script (sys.argv[0]) when no
   checkpoint metadata is available, preserving sys.path modifications.
2. The subprocess inherits the current process's sys.path via PYTHONPATH,
   ensuring custom path entries (e.g. project_root) are available.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, call

import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from lightning_reflow.core import LightningReflow


class TestResumeFallbackUsesInvokingScript:
    """Test that the fallback subprocess uses sys.argv[0] when available."""

    @patch("subprocess.run")
    @patch("sys.exit")
    def test_fallback_uses_sys_argv0_when_valid_py_file(
        self, mock_exit, mock_subprocess, tmp_path
    ):
        """When sys.argv[0] is a valid .py file the fallback should use it."""
        # Create a dummy train.py so Path.exists() returns True
        train_script = tmp_path / "train.py"
        train_script.write_text("# dummy training script\n")

        mock_subprocess.return_value.returncode = 0

        reflow = LightningReflow(auto_configure_logging=False)

        with patch("sys.argv", [str(train_script), "resume", "--checkpoint-path", "x"]):
            reflow._execute_fit_subprocess(
                checkpoint_path=None,
                embedded_config_yaml=None,
                config_overrides=None,
                wandb_run_id=None,
                extra_cli_args=None,
            )

        # Verify subprocess was invoked with the invoking script
        mock_subprocess.assert_called_once()
        cmd = mock_subprocess.call_args[0][0]
        assert cmd[0] == sys.executable
        assert cmd[1] == str(train_script)
        assert cmd[2] == "fit"

    @patch("subprocess.run")
    @patch("sys.exit")
    def test_fallback_uses_generic_cli_when_argv0_not_py(
        self, mock_exit, mock_subprocess
    ):
        """When sys.argv[0] is not a .py file, fall back to generic CLI."""
        mock_subprocess.return_value.returncode = 0

        reflow = LightningReflow(auto_configure_logging=False)

        # Simulate running via python -m (argv[0] has no .py extension)
        with patch("sys.argv", ["lightning_reflow", "resume"]):
            reflow._execute_fit_subprocess(
                checkpoint_path=None,
                embedded_config_yaml=None,
            )

        cmd = mock_subprocess.call_args[0][0]
        assert cmd[0] == sys.executable
        assert cmd[1] == "-m"
        assert cmd[2] == "lightning_reflow.cli"
        assert cmd[3] == "fit"

    @patch("subprocess.run")
    @patch("sys.exit")
    def test_fallback_uses_generic_cli_when_argv0_does_not_exist(
        self, mock_exit, mock_subprocess
    ):
        """When sys.argv[0] points to a nonexistent .py file, use generic CLI."""
        mock_subprocess.return_value.returncode = 0

        reflow = LightningReflow(auto_configure_logging=False)

        with patch("sys.argv", ["/nonexistent/train.py", "resume"]):
            reflow._execute_fit_subprocess(
                checkpoint_path=None,
                embedded_config_yaml=None,
            )

        cmd = mock_subprocess.call_args[0][0]
        assert cmd[1] == "-m"
        assert cmd[2] == "lightning_reflow.cli"

    @patch("subprocess.run")
    @patch("sys.exit")
    def test_checkpoint_original_cmd_takes_priority_over_argv0(
        self, mock_exit, mock_subprocess, tmp_path
    ):
        """When checkpoint has original_command, it takes priority over sys.argv[0]."""
        # Create checkpoint with original_command metadata
        original_script = tmp_path / "original_train.py"
        original_script.write_text("# original\n")

        checkpoint_path = tmp_path / "checkpoint.ckpt"
        torch.save(
            {
                "state_dict": {},
                "pause_callback_metadata": {
                    "original_command": [str(original_script), "fit", "--config", "x.yaml"]
                },
            },
            checkpoint_path,
        )

        mock_subprocess.return_value.returncode = 0

        reflow = LightningReflow(auto_configure_logging=False)

        # Even with a different sys.argv[0], the checkpoint's original_command wins
        fallback_script = tmp_path / "other_train.py"
        fallback_script.write_text("# other\n")

        with patch("sys.argv", [str(fallback_script), "resume"]):
            reflow._execute_fit_subprocess(
                checkpoint_path=checkpoint_path,
                embedded_config_yaml=None,
            )

        cmd = mock_subprocess.call_args[0][0]
        assert cmd[1] == str(original_script)


class TestResumeFallbackPropagatesPythonPath:
    """Test that PYTHONPATH is propagated to the subprocess."""

    @patch("subprocess.run")
    @patch("sys.exit")
    def test_subprocess_receives_current_sys_path_in_pythonpath(
        self, mock_exit, mock_subprocess
    ):
        """subprocess.run should be called with env containing current sys.path."""
        mock_subprocess.return_value.returncode = 0

        reflow = LightningReflow(auto_configure_logging=False)

        with patch("sys.argv", ["lightning_reflow", "resume"]):
            reflow._execute_fit_subprocess(
                checkpoint_path=None,
                embedded_config_yaml=None,
            )

        # Extract the env kwarg
        call_kwargs = mock_subprocess.call_args[1]
        assert "env" in call_kwargs
        env = call_kwargs["env"]
        assert "PYTHONPATH" in env

        # Every non-empty entry in sys.path should be in PYTHONPATH
        pythonpath_entries = env["PYTHONPATH"].split(os.pathsep)
        for p in sys.path:
            if p:
                assert p in pythonpath_entries, f"{p} not found in PYTHONPATH"

    @patch("subprocess.run")
    @patch("sys.exit")
    def test_existing_pythonpath_is_preserved(
        self, mock_exit, mock_subprocess
    ):
        """Pre-existing PYTHONPATH entries must not be lost."""
        mock_subprocess.return_value.returncode = 0

        reflow = LightningReflow(auto_configure_logging=False)

        sentinel = "/some/unique/sentinel/path"
        env_with_sentinel = os.environ.copy()
        env_with_sentinel["PYTHONPATH"] = sentinel

        with patch("sys.argv", ["lightning_reflow", "resume"]):
            with patch.dict(os.environ, {"PYTHONPATH": sentinel}):
                reflow._execute_fit_subprocess(
                    checkpoint_path=None,
                    embedded_config_yaml=None,
                )

        env = mock_subprocess.call_args[1]["env"]
        assert sentinel in env["PYTHONPATH"]

    @patch("subprocess.run")
    @patch("sys.exit")
    def test_custom_sys_path_entry_propagated(
        self, mock_exit, mock_subprocess
    ):
        """A custom sys.path entry (e.g. project_root) must appear in PYTHONPATH."""
        mock_subprocess.return_value.returncode = 0

        reflow = LightningReflow(auto_configure_logging=False)

        custom_path = "/my/custom/project_root"
        original_sys_path = sys.path.copy()

        try:
            sys.path.insert(0, custom_path)

            with patch("sys.argv", ["lightning_reflow", "resume"]):
                reflow._execute_fit_subprocess(
                    checkpoint_path=None,
                    embedded_config_yaml=None,
                )

            env = mock_subprocess.call_args[1]["env"]
            assert custom_path in env["PYTHONPATH"].split(os.pathsep)
        finally:
            sys.path = original_sys_path

    @patch("subprocess.run")
    @patch("sys.exit")
    def test_pythonpath_propagated_with_checkpoint(
        self, mock_exit, mock_subprocess, tmp_path
    ):
        """PYTHONPATH propagation works for the normal resume path (with checkpoint) too."""
        # Create a checkpoint without original_command so it falls through
        checkpoint_path = tmp_path / "checkpoint.ckpt"
        torch.save({"state_dict": {}}, checkpoint_path)

        mock_subprocess.return_value.returncode = 0

        reflow = LightningReflow(auto_configure_logging=False)

        with patch("sys.argv", ["lightning_reflow", "resume"]):
            reflow._execute_fit_subprocess(
                checkpoint_path=checkpoint_path,
                embedded_config_yaml=None,
            )

        call_kwargs = mock_subprocess.call_args[1]
        assert "env" in call_kwargs
        assert "PYTHONPATH" in call_kwargs["env"]

    @patch("subprocess.run")
    @patch("sys.exit")
    def test_empty_pythonpath_when_no_existing(
        self, mock_exit, mock_subprocess
    ):
        """When no PYTHONPATH exists, only sys.path entries are set (no trailing separator)."""
        mock_subprocess.return_value.returncode = 0

        reflow = LightningReflow(auto_configure_logging=False)

        env_without_pythonpath = os.environ.copy()
        env_without_pythonpath.pop("PYTHONPATH", None)

        with patch("sys.argv", ["lightning_reflow", "resume"]):
            with patch.dict(os.environ, {}, clear=True):
                # Restore essential env vars
                with patch.dict(os.environ, {
                    k: v for k, v in os.environ.items()
                    if k != "PYTHONPATH"
                }):
                    reflow._execute_fit_subprocess(
                        checkpoint_path=None,
                        embedded_config_yaml=None,
                    )

        env = mock_subprocess.call_args[1]["env"]
        pythonpath = env["PYTHONPATH"]
        # Should not end with a pathsep (no trailing empty entry)
        assert not pythonpath.endswith(os.pathsep)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
