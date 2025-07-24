"""
Comprehensive integration tests for W&B resume functionality in Lightning Reflow.

Self-contained test suite that covers:
1. Resume subcommand execution flow
2. Argument linking between --wandb_run and trainer.logger.init_args
3. Command construction and subprocess execution
4. W&B run ID extraction and propagation
5. Config override handling in resume context

These tests use the minimal Lightning Reflow pipeline for testing.
"""

import pytest
import tempfile
import yaml
import torch
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from argparse import Namespace

from lightning_reflow.cli.lightning_cli import LightningReflowCLI
from lightning_reflow.utils.wandb.resume_command_handler import ResumeCommandHandler


class TestResumeSubcommandExecution:
    """Test the complete resume subcommand execution flow."""
    
    def test_resume_subcommand_with_checkpoint_artifact_flow(self, temp_dir, mock_checkpoint):
        """Test complete flow: resume subcommand -> artifact download -> W&B ID extraction -> execution."""
        artifact_reference = "entity/project/test-run-123-pause:latest"
        
        with patch('lightning_reflow.cli.lightning_cli.LightningReflowCLI._download_wandb_checkpoint') as mock_download:
            with patch('lightning_reflow.cli.lightning_cli.LightningReflowCLI._extract_wandb_run_id_from_path') as mock_extract_id:
                with patch('lightning_reflow.cli.lightning_cli.LightningReflowCLI._execute_resume_from_checkpoint') as mock_execute:
                    
                    # Setup mocks
                    mock_download.return_value = (mock_checkpoint, None)
                    mock_extract_id.return_value = "test-run-123"
                    mock_execute.return_value = True
                    
                    # Create CLI instance
                    cli = LightningReflowCLI.__new__(LightningReflowCLI)
                    
                    # Test the complete artifact resume flow
                    result = cli._execute_resume_from_artifact(
                        artifact_reference,
                        wandb_run_override=None,
                        config_overrides=None,
                        dry_run=False,
                        extra_cli_args=None
                    )
                    
                    # Verify the flow
                    assert result is True
                    mock_download.assert_called_once_with(artifact_reference, use_wandb_config=True)
                    mock_extract_id.assert_called_once_with(mock_checkpoint)
                    mock_execute.assert_called_once_with(
                        mock_checkpoint,
                        config_overrides=None,
                        wandb_run_override="test-run-123",
                        dry_run=False,
                        extra_cli_args=None
                    )
    
    def test_resume_subcommand_with_config_overrides(self, temp_dir, mock_checkpoint, config_file):
        """Test resume with both artifact config and override configs."""
        artifact_reference = "entity/project/test-run-123-pause:latest"
        artifact_config = temp_dir / "artifact_config.yaml"
        override_config = Path(config_file)
        
        # Create artifact config
        with open(artifact_config, 'w') as f:
            yaml.dump({'trainer': {'max_epochs': 5}}, f)
        
        with patch('lightning_reflow.cli.lightning_cli.LightningReflowCLI._download_wandb_checkpoint') as mock_download:
            with patch('lightning_reflow.cli.lightning_cli.LightningReflowCLI._extract_wandb_run_id_from_path') as mock_extract_id:
                with patch('lightning_reflow.cli.lightning_cli.LightningReflowCLI._execute_resume_from_checkpoint') as mock_execute:
                    
                    # Setup mocks - artifact has config
                    mock_download.return_value = (mock_checkpoint, str(artifact_config))
                    mock_extract_id.return_value = "test-run-123"
                    mock_execute.return_value = True
                    
                    cli = LightningReflowCLI.__new__(LightningReflowCLI)
                    
                    # Test with config overrides
                    result = cli._execute_resume_from_artifact(
                        artifact_reference,
                        config_overrides=[override_config],
                        dry_run=False
                    )
                    
                    # Verify config merging
                    assert result is True
                    mock_execute.assert_called_once()
                    
                    # Check that all configs are passed
                    call_args = mock_execute.call_args[1]
                    config_overrides = call_args['config_overrides']
                    assert len(config_overrides) == 2
                    assert str(artifact_config) in [str(c) for c in config_overrides]
                    assert str(override_config) in [str(c) for c in config_overrides]


class TestArgumentLinking:
    """Test CLI argument linking functionality."""
    
    def test_wandb_run_argument_linking_to_logger_id(self):
        """Test that --wandb_run is properly linked to trainer.logger.init_args.id."""
        with patch('lightning.pytorch.cli.LightningArgumentParser') as mock_parser_class:
            mock_parser = Mock()
            mock_parser_class.return_value = mock_parser
            mock_parser.link_arguments = Mock()
            
            # Create CLI instance and call add_arguments_to_parser
            cli = LightningReflowCLI.__new__(LightningReflowCLI)
            cli.add_arguments_to_parser(mock_parser)
            
            # Verify wandb_run is linked to logger id
            link_calls = mock_parser.link_arguments.call_args_list
            
            # Check for the wandb_run -> trainer.logger.init_args.id link
            id_link_found = False
            resume_link_found = False
            
            for call_args in link_calls:
                args = call_args[0] if call_args[0] else []
                if len(args) >= 2:
                    if args[0] == "wandb_run" and args[1] == "trainer.logger.init_args.id":
                        id_link_found = True
                    elif args[0] == "wandb_run" and args[1] == "trainer.logger.init_args.resume":
                        resume_link_found = True
            
            assert id_link_found, "wandb_run should be linked to trainer.logger.init_args.id"
            assert resume_link_found, "wandb_run should be linked to trainer.logger.init_args.resume"
    
    def test_wandb_run_resume_compute_function(self):
        """Test that the resume compute function returns 'allow' when wandb_run is provided."""
        with patch('lightning.pytorch.cli.LightningArgumentParser') as mock_parser_class:
            mock_parser = Mock()
            mock_parser_class.return_value = mock_parser
            
            # Capture the compute functions
            link_calls = []
            def capture_link_args(*args, **kwargs):
                link_calls.append((args, kwargs))
            
            mock_parser.link_arguments = capture_link_args
            
            cli = LightningReflowCLI.__new__(LightningReflowCLI)
            cli.add_arguments_to_parser(mock_parser)
            
            # Find the resume compute function
            resume_compute_fn = None
            for call_args, call_kwargs in link_calls:
                if (len(call_args) >= 2 and 
                    call_args[0] == "wandb_run" and 
                    call_args[1] == "trainer.logger.init_args.resume"):
                    resume_compute_fn = call_kwargs.get('compute_fn')
                    break
            
            assert resume_compute_fn is not None, "Resume compute function should be defined"
            
            # Test the compute function
            assert resume_compute_fn("test-run-id") == "allow"
            assert resume_compute_fn(None) is None
            assert resume_compute_fn("") is None


class TestCommandConstruction:
    """Test resume command construction and subprocess execution."""
    
    def test_resume_command_construction_with_wandb_run(self, mock_checkpoint, config_file):
        """Test that resume command is constructed correctly with --wandb_run argument."""
        with patch('subprocess.run') as mock_subprocess:
            mock_subprocess.return_value.returncode = 0
            
            cli = LightningReflowCLI.__new__(LightningReflowCLI)
            
            # Test command construction
            result = cli._execute_resume_from_checkpoint(
                checkpoint_path=mock_checkpoint,
                config_overrides=[Path(config_file)],
                wandb_run_override="test-run-123",
                dry_run=False
            )
            
            assert result is True
            mock_subprocess.assert_called_once()
            
            # Check the command that was executed
            executed_cmd = mock_subprocess.call_args[0][0]
            
            # Verify key components
            assert "python" in executed_cmd
            assert "train_lightning.py" in executed_cmd
            assert "fit" in executed_cmd
            assert "--ckpt_path" in executed_cmd
            assert mock_checkpoint in executed_cmd
            assert "--wandb_run" in executed_cmd
            assert "test-run-123" in executed_cmd
            assert "--config" in executed_cmd
            assert config_file in executed_cmd
    
    def test_resume_command_dry_run_output(self, mock_checkpoint, config_file):
        """Test that dry run shows the correct command."""
        cli = LightningReflowCLI.__new__(LightningReflowCLI)
        
        with patch('builtins.print') as mock_print:
            # Provide config overrides to avoid "no config found" error
            result = cli._execute_resume_from_checkpoint(
                checkpoint_path=mock_checkpoint,
                config_overrides=[Path(config_file)],
                wandb_run_override="test-run-123",
                dry_run=True
            )
            
            assert result is True
            
            # Check that debug information was printed
            print_calls = [call[0][0] for call in mock_print.call_args_list]
            dry_run_output = [msg for msg in print_calls if "[DRY RUN]" in msg]
            
            assert len(dry_run_output) > 0, "Dry run should print command"
            assert "test-run-123" in dry_run_output[0], "Dry run should show wandb_run argument"
    
    def test_resume_command_without_wandb_run(self, mock_checkpoint, config_file):
        """Test resume command construction without W&B run ID."""
        with patch('subprocess.run') as mock_subprocess:
            mock_subprocess.return_value.returncode = 0
            
            cli = LightningReflowCLI.__new__(LightningReflowCLI)
            
            # Provide config overrides to avoid "no config found" error
            result = cli._execute_resume_from_checkpoint(
                checkpoint_path=mock_checkpoint,
                config_overrides=[Path(config_file)],
                wandb_run_override=None,
                dry_run=False
            )
            
            assert result is True
            executed_cmd = mock_subprocess.call_args[0][0]
            
            # Should not contain wandb_run argument
            assert "--wandb_run" not in executed_cmd


class TestWandbRunContinuity:
    """Test W&B run continuity and ID extraction."""
    
    def test_wandb_run_id_extraction_from_checkpoint(self, mock_checkpoint):
        """Test extraction of W&B run ID from checkpoint metadata."""
        cli = LightningReflowCLI.__new__(LightningReflowCLI)
        
        # Test extraction
        run_id = cli._extract_wandb_run_id_from_path(mock_checkpoint)
        
        assert run_id == "test-run-123", f"Expected 'test-run-123', got '{run_id}'"
    
    def test_wandb_run_id_auto_detection_in_artifact_resume(self, temp_dir, mock_checkpoint):
        """Test that W&B run ID is auto-detected when not explicitly provided."""
        artifact_reference = "entity/project/test-run-123-pause:latest"
        
        with patch('lightning_reflow.cli.lightning_cli.LightningReflowCLI._download_wandb_checkpoint') as mock_download:
            with patch('lightning_reflow.cli.lightning_cli.LightningReflowCLI._execute_resume_from_checkpoint') as mock_execute:
                with patch('builtins.print') as mock_print:
                    
                    mock_download.return_value = (mock_checkpoint, None)
                    mock_execute.return_value = True
                    
                    cli = LightningReflowCLI.__new__(LightningReflowCLI)
                    
                    # Test without explicit wandb_run_override
                    result = cli._execute_resume_from_artifact(artifact_reference)
                    
                    assert result is True
                    
                    # Verify auto-detection message was printed
                    print_calls = [call[0][0] for call in mock_print.call_args_list]
                    auto_detect_msgs = [msg for msg in print_calls if "Auto-detected W&B run ID" in msg]
                    assert len(auto_detect_msgs) > 0, "Should print auto-detection message"
                    assert "test-run-123" in auto_detect_msgs[0]
                    
                    # Verify the extracted run ID was passed to resume
                    mock_execute.assert_called_once()
                    call_kwargs = mock_execute.call_args[1]
                    assert call_kwargs['wandb_run_override'] == "test-run-123"
    
    def test_wandb_run_id_explicit_override(self, temp_dir, mock_checkpoint):
        """Test that explicit wandb_run_override takes precedence over extracted ID."""
        artifact_reference = "entity/project/test-run-123-pause:latest"
        explicit_run_id = "explicit-run-456"
        
        with patch('lightning_reflow.cli.lightning_cli.LightningReflowCLI._download_wandb_checkpoint') as mock_download:
            with patch('lightning_reflow.cli.lightning_cli.LightningReflowCLI._execute_resume_from_checkpoint') as mock_execute:
                
                mock_download.return_value = (mock_checkpoint, None)
                mock_execute.return_value = True
                
                cli = LightningReflowCLI.__new__(LightningReflowCLI)
                
                # Test with explicit override
                result = cli._execute_resume_from_artifact(
                    artifact_reference,
                    wandb_run_override=explicit_run_id
                )
                
                assert result is True
                
                # Verify explicit run ID was used
                mock_execute.assert_called_once()
                call_kwargs = mock_execute.call_args[1]
                assert call_kwargs['wandb_run_override'] == explicit_run_id


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""
    
    def test_missing_checkpoint_file_error(self):
        """Test error handling when checkpoint file doesn't exist."""
        cli = LightningReflowCLI.__new__(LightningReflowCLI)
        
        result = cli._execute_resume_from_checkpoint(
            checkpoint_path="/nonexistent/path.ckpt",
            dry_run=False
        )
        
        assert result is False
    
    def test_artifact_download_failure(self, temp_dir):
        """Test handling of artifact download failures."""
        artifact_reference = "entity/project/nonexistent:latest"
        
        with patch('lightning_reflow.cli.lightning_cli.LightningReflowCLI._download_wandb_checkpoint') as mock_download:
            mock_download.side_effect = Exception("Artifact not found")
            
            cli = LightningReflowCLI.__new__(LightningReflowCLI)
            
            result = cli._execute_resume_from_artifact(artifact_reference)
            
            assert result is False
    
    def test_wandb_run_id_extraction_failure(self, temp_dir):
        """Test handling when W&B run ID cannot be extracted."""
        # Create a checkpoint without W&B metadata
        checkpoint_path = temp_dir / "no_wandb_checkpoint.ckpt"
        checkpoint_data = {
            'state_dict': {},
            'epoch': 5,
            'global_step': 1000
            # No wandb_run_id or pause_callback_metadata
        }
        torch.save(checkpoint_data, checkpoint_path)
        
        artifact_reference = "entity/project/unknown:latest"
        
        with patch('lightning_reflow.cli.lightning_cli.LightningReflowCLI._download_wandb_checkpoint') as mock_download:
            with patch('lightning_reflow.cli.lightning_cli.LightningReflowCLI._execute_resume_from_checkpoint') as mock_execute:
                with patch('builtins.print') as mock_print:
                    
                    mock_download.return_value = (str(checkpoint_path), None)
                    mock_execute.return_value = True
                    
                    cli = LightningReflowCLI.__new__(LightningReflowCLI)
                    
                    result = cli._execute_resume_from_artifact(artifact_reference)
                    
                    assert result is True  # Should still work, just create new run
                    
                    # Should print warning about not extracting run ID
                    print_calls = [call[0][0] for call in mock_print.call_args_list]
                    warning_msgs = [msg for msg in print_calls if "Could not extract W&B run ID" in msg]
                    assert len(warning_msgs) > 0
                    
                    # Should call resume with None wandb_run_override
                    mock_execute.assert_called_once()
                    call_kwargs = mock_execute.call_args[1]
                    assert call_kwargs['wandb_run_override'] is None


if __name__ == "__main__":
    pytest.main([__file__])