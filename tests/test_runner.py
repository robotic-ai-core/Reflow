"""
Test runner for Lightning Reflow test suite.

Provides utilities to run specific test categories and comprehensive test coverage.
"""

import pytest
import sys
from pathlib import Path

# Add lightning_reflow to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_all_tests():
    """Run all Lightning Reflow tests."""
    test_dir = Path(__file__).parent
    return pytest.main([str(test_dir), "-v", "--tb=short"])


def run_unit_tests():
    """Run only unit tests."""
    test_dir = Path(__file__).parent / "unit"
    return pytest.main([str(test_dir), "-v", "--tb=short"])


def run_integration_tests():
    """Run only integration tests."""
    test_dir = Path(__file__).parent / "integration"
    return pytest.main([str(test_dir), "-v", "--tb=short"])


def run_model_tests():
    """Run model-specific tests."""
    test_files = [
        "unit/test_simple_model.py",
        "unit/test_simple_data.py"
    ]
    test_dir = Path(__file__).parent
    test_paths = [str(test_dir / f) for f in test_files]
    return pytest.main(test_paths + ["-v", "--tb=short"])


def run_cli_tests():
    """Run CLI-specific tests."""
    test_files = [
        "integration/test_cli_integration.py",
        "integration/test_wandb_resume_integration.py"
    ]
    test_dir = Path(__file__).parent
    test_paths = [str(test_dir / f) for f in test_files]
    return pytest.main(test_paths + ["-v", "--tb=short"])


def run_callback_tests():
    """Run callback-specific tests.""" 
    test_dir = Path(__file__).parent / "unit" / "callbacks"
    return pytest.main([str(test_dir), "-v", "--tb=short"])


def run_with_coverage():
    """Run tests with coverage reporting."""
    test_dir = Path(__file__).parent
    return pytest.main([
        str(test_dir),
        "--cov=lightning_reflow",
        "--cov-report=html",
        "--cov-report=term-missing",
        "-v"
    ])


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Lightning Reflow Test Runner")
    parser.add_argument("--type", choices=[
        "all", "unit", "integration", "model", "cli", "callback", "coverage"
    ], default="all", help="Type of tests to run")
    
    args = parser.parse_args()
    
    if args.type == "all":
        exit_code = run_all_tests()
    elif args.type == "unit":
        exit_code = run_unit_tests()
    elif args.type == "integration":
        exit_code = run_integration_tests()
    elif args.type == "model":
        exit_code = run_model_tests()
    elif args.type == "cli":
        exit_code = run_cli_tests()
    elif args.type == "callback":
        exit_code = run_callback_tests()
    elif args.type == "coverage":
        exit_code = run_with_coverage()
    else:
        exit_code = run_all_tests()
    
    sys.exit(exit_code)