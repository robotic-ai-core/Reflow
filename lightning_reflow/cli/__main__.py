#!/usr/bin/env python3
"""
Main entry point for lightning_reflow.cli package.

This allows the CLI to be invoked as:
    python -m lightning_reflow.cli

This approach prevents the RuntimeWarning about modules being found in sys.modules.
"""

from .lightning_cli import main

if __name__ == "__main__":
    main() 