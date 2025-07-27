#!/usr/bin/env python3
"""
Main entry point for lightning_reflow package.

This allows the CLI to be invoked as:
    python -m lightning_reflow

Delegates to the CLI module.
"""

from .cli.lightning_cli import main

if __name__ == "__main__":
    main() 