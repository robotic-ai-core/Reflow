from .lightning_cli import LightningReflowCLI

__all__ = ["LightningReflowCLI"]

def main():
    """Main entry point for the lightning-reflow console script."""
    cli = LightningReflowCLI()

if __name__ == "__main__":
    main() 