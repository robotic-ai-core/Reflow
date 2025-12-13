"""
Canonical cleanup utilities for PyTorch Lightning DataLoaders.

This module provides the single source of truth for DataLoader worker cleanup,
preventing thread accumulation in long-running training sessions and HPO.

The cleanup functions here are used by:
- LightningReflow.fit() finally block (automatic)
- MemoryCleanupCallback (callback-based)
- LightningTune optimizer (HPO trials)
"""

import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)


def cleanup_dataloader_workers(
    trainer: Optional[Any] = None,
    datamodule: Optional[Any] = None,
    verbose: bool = False
) -> None:
    """
    Clean up PyTorch DataLoader workers to prevent thread accumulation.

    This is the canonical implementation used across all of LightningReflow
    and dependent libraries (LightningTune, callbacks, etc.).

    DataLoader workers (QueueFeederThread, _pin_memory_loop threads) can
    accumulate across training runs if not properly terminated. This function
    ensures they are cleaned up by:

    1. Calling datamodule.teardown('fit') to stop workers
    2. Deleting DataLoader iterators from trainer
    3. Explicitly freeing references

    This prevents thread leaks that can cause:
    - Memory growth
    - Thread exhaustion
    - System instability in long-running HPO

    Args:
        trainer: Lightning Trainer instance (optional)
            If provided, will clean up DataLoader iterators from:
            - train_dataloader
            - val_dataloaders
            - test_dataloaders
        datamodule: Lightning DataModule instance (optional)
            If provided, will call teardown('fit') to stop workers
        verbose: Whether to log cleanup actions (default: False)

    Example:
        >>> from lightning_reflow.utils.cleanup_utils import cleanup_dataloader_workers
        >>> # After training
        >>> cleanup_dataloader_workers(trainer=trainer, datamodule=datamodule)

    Note:
        This function is idempotent and safe to call multiple times.
        It will not raise exceptions - errors are logged as warnings.
    """
    if verbose:
        logger.info("[CleanupUtils] Starting DataLoader worker cleanup")

    # Step 1: Call teardown on datamodule to stop DataLoader workers
    if datamodule is not None:
        try:
            # Lightning 2.0+ teardown signature
            datamodule.teardown('fit')
            if verbose:
                logger.info("[CleanupUtils] DataModule teardown('fit') completed")
        except Exception as e:
            logger.warning(f"[CleanupUtils] DataModule teardown failed: {e}")

    # Step 2: Clean up DataLoader iterators from trainer
    if trainer is not None:
        try:
            # Train DataLoader
            if hasattr(trainer, 'train_dataloader') and trainer.train_dataloader is not None:
                if hasattr(trainer.train_dataloader, '_iterator'):
                    del trainer.train_dataloader._iterator
                    if verbose:
                        logger.info("[CleanupUtils] Cleaned train_dataloader iterator")

            # Validation DataLoaders (can be a list)
            if hasattr(trainer, 'val_dataloaders') and trainer.val_dataloaders is not None:
                val_dls = trainer.val_dataloaders if isinstance(trainer.val_dataloaders, list) else [trainer.val_dataloaders]
                for idx, val_dl in enumerate(val_dls):
                    if val_dl is not None and hasattr(val_dl, '_iterator'):
                        del val_dl._iterator
                        if verbose:
                            logger.info(f"[CleanupUtils] Cleaned val_dataloader[{idx}] iterator")

            # Test DataLoaders (can be a list)
            if hasattr(trainer, 'test_dataloaders') and trainer.test_dataloaders is not None:
                test_dls = trainer.test_dataloaders if isinstance(trainer.test_dataloaders, list) else [trainer.test_dataloaders]
                for idx, test_dl in enumerate(test_dls):
                    if test_dl is not None and hasattr(test_dl, '_iterator'):
                        del test_dl._iterator
                        if verbose:
                            logger.info(f"[CleanupUtils] Cleaned test_dataloader[{idx}] iterator")

            if verbose:
                logger.info("[CleanupUtils] Trainer DataLoader cleanup completed")

        except Exception as e:
            logger.warning(f"[CleanupUtils] Trainer DataLoader cleanup failed: {e}")

    # Step 3: Force deletion of references (helps garbage collector)
    try:
        del trainer, datamodule
    except:
        pass  # References may not exist or already be deleted

    if verbose:
        logger.info("[CleanupUtils] DataLoader worker cleanup finished")


def should_cleanup_dataloaders(trainer: Optional[Any] = None, datamodule: Optional[Any] = None) -> bool:
    """
    Determine if DataLoader cleanup is needed.

    Args:
        trainer: Lightning Trainer instance (optional)
        datamodule: Lightning DataModule instance (optional)

    Returns:
        True if cleanup should be performed, False otherwise
    """
    # If either trainer or datamodule is provided, cleanup is needed
    return trainer is not None or datamodule is not None


# Alias for backward compatibility with old cleanup_trial_resources calls
cleanup_trial_dataloader_resources = cleanup_dataloader_workers
