"""Register classes that lightning_reflow checkpoints need for torch.load(weights_only=True).

PyTorch Lightning stores `pytorch-lightning_version` as a `TorchVersion`, and our
own checkpoint state may include numpy arrays/dtypes. With weights_only=True
(Lightning CLI's default for `_parse_ckpt_path`), torch.load rejects these unless
the classes are explicitly allowlisted.

Call `register_checkpoint_safe_globals()` once before loading a checkpoint —
it's idempotent.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_registered = False


def register_checkpoint_safe_globals() -> None:
    """Allowlist numpy and TorchVersion classes for torch.load(weights_only=True).

    Safe to call multiple times; subsequent calls are no-ops.
    """
    global _registered
    if _registered:
        return

    try:
        import numpy as np
        import torch

        safe_globals = [
            np._core.multiarray._reconstruct,
            np.ndarray,
            np.dtype,
        ]
        if hasattr(np, 'dtypes'):
            safe_globals.extend(
                getattr(np.dtypes, attr) for attr in dir(np.dtypes) if 'DType' in attr
            )

        if hasattr(torch, 'torch_version') and hasattr(torch.torch_version, 'TorchVersion'):
            safe_globals.append(torch.torch_version.TorchVersion)

        torch.serialization.add_safe_globals(safe_globals)
        logger.debug("Registered %d safe globals for checkpoint loading", len(safe_globals))
        _registered = True

    except Exception as e:
        logger.warning("Could not register safe globals: %s", e)
