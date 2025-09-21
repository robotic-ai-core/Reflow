"""
Memory cleanup callback for PyTorch Lightning.

This callback provides configurable memory cleanup at any Lightning hook point
to help manage GPU memory fragmentation and reduce memory pressure.

Default behavior:
- Cleans up after validation epochs (most common need)
- Can be configured for any Lightning hook
- Optionally forces Python garbage collection
"""

import gc
from typing import List, Optional

import torch
from .base_reflow_callback import BaseReflowCallback


class MemoryCleanupCallback(BaseReflowCallback):
    """
    Callback that performs memory cleanup at configurable Lightning hook points.
    
    This helps manage GPU memory fragmentation and can reduce VRAM usage,
    especially during validation phases where memory patterns differ from training.
    
    Args:
        cleanup_points: List of Lightning hook names where cleanup should occur.
            Examples: ["on_validation_epoch_end", "on_train_epoch_end", "on_fit_end"]
            Default: ["on_validation_epoch_end"]
        force_gc: Whether to force Python garbage collection (default: True)
        cuda_empty_cache: Whether to empty CUDA cache (default: True)
        aggressive_cleanup: Whether to use aggressive CUDA cleanup (ipc_collect, clear cuBLAS)
            at on_fit_end. Helps with memory retention between training runs (default: True)
        verbose: Whether to log cleanup actions (default: False)
    
    Example:
        # Clean after validation and at end of training
        MemoryCleanupCallback(
            cleanup_points=["on_validation_epoch_end", "on_fit_end"],
            verbose=True
        )
    """
    
    def __init__(
        self,
        cleanup_points: Optional[List[str]] = None,
        force_gc: bool = True,
        cuda_empty_cache: bool = True,
        aggressive_cleanup: bool = True,
        verbose: bool = False
    ):
        super().__init__(enable_state_management=False, verbose=verbose)
        
        if cleanup_points is None:
            cleanup_points = ["on_validation_epoch_end"]  # Default to validation cleanup
            
        self.cleanup_points = set(cleanup_points)
        self.force_gc = force_gc
        self.cuda_empty_cache = cuda_empty_cache
        self.aggressive_cleanup = aggressive_cleanup
        self.verbose = verbose
        
        # Dynamically create cleanup methods for each specified hook
        self._setup_cleanup_hooks()
    
    def _setup_cleanup_hooks(self):
        """Dynamically create methods for each cleanup point."""
        for hook_name in self.cleanup_points:
            if not hook_name.startswith("on_"):
                raise ValueError(f"Invalid hook name: {hook_name}. Lightning hooks must start with 'on_'")
            
            # Create a closure to capture the hook name
            def make_cleanup_method(hook):
                def cleanup_method(self, trainer, pl_module, *args, **kwargs):
                    self._cleanup(hook, trainer)
                return cleanup_method
            
            # Set the method on this instance
            setattr(self, hook_name, make_cleanup_method(hook_name).__get__(self, self.__class__))
    
    def _cleanup(self, hook_name: str, trainer=None):
        """Perform memory cleanup operations."""
        # Skip redundant cleanup if pause callback is handling memory cleanup
        # (PauseCallback does its own explicit cleanup during pause)
        if trainer and hasattr(trainer, 'should_stop') and trainer.should_stop:
            if self.verbose:
                print(f"[MemoryCleanup] Skipping cleanup at {hook_name} - pause callback handling memory cleanup")
            return
            
        if self.verbose:
            print(f"[MemoryCleanup] Performing cleanup at {hook_name}")
            
        # CUDA cache cleanup
        if self.cuda_empty_cache and torch.cuda.is_available():
            if self.verbose:
                # Get memory stats before cleanup
                allocated_before = torch.cuda.memory_allocated() / 1024**3
                reserved_before = torch.cuda.memory_reserved() / 1024**3
                
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Additional aggressive cleanup for better memory reclamation
            if self.aggressive_cleanup and hook_name == "on_fit_end":
                if self.verbose:
                    print("[MemoryCleanup] Performing aggressive CUDA cleanup...")
                
                try:
                    torch.cuda.ipc_collect()  # Collects IPC memory handles
                    if self.verbose:
                        print("[MemoryCleanup] Collected IPC memory handles")
                except AttributeError:
                    pass  # Not available in older PyTorch versions
                
                try:
                    torch._C._cuda_clearCublasWorkspaces()  # Clear cuBLAS workspaces
                    if self.verbose:
                        print("[MemoryCleanup] Cleared cuBLAS workspaces")
                except AttributeError:
                    pass  # Not available in older PyTorch versions
            
            if self.verbose:
                allocated_after = torch.cuda.memory_allocated() / 1024**3
                reserved_after = torch.cuda.memory_reserved() / 1024**3
                print(f"[MemoryCleanup] CUDA memory - Allocated: {allocated_before:.2f}GB -> {allocated_after:.2f}GB, "
                      f"Reserved: {reserved_before:.2f}GB -> {reserved_after:.2f}GB")
        
        # Python garbage collection
        if self.force_gc:
            collected = gc.collect()
            if self.verbose:
                print(f"[MemoryCleanup] Garbage collected {collected} objects")