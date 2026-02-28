"""Training profiler callback using torch.profiler.

Wraps torch.profiler.profile as a Lightning callback so profiling
can be added to any training run via YAML config or CLI override,
without modifying training scripts.

Usage (YAML):
    callbacks:
      - class_path: lightning_reflow.callbacks.TrainingProfilerCallback
        init_args:
          active_steps: 10
          warmup_steps: 3

Usage (CLI override):
    python train.py fit --config my.yaml \
        --trainer.max_epochs 1 --trainer.limit_train_batches 20 \
        --trainer.callbacks+=lightning_reflow.callbacks.TrainingProfilerCallback \
        --trainer.callbacks.init_args.active_steps=10
"""

import logging
import os
import time

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback

logger = logging.getLogger(__name__)


def _std(values: list[float]) -> float:
    """Sample standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return variance**0.5


class TrainingProfilerCallback(Callback):
    """Lightning callback for torch.profiler-based training profiling.

    Profiles a configurable window of training steps, then prints
    summary tables and exports a Chrome trace for visualization.

    The profiler uses explicit start()/stop() tied to Lightning hooks,
    so it works with any model configuration (backbone, encoder, etc.)
    without needing to instantiate components manually.

    Args:
        active_steps: Number of steps to actively profile.
        warmup_steps: Steps for GPU warmup before active profiling.
        wait_steps: Steps to skip entirely before warmup begins.
        trace_dir: Directory for Chrome trace output.
        record_shapes: Record tensor shapes in profiler events.
        profile_memory: Track CUDA memory allocations.
        with_stack: Record Python call stacks (larger traces).
        with_flops: Estimate FLOPs per operator.
        row_limit: Number of rows in summary tables.
    """

    def __init__(
        self,
        active_steps: int = 10,
        warmup_steps: int = 3,
        wait_steps: int = 2,
        trace_dir: str = "tmp",
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = False,
        with_flops: bool = False,
        row_limit: int = 30,
    ):
        super().__init__()
        self.active_steps = active_steps
        self.warmup_steps = warmup_steps
        self.wait_steps = wait_steps
        self.trace_dir = trace_dir
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_flops = with_flops
        self.row_limit = row_limit

        self._prof: torch.profiler.profile | None = None
        self._started = False
        self._finished = False
        self._step_times: list[float] = []
        self._step_t0: float | None = None

    @property
    def _total_steps(self) -> int:
        return self.wait_steps + self.warmup_steps + self.active_steps

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch,
        batch_idx: int,
    ) -> None:
        if self._finished:
            return

        # Start profiler on the first batch
        if not self._started:
            self._start_profiler()

        # Record wall-clock time per step
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._step_t0 = time.perf_counter()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        if self._finished or self._prof is None:
            return

        # Record step wall-clock time
        if self._step_t0 is not None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - self._step_t0) * 1000
            self._step_times.append(elapsed_ms)
            self._step_t0 = None

        step_count = len(self._step_times)
        phase = self._step_phase(step_count - 1)
        logger.info(
            f"Profiler step {step_count}/{self._total_steps} [{phase}] | "
            f"time={self._step_times[-1]:.1f}ms"
        )

        self._prof.step()

        # Stop after all scheduled steps
        if step_count >= self._total_steps:
            self._stop_and_report()

    def on_fit_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._cleanup()

    def on_exception(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        exception: BaseException,
    ) -> None:
        self._cleanup()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _start_profiler(self) -> None:
        """Create and start the torch.profiler.profile context."""
        schedule = torch.profiler.schedule(
            wait=self.wait_steps,
            warmup=self.warmup_steps,
            active=self.active_steps,
            repeat=1,
        )

        self._prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=schedule,
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
            with_flops=self.with_flops,
        )
        self._prof.__enter__()
        self._started = True

        logger.info(
            f"Profiler started: wait={self.wait_steps}, "
            f"warmup={self.warmup_steps}, active={self.active_steps}"
        )

    def _stop_and_report(self) -> None:
        """Stop profiler, export trace, and print summary tables."""
        if self._prof is None:
            return

        self._prof.__exit__(None, None, None)
        self._finished = True

        # Export Chrome trace
        os.makedirs(self.trace_dir, exist_ok=True)
        trace_path = os.path.join(self.trace_dir, "profile_trace.json")
        self._prof.export_chrome_trace(trace_path)
        logger.info(f"Chrome trace saved to {trace_path}")

        self._print_summary()
        self._prof = None

    def _cleanup(self) -> None:
        """Ensure profiler is stopped if still active."""
        if self._prof is not None and not self._finished:
            self._prof.__exit__(None, None, None)
            self._finished = True
            self._prof = None

    def _step_phase(self, step_idx: int) -> str:
        if step_idx < self.wait_steps:
            return "wait"
        elif step_idx < self.wait_steps + self.warmup_steps:
            return "warmup"
        else:
            return "active"

    def _print_summary(self) -> None:
        """Print profiling summary tables to stdout."""
        if self._prof is None:
            return

        sep = "=" * 80

        # 1. CUDA time summary
        print(f"\n{sep}")
        print("PROFILER SUMMARY -- Sorted by CUDA Time Total")
        print(sep)
        print(
            self._prof.key_averages().table(
                sort_by="cuda_time_total",
                row_limit=self.row_limit,
            )
        )

        # 2. CPU time summary
        print(f"\n{sep}")
        print("PROFILER SUMMARY -- Sorted by CPU Time Total")
        print(sep)
        print(
            self._prof.key_averages().table(
                sort_by="cpu_time_total",
                row_limit=min(20, self.row_limit),
            )
        )

        # 3. Memory summary
        print(f"\n{sep}")
        print("PROFILER SUMMARY -- Sorted by CUDA Memory Usage")
        print(sep)
        print(
            self._prof.key_averages().table(
                sort_by="self_cuda_memory_usage",
                row_limit=min(20, self.row_limit),
            )
        )

        # 4. Step timing summary
        skip = self.wait_steps + self.warmup_steps
        active_times = self._step_times[skip:]
        if active_times:
            avg_ms = sum(active_times) / len(active_times)
            print(f"\n{sep}")
            print("STEP TIMING SUMMARY (active steps only)")
            print(sep)
            print(f"  Steps profiled : {len(active_times)}")
            print(f"  Avg step time  : {avg_ms:.1f} ms")
            print(f"  Min step time  : {min(active_times):.1f} ms")
            print(f"  Max step time  : {max(active_times):.1f} ms")
            print(f"  Std step time  : {_std(active_times):.1f} ms")

        # 5. GPU memory summary
        if torch.cuda.is_available():
            print(f"\n{sep}")
            print("GPU MEMORY SUMMARY")
            print(sep)
            print(
                f"  Peak allocated : "
                f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB"
            )
            print(
                f"  Peak reserved  : "
                f"{torch.cuda.max_memory_reserved() / 1e9:.2f} GB"
            )
            print(
                f"  Current alloc  : "
                f"{torch.cuda.memory_allocated() / 1e9:.2f} GB"
            )
            print(sep)
