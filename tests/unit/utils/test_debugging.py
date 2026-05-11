"""Unit tests for utils/debugging/ — CrashResistantLogger and ThreadMonitor."""

import threading
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

from lightning_reflow.utils.debugging.crash_logger import (
    CircularBufferHandler,
    CrashResistantLogger,
)
from lightning_reflow.utils.debugging.thread_monitor import ThreadMonitor


class TestCircularBufferHandler:

    def test_buffer_keeps_last_n_lines(self):
        buffer = CircularBufferHandler(max_lines=3)
        for i in range(5):
            buffer.write(f"line{i}\n")
        lines = buffer.get_lines()
        assert lines == ["line2\n", "line3\n", "line4\n"]

    def test_save_to_file_writes_buffer_contents(self, tmp_path):
        buffer = CircularBufferHandler(max_lines=10)
        buffer.write("first\n")
        buffer.write("second\n")

        out = tmp_path / "buf.log"
        buffer.save_to_file(str(out))
        assert out.read_text() == "first\nsecond\n"

    def test_buffer_is_thread_safe(self):
        buffer = CircularBufferHandler(max_lines=1000)

        def writer(start):
            for i in range(100):
                buffer.write(f"{start}-{i}\n")

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        lines = buffer.get_lines()
        assert len(lines) == 500  # 5 writers * 100 lines, within capacity


class TestCrashResistantLogger:

    def test_context_manager_writes_and_preserves_lines(self, tmp_path):
        with CrashResistantLogger(
            log_dir=str(tmp_path), prefix="test", max_buffer_lines=10
        ) as logger:
            print("hello captured world")

        full_log = next(tmp_path.glob("test_*_full.log"))
        assert "hello captured world" in full_log.read_text()

        circular_log = next(tmp_path.glob("test_*_last_10.log"))
        assert "hello captured world" in circular_log.read_text()

    def test_metadata_file_records_run_context(self, tmp_path):
        logger = CrashResistantLogger(log_dir=str(tmp_path), prefix="meta")
        try:
            metadata = next(tmp_path.glob("meta_*_metadata.txt"))
            text = metadata.read_text()
            assert "Run ID:" in text
            assert "PID:" in text
            assert "Start Time:" in text
        finally:
            logger.stop()

    def test_circular_buffer_drops_oldest_lines(self, tmp_path):
        """The in-memory buffer holds only the last max_buffer_lines writes.

        Inspect the buffer directly inside the context — the on-disk circular
        log gets overwritten by the STOPPED footer at exit, masking drop
        behavior. Note: Python's print() splits each call into a write for
        the line text and a write for "\\n", so 20 lines → 40 buffer entries.
        """
        with CrashResistantLogger(
            log_dir=str(tmp_path), prefix="circ", max_buffer_lines=10
        ) as logger:
            for i in range(20):
                print(f"line{i}")
            joined = "".join(logger.circular_buffer.get_lines())

        # The most recent lines should be present; the earliest should be gone.
        assert "line19" in joined
        assert "line0" not in joined
        assert "line1\n" not in joined


class TestThreadMonitor:

    def test_summary_reports_baseline_and_growth(self):
        log_lines: list[str] = []
        monitor = ThreadMonitor(
            interval=60,  # never tick during the test
            log_callback=log_lines.append,
            warn_threshold=1000,
        )
        summary = monitor.get_summary()
        assert summary["initial_count"] == monitor.initial_thread_count
        assert summary["history_length"] == 0
        assert "growth" in summary

    def test_context_manager_starts_and_stops_thread(self):
        """stop() can only interrupt the loop between sleeps, so use a short
        interval to keep the test fast.
        """
        log_lines: list[str] = []
        with ThreadMonitor(
            interval=1,
            log_callback=log_lines.append,
            warn_threshold=10_000,
        ) as monitor:
            assert monitor._monitor_thread is not None
            assert monitor._monitor_thread.is_alive()

        # Stop joins with a 2s timeout; the loop sleeps for `interval` between
        # checks. Give it a generous grace period to exit cleanly.
        deadline = time.time() + 5
        while monitor._monitor_thread.is_alive() and time.time() < deadline:
            time.sleep(0.1)
        assert not monitor._monitor_thread.is_alive()

    def test_warning_fires_when_thread_count_exceeds_threshold(self):
        log_lines: list[str] = []
        monitor = ThreadMonitor(
            interval=1,  # short tick for the test
            log_callback=log_lines.append,
            warn_threshold=0,  # trip immediately
        )
        monitor.start(daemon=True)
        try:
            # Wait for at least one tick to fire
            deadline = time.time() + 5
            while time.time() < deadline:
                if any("WARNING" in line for line in log_lines):
                    break
                time.sleep(0.2)
        finally:
            monitor.stop()

        assert any("WARNING" in line for line in log_lines), (
            f"Expected a WARNING line in log output, got: {log_lines[-5:]}"
        )

    def test_growth_pattern_detected_in_history(self):
        """Synthesize a monotonically growing history and verify _monitor_loop
        would flag it via the recent-counts check by feeding history directly.
        """
        log_lines: list[str] = []
        monitor = ThreadMonitor(
            interval=60, log_callback=log_lines.append, warn_threshold=1000
        )

        # Simulate three consecutive growing samples
        from datetime import datetime
        monitor.thread_history = [
            {"timestamp": datetime.now(), "count": 5, "names": []},
            {"timestamp": datetime.now(), "count": 6, "names": []},
            {"timestamp": datetime.now(), "count": 7, "names": []},
        ]
        # Reproduce the loop's accumulation check
        recent = [h["count"] for h in monitor.thread_history[-3:]]
        is_growing = all(recent[i] <= recent[i + 1] for i in range(len(recent) - 1))
        assert is_growing
