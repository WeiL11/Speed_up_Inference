"""
Benchmarking utilities.

Provides timing, memory tracking, and result formatting helpers
used across all chapters.
"""

import timeit
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import numpy as np


@dataclass
class BenchmarkResult:
    """Container for benchmark measurements."""
    name: str
    times_ms: list[float] = field(default_factory=list)
    peak_memory_mb: float = 0.0
    allocated_memory_mb: float = 0.0

    @property
    def mean_ms(self) -> float:
        return float(np.mean(self.times_ms)) if self.times_ms else 0.0

    @property
    def std_ms(self) -> float:
        return float(np.std(self.times_ms)) if self.times_ms else 0.0

    @property
    def min_ms(self) -> float:
        return float(np.min(self.times_ms)) if self.times_ms else 0.0

    @property
    def max_ms(self) -> float:
        return float(np.max(self.times_ms)) if self.times_ms else 0.0

    @property
    def throughput(self) -> float:
        """Steps per second."""
        return 1000.0 / self.mean_ms if self.mean_ms > 0 else 0.0

    def summary(self) -> str:
        lines = [
            f"=== {self.name} ===",
            f"  Mean:       {self.mean_ms:8.2f} ms",
            f"  Std:        {self.std_ms:8.2f} ms",
            f"  Min:        {self.min_ms:8.2f} ms",
            f"  Max:        {self.max_ms:8.2f} ms",
            f"  Throughput: {self.throughput:8.2f} steps/s",
        ]
        if self.peak_memory_mb > 0:
            lines.append(f"  Peak mem:   {self.peak_memory_mb:8.1f} MB")
            lines.append(f"  Alloc mem:  {self.allocated_memory_mb:8.1f} MB")
        return "\n".join(lines)


def benchmark_fn(
    fn: Callable,
    warmup_steps: int = 5,
    measure_steps: int = 20,
    sync_cuda: bool = True,
    name: str = "benchmark",
    track_memory: bool = True,
) -> BenchmarkResult:
    """
    Benchmark a callable with warm-up, timing, and optional memory tracking.

    Args:
        fn: Zero-argument callable to benchmark.
        warmup_steps: Number of untimed warm-up iterations.
        measure_steps: Number of timed iterations.
        sync_cuda: Whether to call torch.cuda.synchronize() after each step.
        name: Label for the result.
        track_memory: Whether to track peak GPU memory.

    Returns:
        BenchmarkResult with timing and memory stats.
    """
    result = BenchmarkResult(name=name)

    # Warm-up
    for _ in range(warmup_steps):
        fn()
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

    # Reset memory stats before measurement
    if track_memory and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Timed runs
    times = []
    for _ in range(measure_steps):
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

        start = timeit.default_timer()
        fn()

        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

        end = timeit.default_timer()
        times.append((end - start) * 1000.0)  # Convert to ms

    result.times_ms = times

    if track_memory and torch.cuda.is_available():
        result.peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        result.allocated_memory_mb = torch.cuda.memory_allocated() / (1024 ** 2)

    return result


@contextmanager
def cuda_memory_tracker():
    """Context manager that yields peak memory usage in MB."""
    if not torch.cuda.is_available():
        yield {"peak_mb": 0.0, "allocated_mb": 0.0}
        return

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    stats = {}
    yield stats

    torch.cuda.synchronize()
    stats["peak_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
    stats["allocated_mb"] = torch.cuda.memory_allocated() / (1024 ** 2)


def print_gpu_info() -> None:
    """Print basic GPU information."""
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    print(f"GPU:          {props.name}")
    print(f"Compute cap:  {props.major}.{props.minor}")
    print(f"Total memory: {props.total_mem / (1024**3):.1f} GB")
    print(f"SM count:     {props.multi_processor_count}")
