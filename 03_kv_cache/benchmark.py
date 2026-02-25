"""
Chapter 3 — KV Cache Management
File: benchmark.py

Unified benchmark comparing all KV cache strategies:
  1. Naive   (append + cat every step)
  2. Static  (pre-allocated fixed buffer)
  3. Paged   (block-based allocation)

Measures:
  - Allocation cost per decode step
  - Total generation latency
  - Peak memory usage
  - Memory utilization efficiency
"""

import sys
import time
import math
import argparse
import statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from naive_kv_cache import NaiveKVCache
from static_kv_cache import StaticKVCache
from paged_kv_cache import BlockManager, PagedKVCache


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def bench_naive_cache(
    batch: int, nKV: int, hD: int, decode_steps: int, device: str
) -> dict:
    """Benchmark naive append-and-cat cache."""
    cache = NaiveKVCache()
    times = []

    for step in range(decode_steps):
        k = torch.randn(batch, nKV, 1, hD, device=device)
        v = torch.randn(batch, nKV, 1, hD, device=device)
        if torch.cuda.is_available() and "cuda" in device:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        cache.update(k, v)
        if torch.cuda.is_available() and "cuda" in device:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return {
        "name": "Naive",
        "total_ms": sum(times),
        "mean_step_ms": statistics.mean(times),
        "last_10_ms": statistics.mean(times[-10:]),
        "times": times,
    }


def bench_static_cache(
    batch: int, nKV: int, hD: int, decode_steps: int, max_seq: int, device: str
) -> dict:
    """Benchmark static pre-allocated cache."""
    cache = StaticKVCache(batch, nKV, max_seq, hD, device)
    times = []

    for step in range(decode_steps):
        k = torch.randn(batch, nKV, 1, hD, device=device)
        v = torch.randn(batch, nKV, 1, hD, device=device)
        if torch.cuda.is_available() and "cuda" in device:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        cache.update(k, v)
        if torch.cuda.is_available() and "cuda" in device:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return {
        "name": "Static",
        "total_ms": sum(times),
        "mean_step_ms": statistics.mean(times),
        "last_10_ms": statistics.mean(times[-10:]),
        "times": times,
        "memory_mb": cache.memory_mb(),
    }


def bench_paged_cache(
    batch: int, nKV: int, hD: int, decode_steps: int, block_size: int, device: str
) -> dict:
    """Benchmark paged block-based cache."""
    num_blocks = math.ceil(decode_steps / block_size) + 4
    manager = BlockManager(num_blocks, block_size, nKV, hD, device)
    cache = PagedKVCache(manager)
    times = []

    for step in range(decode_steps):
        k = torch.randn(nKV, 1, hD, device=device)
        v = torch.randn(nKV, 1, hD, device=device)
        if torch.cuda.is_available() and "cuda" in device:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        cache.append(k, v)
        if torch.cuda.is_available() and "cuda" in device:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return {
        "name": "Paged",
        "total_ms": sum(times),
        "mean_step_ms": statistics.mean(times),
        "last_10_ms": statistics.mean(times[-10:]),
        "times": times,
        "blocks_used": cache.num_blocks_used,
        "pool_memory_mb": manager.memory_mb(),
    }


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    decode_steps: int = 256,
    batch: int = 1,
    nKV: int = 8,
    hD: int = 64,
    block_size: int = 16,
    device: str = "cuda",
):
    """Run all cache benchmarks and print comparison."""
    max_seq = decode_steps + 64  # extra room for static cache

    print(f"\n{'=' * 75}")
    print(f"  Chapter 3: KV Cache Benchmark")
    print(f"  device={device}, decode_steps={decode_steps}, batch={batch}")
    print(f"  nKV={nKV}, hD={hD}, block_size={block_size}")
    print(f"{'=' * 75}")

    # Run benchmarks
    results = []
    results.append(bench_naive_cache(batch, nKV, hD, decode_steps, device))
    results.append(bench_static_cache(batch, nKV, hD, decode_steps, max_seq, device))
    results.append(bench_paged_cache(batch, nKV, hD, decode_steps, block_size, device))

    # Summary table
    print(f"\n  {'Method':<10}  {'Total (ms)':>12}  {'Mean/step':>12}  {'Last 10':>12}  {'Speedup':>10}")
    print(f"  {'-' * 10}  {'-' * 12}  {'-' * 12}  {'-' * 12}  {'-' * 10}")

    naive_total = results[0]["total_ms"]
    for r in results:
        speedup = naive_total / r["total_ms"] if r["total_ms"] > 0 else 0
        print(f"  {r['name']:<10}  {r['total_ms']:>12.2f}  "
              f"{r['mean_step_ms']:>12.4f}  {r['last_10_ms']:>12.4f}  "
              f"{speedup:>9.2f}x")

    # Scaling analysis: how step cost grows with position
    print(f"\n  Step cost scaling (first 10 vs last 10 steps):")
    for r in results:
        first_10 = statistics.mean(r["times"][:10])
        last_10 = r["last_10_ms"]
        ratio = last_10 / first_10 if first_10 > 0 else 0
        print(f"    {r['name']:<10}: first_10={first_10:.4f} ms, "
              f"last_10={last_10:.4f} ms, ratio={ratio:.2f}x")

    print(f"\n  Key observations:")
    print(f"  - Naive: O(N) per step → O(N²) total (copy grows each step)")
    print(f"  - Static: O(1) per step (write at offset, no copy)")
    print(f"  - Paged: ~O(1) per step (occasional block allocation)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chapter 3: KV Cache Benchmark")
    parser.add_argument("--decode_steps", type=int, default=256)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--nKV", type=int, default=8)
    parser.add_argument("--hD", type=int, default=64)
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_benchmark(
        decode_steps=args.decode_steps,
        batch=args.batch,
        nKV=args.nKV,
        hD=args.hD,
        block_size=args.block_size,
        device=args.device,
    )
