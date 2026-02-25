"""
Chapter 4 — Batching & Scheduling
File: benchmark.py

Unified benchmark comparing all batching strategies:
  1. Static batching    (pad to max length)
  2. Dynamic batching   (bucket by length)
  3. Continuous batching (in-flight insertion/eviction)

Measures throughput (tokens/sec) at various concurrent request counts.
"""

import sys
import os
import time
import random
import argparse
import statistics
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from static_batching import StaticBatcher, generate_sequences
from dynamic_batching import DynamicBatcher
from continuous_batching import Request, ContinuousBatcher


# ---------------------------------------------------------------------------
# Throughput simulation
# ---------------------------------------------------------------------------

def simulate_static_throughput(
    sequences: list[list[int]],
    batch_size: int,
) -> dict:
    """Simulate static batching throughput."""
    batcher = StaticBatcher(max_batch_size=batch_size)
    total_tokens = sum(len(s) for s in sequences)

    # Process in fixed batches
    total_padded_tokens = 0
    num_batches = 0
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i + batch_size]
        max_len = max(len(s) for s in batch_seqs)
        total_padded_tokens += max_len * len(batch_seqs)
        num_batches += 1

    efficiency = total_tokens / total_padded_tokens if total_padded_tokens > 0 else 0

    return {
        "name": "Static",
        "total_tokens": total_tokens,
        "padded_tokens": total_padded_tokens,
        "efficiency": efficiency,
        "num_batches": num_batches,
    }


def simulate_dynamic_throughput(
    sequences: list[list[int]],
    batch_size: int,
    bucket_sizes: list[int] = None,
) -> dict:
    """Simulate dynamic batching throughput."""
    if bucket_sizes is None:
        bucket_sizes = [64, 128, 256, 512, 1024]

    batcher = DynamicBatcher(
        max_batch_size=batch_size,
        bucket_sizes=bucket_sizes,
    )
    total_tokens = sum(len(s) for s in sequences)

    # Bucket sequences
    total_padded_tokens = 0
    num_batches = 0
    buckets = {}
    for seq in sequences:
        # Find appropriate bucket
        bucket = bucket_sizes[-1]
        for bs in bucket_sizes:
            if len(seq) <= bs:
                bucket = bs
                break
        buckets.setdefault(bucket, []).append(seq)

    for bucket_size, bucket_seqs in buckets.items():
        for i in range(0, len(bucket_seqs), batch_size):
            batch = bucket_seqs[i:i + batch_size]
            total_padded_tokens += bucket_size * len(batch)
            num_batches += 1

    efficiency = total_tokens / total_padded_tokens if total_padded_tokens > 0 else 0

    return {
        "name": "Dynamic",
        "total_tokens": total_tokens,
        "padded_tokens": total_padded_tokens,
        "efficiency": efficiency,
        "num_batches": num_batches,
    }


def simulate_continuous_throughput(
    num_requests: int,
    gen_lengths: list[int],
    batch_size: int,
) -> dict:
    """Simulate continuous batching throughput."""
    batcher = ContinuousBatcher(max_batch_size=batch_size)
    now = time.perf_counter()

    for i, gen_len in enumerate(gen_lengths):
        req = Request(
            id=i,
            prompt_tokens=[0] * 10,
            max_new_tokens=gen_len,
            arrival_time=now,
        )
        batcher.add_request(req)

    summary = batcher.run_to_completion(verbose=False)
    total_tokens = sum(gen_lengths)

    return {
        "name": "Continuous",
        "total_tokens": total_tokens,
        "padded_tokens": total_tokens,  # no padding waste
        "efficiency": 1.0,
        "total_steps": summary["total_steps"],
        "throughput_per_step": summary["throughput_tok_per_step"],
    }


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    num_requests_list: list = None,
    batch_size: int = 8,
    min_len: int = 20,
    max_len: int = 500,
):
    """Compare all batching strategies at various request counts."""
    if num_requests_list is None:
        num_requests_list = [16, 32, 64, 128, 256]

    random.seed(42)

    print(f"\n{'=' * 80}")
    print(f"  Chapter 4: Batching Strategy Benchmark")
    print(f"  batch_size={batch_size}, seq_lengths={min_len}-{max_len}")
    print(f"{'=' * 80}")

    # Efficiency comparison
    print(f"\n  COMPUTE EFFICIENCY (useful tokens / total padded tokens)")
    print(f"\n  {'Requests':>10}  {'Static':>10}  {'Dynamic':>10}  {'Continuous':>12}")
    print(f"  {'-' * 10}  {'-' * 10}  {'-' * 10}  {'-' * 12}")

    for num_requests in num_requests_list:
        lengths = [random.randint(min_len, max_len) for _ in range(num_requests)]
        sequences = [[random.randint(1, 31999) for _ in range(l)] for l in lengths]

        static = simulate_static_throughput(sequences, batch_size)
        dynamic = simulate_dynamic_throughput(sequences, batch_size)
        continuous = simulate_continuous_throughput(num_requests, lengths, batch_size)

        print(f"  {num_requests:>10}  {static['efficiency']*100:>9.1f}%  "
              f"{dynamic['efficiency']*100:>9.1f}%  {continuous['efficiency']*100:>11.1f}%")

    # Throughput scaling with batch size
    print(f"\n  THROUGHPUT SCALING (tokens/step with continuous batching)")
    batch_sizes = [1, 2, 4, 8, 16, 32]
    num_requests = 64
    lengths = [random.randint(min_len, max_len) for _ in range(num_requests)]

    print(f"\n  {'Batch Size':>12}  {'Steps':>8}  {'Tok/Step':>10}  {'Speedup':>10}")
    print(f"  {'-' * 12}  {'-' * 8}  {'-' * 10}  {'-' * 10}")

    base_steps = None
    for bs in batch_sizes:
        result = simulate_continuous_throughput(num_requests, lengths, bs)
        if base_steps is None:
            base_steps = result["total_steps"]
        speedup = base_steps / result["total_steps"] if result["total_steps"] > 0 else 0
        print(f"  {bs:>12}  {result['total_steps']:>8}  "
              f"{result['throughput_per_step']:>10.1f}  {speedup:>9.1f}x")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chapter 4: Batching Benchmark")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--min_len", type=int, default=20)
    parser.add_argument("--max_len", type=int, default=500)
    parser.add_argument("--requests", type=int, nargs="+", default=None)
    args = parser.parse_args()

    run_benchmark(
        num_requests_list=args.requests,
        batch_size=args.batch_size,
        min_len=args.min_len,
        max_len=args.max_len,
    )
