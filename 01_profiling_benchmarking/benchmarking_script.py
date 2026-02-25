#!/usr/bin/env python3
"""
Chapter 1 — Benchmarking Script
================================

End-to-end benchmarking of transformer forward and backward passes.

Features:
  - Initialize a model from hyperparameters (num_layers, hidden_dim, etc.)
  - Generate a random batch of data
  - Run w warm-up steps (untimed), then time n measured steps
  - Support forward-only or forward+backward modes
  - Use timeit.default_timer() for high-resolution timing
  - Call torch.cuda.synchronize() after each step for accurate GPU timing
  - Report mean, std, min, max latency + throughput + GPU memory

Usage:
  python benchmarking_script.py --mode forward --num_layers 6 --hidden_dim 1024
  python benchmarking_script.py --mode both --warmup 10 --steps 50
"""

import argparse
import sys
import timeit
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.model_loader import create_model, generate_random_batch, TransformerConfig
from utils.benchmarking import BenchmarkResult, print_gpu_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark transformer forward/backward passes"
    )

    # Model hyperparameters
    parser.add_argument("--num_layers", type=int, default=6,
                        help="Number of transformer layers (default: 6)")
    parser.add_argument("--hidden_dim", type=int, default=1024,
                        help="Hidden dimension (default: 1024)")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of attention heads (default: 8)")
    parser.add_argument("--vocab_size", type=int, default=32000,
                        help="Vocabulary size (default: 32000)")

    # Data parameters
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size (default: 8)")
    parser.add_argument("--seq_len", type=int, default=512,
                        help="Sequence length (default: 512)")

    # Benchmark parameters
    parser.add_argument("--warmup", type=int, default=5,
                        help="Number of warm-up steps (default: 5)")
    parser.add_argument("--steps", type=int, default=20,
                        help="Number of measured steps (default: 20)")
    parser.add_argument("--mode", type=str, default="forward",
                        choices=["forward", "both"],
                        help="'forward' = forward only; 'both' = forward + backward (default: forward)")

    # Device / precision
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device to run on (default: cuda)")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16", "bfloat16"],
                        help="Data type (default: float32)")

    return parser.parse_args()


DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def run_benchmark(args: argparse.Namespace) -> BenchmarkResult:
    """Run the benchmark with the given arguments and return results."""

    device = args.device
    dtype = DTYPE_MAP[args.dtype]

    # ---- Print configuration ----
    print("=" * 60)
    print("BENCHMARK CONFIGURATION")
    print("=" * 60)
    print(f"  Model:      {args.num_layers} layers, dim={args.hidden_dim}, "
          f"heads={args.num_heads}")
    print(f"  Data:       batch={args.batch_size}, seq_len={args.seq_len}")
    print(f"  Timing:     {args.warmup} warmup + {args.steps} measured steps")
    print(f"  Mode:       {args.mode}")
    print(f"  Device:     {device}")
    print(f"  Dtype:      {args.dtype}")
    print()

    if device == "cuda" and torch.cuda.is_available():
        print_gpu_info()
        print()

    # ---- Initialize model ----
    print("Initializing model...")
    model = create_model(
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        vocab_size=args.vocab_size,
        dtype=dtype,
        device=device,
    )
    print(f"  Parameters: {model.param_count_str()} "
          f"({model.param_count():,} total)")

    model_mem = torch.cuda.memory_allocated() / (1024 ** 2) if device == "cuda" else 0
    print(f"  Model mem:  {model_mem:.1f} MB")
    print()

    # ---- Generate random data ----
    input_ids = generate_random_batch(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        device=device,
    )

    # For backward pass, we need a target
    targets = generate_random_batch(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        device=device,
    ) if args.mode == "both" else None

    loss_fn = nn.CrossEntropyLoss() if args.mode == "both" else None

    # ---- Define step functions ----
    def forward_step():
        with torch.no_grad():
            _ = model(input_ids)

    def forward_backward_step():
        model.zero_grad(set_to_none=True)
        logits = model(input_ids)
        # Reshape for cross-entropy: (B*T, vocab) vs (B*T,)
        loss = loss_fn(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
        )
        loss.backward()

    step_fn = forward_step if args.mode == "forward" else forward_backward_step

    # ---- Warm-up ----
    print(f"Running {args.warmup} warm-up steps...")
    for _ in range(args.warmup):
        step_fn()
        if device == "cuda":
            torch.cuda.synchronize()
    print("  Warm-up complete.")
    print()

    # ---- Measured runs ----
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    print(f"Running {args.steps} measured steps...")
    times_ms = []
    for i in range(args.steps):
        if device == "cuda":
            torch.cuda.synchronize()

        start = timeit.default_timer()
        step_fn()

        if device == "cuda":
            torch.cuda.synchronize()

        end = timeit.default_timer()
        elapsed_ms = (end - start) * 1000.0
        times_ms.append(elapsed_ms)

    # ---- Collect results ----
    result = BenchmarkResult(
        name=f"{args.mode} pass ({args.num_layers}L, dim={args.hidden_dim})",
        times_ms=times_ms,
    )

    if device == "cuda":
        result.peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        result.allocated_memory_mb = torch.cuda.memory_allocated() / (1024 ** 2)

    # ---- Print results ----
    print()
    print(result.summary())
    print()

    # Per-step breakdown
    tokens_per_step = args.batch_size * args.seq_len
    total_tokens = tokens_per_step * args.steps
    total_time_s = sum(times_ms) / 1000.0
    print(f"  Tokens/step:    {tokens_per_step:,}")
    print(f"  Total tokens:   {total_tokens:,}")
    print(f"  Total time:     {total_time_s:.3f} s")
    print(f"  Tokens/sec:     {total_tokens / total_time_s:,.0f}")

    return result


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        args.device = "cpu"

    result = run_benchmark(args)
    return result


if __name__ == "__main__":
    main()
