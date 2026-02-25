#!/usr/bin/env python3
"""
Chapter 1 — Memory Profiling Script
=====================================

Profile GPU memory usage of the transformer model during forward pass,
backward pass, and optimizer step, using PyTorch's built-in memory
snapshot facility (torch.cuda.memory).

Features:
  - Record a full memory timeline via torch.cuda.memory._record_memory_history()
  - Export snapshots compatible with pytorch.org/memory_viz
  - Support multiple context lengths (e.g., 128, 256, 512)
  - Support profiling modes: forward, fwd_bwd, train (with AdamW)
  - Support mixed-precision (FP16 / BF16 via torch.amp)
  - Report peak memory per configuration
  - Compute theoretical activation tensor sizes for analysis

This script is designed to answer the following questions:

  (a) Memory timeline: What does the active memory timeline look like for
      forward-only vs. a full training step? Can you identify stages from
      the peaks?

  (b) Peak memory by context length: What is the peak memory for forward
      pass and full training step at context lengths 128, 256, 512?

  (c) Mixed-precision impact: How does FP16/BF16 affect peak memory?

  (d) Activation tensor size: What is the size of a residual-stream
      activation tensor in single-precision? (batch * seq_len * hidden_dim * 4 bytes)

  (e) Memory timeline detail: When reducing detail level in memory_viz,
      what allocations remain visible?

Usage:
  # Peak memory table across context lengths (forward only)
  python memory_profiling.py --mode forward --seq_lens 128 256 512

  # Peak memory table (full training step)
  python memory_profiling.py --mode train --seq_lens 128 256 512

  # Export memory snapshot for pytorch.org/memory_viz
  python memory_profiling.py --mode forward --seq_len 512 --snapshot snapshot_fwd.pickle

  # Full training step snapshot
  python memory_profiling.py --mode train --seq_len 512 --snapshot snapshot_train.pickle

  # Mixed-precision
  python memory_profiling.py --mode train --seq_lens 128 256 512 --mixed_precision

  # Compute theoretical activation size
  python memory_profiling.py --calc_activation_size --batch_size 8 --seq_len 512 --hidden_dim 2560
"""

import argparse
import pickle
import sys
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.model_loader import create_model, generate_random_batch
from utils.benchmarking import print_gpu_info


# ---------------------------------------------------------------------------
# Memory snapshot helpers
# ---------------------------------------------------------------------------

def start_memory_recording():
    """Start recording memory allocation history for memory_viz."""
    torch.cuda.memory._record_memory_history(max_entries=100_000)


def stop_memory_recording():
    """Stop recording memory allocation history."""
    torch.cuda.memory._record_memory_history(enabled=None)


def export_memory_snapshot(filepath: str):
    """
    Export the memory snapshot to a pickle file compatible with
    pytorch.org/memory_viz.

    To visualize:
      1. Go to https://pytorch.org/memory_viz
      2. Drag-and-drop the .pickle file
      3. Select "Active Memory Timeline" view
    """
    snapshot = torch.cuda.memory._snapshot()
    with open(filepath, "wb") as f:
        pickle.dump(snapshot, f)
    print(f"  Snapshot saved to: {filepath}")
    print(f"  Visualize at: https://pytorch.org/memory_viz")


# ---------------------------------------------------------------------------
# Profiling functions
# ---------------------------------------------------------------------------

def profile_peak_memory(
    num_layers: int,
    hidden_dim: int,
    num_heads: int,
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    mode: str,
    mixed_precision: bool = False,
    device: str = "cuda",
    snapshot_path: str | None = None,
) -> float:
    """
    Run one profiling pass and return peak GPU memory in MB.

    If snapshot_path is given, also export a memory snapshot pickle.
    """
    dtype = torch.float32

    # Clear everything
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Build model
    model = create_model(
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        vocab_size=vocab_size,
        dtype=dtype,
        device=device,
    )

    # Data
    input_ids = generate_random_batch(batch_size, seq_len, vocab_size, device)
    targets = generate_random_batch(batch_size, seq_len, vocab_size, device)
    loss_fn = nn.CrossEntropyLoss()

    optimizer = None
    if mode == "train":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Warm-up (1 step, no recording)
    _run_step(model, input_ids, targets, loss_fn, optimizer, mode, mixed_precision)
    torch.cuda.synchronize()

    # Reset after warm-up
    torch.cuda.reset_peak_memory_stats()

    # Start snapshot recording if requested
    if snapshot_path:
        start_memory_recording()

    # Profiled step
    _run_step(model, input_ids, targets, loss_fn, optimizer, mode, mixed_precision)
    torch.cuda.synchronize()

    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # Export snapshot
    if snapshot_path:
        export_memory_snapshot(snapshot_path)
        stop_memory_recording()

    # Cleanup
    del model, input_ids, targets, loss_fn, optimizer
    torch.cuda.empty_cache()

    return peak_mb


def _run_step(model, input_ids, targets, loss_fn, optimizer, mode, mixed_precision):
    """Execute one forward/backward/optimizer step."""

    amp_enabled = mixed_precision
    amp_dtype = torch.float16  # Use float16 for mixed precision

    if mode == "forward":
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
            logits = model(input_ids)

    elif mode == "fwd_bwd":
        model.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
            logits = model(input_ids)
            loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()

    elif mode == "train":
        model.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
            logits = model(input_ids)
            loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()


# ---------------------------------------------------------------------------
# Activation size calculator
# ---------------------------------------------------------------------------

def calc_activation_size(batch_size: int, seq_len: int, hidden_dim: int) -> None:
    """
    Calculate the size of a single activation tensor in the residual stream.

    Shape: (batch_size, seq_len, hidden_dim)
    Each element is float32 = 4 bytes.
    """
    num_elements = batch_size * seq_len * hidden_dim
    size_bytes = num_elements * 4  # float32
    size_mb = size_bytes / (1024 ** 2)

    print(f"\nActivation tensor size calculation:")
    print(f"  Shape:     ({batch_size}, {seq_len}, {hidden_dim})")
    print(f"  Elements:  {num_elements:,}")
    print(f"  Bytes:     {size_bytes:,} (float32, 4 bytes/element)")
    print(f"  Size:      {size_mb:.2f} MB")
    print(f"\n  Derivation: {batch_size} x {seq_len} x {hidden_dim} x 4 bytes "
          f"= {size_bytes:,} bytes = {size_mb:.2f} MB")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPU memory profiling for transformer models"
    )

    # Model
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--batch_size", type=int, default=8)

    # Sequence length(s)
    parser.add_argument("--seq_len", type=int, default=512,
                        help="Single sequence length (used with --snapshot)")
    parser.add_argument("--seq_lens", type=int, nargs="+", default=None,
                        help="Multiple sequence lengths for peak memory table "
                             "(e.g., --seq_lens 128 256 512)")

    # Mode
    parser.add_argument("--mode", type=str, default="forward",
                        choices=["forward", "fwd_bwd", "train"],
                        help="Profiling mode (default: forward)")

    # Options
    parser.add_argument("--mixed_precision", action="store_true",
                        help="Enable FP16 mixed-precision via torch.amp")
    parser.add_argument("--snapshot", type=str, default=None,
                        help="Path to save memory snapshot pickle "
                             "(for pytorch.org/memory_viz)")

    # Activation calculator
    parser.add_argument("--calc_activation_size", action="store_true",
                        help="Calculate theoretical activation tensor size and exit")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.calc_activation_size:
        calc_activation_size(args.batch_size, args.seq_len, args.hidden_dim)
        return

    if not torch.cuda.is_available():
        print("ERROR: CUDA is required for memory profiling.")
        sys.exit(1)

    print("=" * 60)
    print("MEMORY PROFILING")
    print("=" * 60)
    print(f"  Model:     {args.num_layers} layers, dim={args.hidden_dim}, "
          f"heads={args.num_heads}")
    print(f"  Batch:     {args.batch_size}")
    print(f"  Mode:      {args.mode}")
    print(f"  Mixed-prec:{' ON (FP16)' if args.mixed_precision else ' OFF (FP32)'}")
    print()
    print_gpu_info()
    print()

    # --- Single snapshot mode ---
    if args.snapshot:
        print(f"Recording memory snapshot (seq_len={args.seq_len})...")
        peak = profile_peak_memory(
            num_layers=args.num_layers,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            vocab_size=args.vocab_size,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            mode=args.mode,
            mixed_precision=args.mixed_precision,
            snapshot_path=args.snapshot,
        )
        print(f"  Peak memory: {peak:.1f} MB")
        return

    # --- Peak memory table across context lengths ---
    seq_lens = args.seq_lens or [128, 256, 512]

    print(f"Profiling peak memory across context lengths: {seq_lens}")
    print()

    results = []
    for sl in seq_lens:
        print(f"  seq_len={sl}...", end=" ", flush=True)
        peak = profile_peak_memory(
            num_layers=args.num_layers,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            vocab_size=args.vocab_size,
            batch_size=args.batch_size,
            seq_len=sl,
            mode=args.mode,
            mixed_precision=args.mixed_precision,
        )
        results.append((sl, peak))
        print(f"{peak:.1f} MB")

    # Print table
    precision_label = "FP16 (mixed)" if args.mixed_precision else "FP32"
    print(f"\n{'=' * 50}")
    print(f"Peak Memory — mode={args.mode}, {precision_label}")
    print(f"{'=' * 50}")
    print(f"{'Context Length':>16} | {'Peak Memory (MB)':>18}")
    print(f"{'-' * 16}-+-{'-' * 18}")
    for sl, peak in results:
        print(f"{sl:>16} | {peak:>18.1f}")
    print()


if __name__ == "__main__":
    main()
