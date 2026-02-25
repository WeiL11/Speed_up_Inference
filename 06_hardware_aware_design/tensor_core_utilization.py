"""
Chapter 6 — Hardware-Aware Design
File: tensor_core_utilization.py

Tensor core utilization: which shapes hit tensor cores and alignment rules.

NVIDIA tensor cores accelerate matrix operations but have strict requirements:
  - Shapes must be multiples of 8 (FP16) or 16 (INT8) for full utilization
  - Memory must be properly aligned
  - Certain dtype combinations are supported (FP16×FP16→FP16/FP32, etc.)

This file demonstrates:
  - How shape alignment affects performance
  - Tensor core eligible vs non-eligible operations
  - Practical implications for model design

Contents:
  - alignment_benchmark()  : show speedup from aligned vs unaligned shapes
  - dtype_support()        : which dtype combos use tensor cores
  - model_design_tips()    : practical guidelines for tensor core utilization
"""

import sys
import os
import time
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

from utils.benchmarking import benchmark_fn


# ---------------------------------------------------------------------------
# Alignment benchmark
# ---------------------------------------------------------------------------

def alignment_benchmark(
    sizes: list = None,
    device: str = None,
    warmup: int = 10,
    steps: int = 50,
):
    """
    Show performance impact of tensor shape alignment.

    Tensor cores on Ampere/Hopper require matrix dimensions to be multiples
    of 8 (for FP16) or 16 (for INT8). Unaligned shapes fall back to slower
    CUDA cores.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    has_cuda = "cuda" in device

    if sizes is None:
        sizes = [
            # (M, N, K, label)
            (512, 512, 512, "aligned (512)"),
            (513, 513, 513, "unaligned (513)"),
            (1024, 1024, 1024, "aligned (1024)"),
            (1023, 1023, 1023, "unaligned (1023)"),
            (768, 768, 768, "aligned (768)"),
            (769, 769, 769, "unaligned (769)"),
            (256, 2048, 1024, "aligned (256×2048×1024)"),
            (257, 2049, 1025, "unaligned (257×2049×1025)"),
        ]

    print(f"\n  Tensor Core Alignment Benchmark (FP16)")
    print(f"  device={device}")
    print(f"\n  {'Shape':<30}  {'Time (ms)':>12}  {'TFLOPS':>10}  {'Aligned':>10}")
    print(f"  {'-' * 30}  {'-' * 12}  {'-' * 10}  {'-' * 10}")

    for M, N, K, label in sizes:
        a = torch.randn(M, K, device=device, dtype=torch.float16)
        b = torch.randn(K, N, device=device, dtype=torch.float16)

        result = benchmark_fn(
            lambda: torch.matmul(a, b),
            warmup_steps=warmup, measure_steps=steps,
            name=label, sync_cuda=has_cuda, track_memory=False,
        )

        flops = 2 * M * N * K
        tflops = flops / (result.mean_ms / 1000) / 1e12
        is_aligned = all(d % 8 == 0 for d in [M, N, K])

        print(f"  {label:<30}  {result.mean_ms:>12.3f}  {tflops:>10.1f}  "
              f"{'YES' if is_aligned else 'NO':>10}")


# ---------------------------------------------------------------------------
# Dtype support for tensor cores
# ---------------------------------------------------------------------------

def dtype_benchmark(device: str = None, warmup: int = 10, steps: int = 50):
    """Benchmark matrix multiplication across dtypes to show tensor core effect."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    has_cuda = "cuda" in device

    M, N, K = 1024, 1024, 1024

    dtypes = [
        (torch.float32, "FP32"),
        (torch.float16, "FP16"),
        (torch.bfloat16, "BF16"),
    ]

    print(f"\n  Dtype Performance Comparison (matmul {M}×{K} @ {K}×{N})")
    print(f"\n  {'Dtype':<10}  {'Time (ms)':>12}  {'TFLOPS':>10}  {'Tensor Cores':>14}")
    print(f"  {'-' * 10}  {'-' * 12}  {'-' * 10}  {'-' * 14}")

    fp32_ms = None
    for dtype, name in dtypes:
        a = torch.randn(M, K, device=device, dtype=dtype)
        b = torch.randn(K, N, device=device, dtype=dtype)

        result = benchmark_fn(
            lambda: torch.matmul(a, b),
            warmup_steps=warmup, measure_steps=steps,
            name=name, sync_cuda=has_cuda, track_memory=False,
        )

        flops = 2 * M * N * K
        tflops = flops / (result.mean_ms / 1000) / 1e12
        uses_tc = name in ["FP16", "BF16"]

        if fp32_ms is None:
            fp32_ms = result.mean_ms

        print(f"  {name:<10}  {result.mean_ms:>12.3f}  {tflops:>10.1f}  "
              f"{'YES' if uses_tc else 'NO (CUDA cores)':>14}")


# ---------------------------------------------------------------------------
# Practical design guidelines
# ---------------------------------------------------------------------------

def model_design_tips():
    """Print practical guidelines for tensor core-friendly model design."""
    print(f"\n  ===  Tensor Core Design Guidelines  ===")
    print(f"""
  1. DIMENSION ALIGNMENT
     - hidden_dim should be divisible by 128 (ideally 256)
       Good: 768, 1024, 1152, 2048, 4096
       Bad:  700, 1000, 1500

     - num_heads * head_dim = hidden_dim
       head_dim should be 64, 128, or 256
       Good: 8 heads × 128 = 1024
       Bad:  7 heads × 100 = 700

     - intermediate_dim (MLP): multiple of 128
       Gemma uses 8/3 × hidden_dim, rounded to nearest 64

  2. BATCH × SEQUENCE LENGTH
     - B × T should be a multiple of 8 (FP16) or 16 (INT8)
     - Padding to alignment is almost free and can give 10-30% speedup

  3. DTYPE SELECTION
     - FP16 or BF16 required for tensor core acceleration
     - FP32 falls back to CUDA cores (much slower for matmul)
     - TF32 mode gives FP32 syntax with tensor core speed

  4. MEMORY LAYOUT
     - Contiguous tensors required (no .permute() without .contiguous())
     - Batch dimension should be outermost for coalesced memory access

  5. COMMON PITFALLS
     - Odd vocabulary sizes (e.g., 32001): pad to 32008 or 32064
     - Odd sequence lengths during prefill: pad to multiple of 8
     - Non-power-of-2 head counts: use 1, 2, 4, 8, 16, etc.
""")


# ---------------------------------------------------------------------------
# Vocab size padding demo
# ---------------------------------------------------------------------------

def vocab_padding_demo(device: str = None, warmup: int = 10, steps: int = 50):
    """Show impact of padding vocabulary size to tensor core-friendly boundary."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    has_cuda = "cuda" in device

    B, T, hidden = 8, 256, 1024

    vocab_sizes = [
        (32000, "32000 (aligned)"),
        (32001, "32001 (unaligned)"),
        (32064, "32064 (padded to 64)"),
        (32128, "32128 (padded to 128)"),
    ]

    print(f"\n  Vocabulary Size Padding Impact")
    print(f"  Linear: ({B}×{T}, {hidden}) → vocab_size, dtype=FP16")
    print(f"\n  {'Vocab Size':<25}  {'Time (ms)':>12}  {'Speedup':>10}")
    print(f"  {'-' * 25}  {'-' * 12}  {'-' * 10}")

    baseline_ms = None
    x = torch.randn(B * T, hidden, device=device, dtype=torch.float16)

    for vocab, label in vocab_sizes:
        linear = nn.Linear(hidden, vocab, bias=False, dtype=torch.float16, device=device)

        result = benchmark_fn(
            lambda: linear(x),
            warmup_steps=warmup, measure_steps=steps,
            name=label, sync_cuda=has_cuda, track_memory=False,
        )

        if baseline_ms is None:
            baseline_ms = result.mean_ms
        speedup = baseline_ms / result.mean_ms if result.mean_ms > 0 else 0

        print(f"  {label:<25}  {result.mean_ms:>12.3f}  {speedup:>9.2f}x")

        del linear

    print(f"\n  Even 1 extra element (32001 vs 32000) can cause tensor core fallback!")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("  Chapter 6: Tensor Core Utilization")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Running on: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")
        cap = torch.cuda.get_device_capability()
        print(f"  Compute capability: {cap[0]}.{cap[1]}")
        has_tc = cap[0] >= 7
        print(f"  Tensor cores: {'YES' if has_tc else 'NO (requires Volta or newer)'}")

    print("\n  [1] Shape alignment benchmark")
    alignment_benchmark(device=device)

    print("\n  [2] Dtype performance comparison")
    dtype_benchmark(device=device)

    print("\n  [3] Vocab size padding impact")
    vocab_padding_demo(device=device)

    print("\n  [4] Design guidelines")
    model_design_tips()
