"""
Chapter 6 — Hardware-Aware Design
File: roofline_model.py

Roofline model: visualize whether each operation is compute-bound or
memory-bound, and how optimizations shift the operating point.

The roofline model plots achievable performance (FLOPS) as a function of
arithmetic intensity (FLOPs/byte):
  - Below the ridge point: memory-bound (limited by bandwidth)
  - Above the ridge point: compute-bound (limited by peak FLOPS)

Contents:
  - compute_roofline()  : calculate roofline for a GPU
  - profile_operations(): measure actual arithmetic intensity per op
  - plot_roofline()     : generate roofline plot (text or matplotlib)
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from memory_hierarchy import GPU_SPECS


# ---------------------------------------------------------------------------
# Roofline computation
# ---------------------------------------------------------------------------

def compute_roofline(
    peak_flops_tflops: float,
    bandwidth_tb_s: float,
    ai_range: tuple = (0.1, 1000),
    num_points: int = 100,
) -> list:
    """
    Compute roofline ceiling for given hardware.

    Args:
        peak_flops_tflops: peak compute in TFLOPS
        bandwidth_tb_s: memory bandwidth in TB/s
        ai_range: range of arithmetic intensity to compute
        num_points: number of points to compute

    Returns:
        List of (ai, achievable_tflops) tuples
    """
    peak_flops = peak_flops_tflops  # TFLOPS
    bandwidth = bandwidth_tb_s * 1000  # GB/s

    # Ridge point: where compute ceiling meets bandwidth ceiling
    ridge_ai = peak_flops * 1e3 / bandwidth  # FLOPS/byte

    points = []
    for i in range(num_points):
        log_ai = math.log10(ai_range[0]) + (
            math.log10(ai_range[1]) - math.log10(ai_range[0])
        ) * i / (num_points - 1)
        ai = 10 ** log_ai

        # Achievable = min(peak, bandwidth * ai)
        bw_limited = bandwidth * ai / 1e3  # TFLOPS
        achievable = min(peak_flops, bw_limited)
        points.append((ai, achievable))

    return points, ridge_ai


# ---------------------------------------------------------------------------
# Operation profiling
# ---------------------------------------------------------------------------

def profile_transformer_ops(
    hidden_dim: int = 1024,
    num_heads: int = 8,
    seq_len: int = 512,
    batch_size: int = 1,
    dtype_bytes: int = 2,
) -> list:
    """
    Calculate arithmetic intensity for each transformer operation.

    Returns list of (name, flops, bytes, ai) tuples.
    """
    hD = hidden_dim // num_heads
    intermediate = ((int(hidden_dim * 8 / 3) + 63) // 64) * 64
    B, T = batch_size, seq_len

    ops = []

    # QKV projection
    flops = 2 * B * T * hidden_dim * hidden_dim * 3
    bw = (hidden_dim * hidden_dim * 3 + B * T * hidden_dim * 4) * dtype_bytes
    ops.append(("QKV Proj", flops, bw, flops / bw))

    # Attention (QK^T)
    flops = 2 * B * num_heads * T * T * hD
    bw = (B * num_heads * T * hD * 2 + B * num_heads * T * T) * dtype_bytes
    ops.append(("Attn QK^T", flops, bw, flops / bw))

    # Softmax
    flops = 5 * B * num_heads * T * T
    bw = 2 * B * num_heads * T * T * dtype_bytes
    ops.append(("Softmax", flops, bw, flops / bw))

    # Attention (wV)
    flops = 2 * B * num_heads * T * T * hD
    bw = (B * num_heads * T * T + B * num_heads * T * hD * 2) * dtype_bytes
    ops.append(("Attn wV", flops, bw, flops / bw))

    # Output projection
    flops = 2 * B * T * hidden_dim * hidden_dim
    bw = (hidden_dim * hidden_dim + B * T * hidden_dim * 2) * dtype_bytes
    ops.append(("Out Proj", flops, bw, flops / bw))

    # MLP (3 projections)
    flops = 2 * B * T * hidden_dim * intermediate * 3
    bw = (hidden_dim * intermediate * 3 + B * T * (hidden_dim + intermediate * 2 + hidden_dim)) * dtype_bytes
    ops.append(("MLP", flops, bw, flops / bw))

    # RMSNorm
    flops = 4 * B * T * hidden_dim
    bw = (2 * B * T * hidden_dim + hidden_dim) * dtype_bytes
    ops.append(("RMSNorm", flops, bw, flops / bw))

    return ops


# ---------------------------------------------------------------------------
# Text-based roofline plot
# ---------------------------------------------------------------------------

def text_roofline_plot(
    gpu_name: str = "A100-80GB",
    operations: list = None,
    dtype: str = "fp16",
):
    """Print a text-based roofline visualization."""
    specs = GPU_SPECS.get(gpu_name, GPU_SPECS["A100-80GB"])

    if dtype == "fp16":
        peak = specs["compute_fp16_tflops"]
    elif dtype == "fp32":
        peak = specs["compute_fp32_tflops"]
    else:
        peak = specs["compute_fp16_tflops"]

    bandwidth = specs["hbm_bandwidth_tb_s"]
    ridge_ai = peak / (bandwidth * 1000) * 1e3

    print(f"\n  Roofline Model: {gpu_name} ({dtype.upper()})")
    print(f"  Peak compute: {peak} TFLOPS")
    print(f"  HBM bandwidth: {bandwidth} TB/s")
    print(f"  Ridge point: {ridge_ai:.1f} FLOPs/byte")

    if operations:
        print(f"\n  {'Operation':<15}  {'AI (F/B)':>10}  {'Region':>12}  {'Roofline':>40}")
        print(f"  {'-' * 15}  {'-' * 10}  {'-' * 12}  {'-' * 40}")

        max_bar = 35
        for name, flops, bw, ai in operations:
            region = "COMPUTE" if ai > ridge_ai else "MEMORY"
            achievable = min(peak, bandwidth * 1000 * ai / 1e3)
            pct = achievable / peak
            bar_len = int(pct * max_bar)
            bar = "#" * bar_len + "." * (max_bar - bar_len)
            pct_str = f"{pct * 100:5.1f}%"
            print(f"  {name:<15}  {ai:>10.1f}  {region:>12}  |{bar}| {pct_str}")

    print(f"\n  Legend: # = achievable fraction of peak, . = unused")
    print(f"  Memory-bound ops benefit from: quantization, fusion, FlashAttention")
    print(f"  Compute-bound ops benefit from: tensor cores, lower precision")


# ---------------------------------------------------------------------------
# Batch size effect on arithmetic intensity
# ---------------------------------------------------------------------------

def batch_size_effect(
    hidden_dim: int = 1024,
    num_heads: int = 8,
    seq_len: int = 512,
    batch_sizes: list = None,
):
    """Show how batch size shifts operations from memory-bound to compute-bound."""
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16, 32, 64]

    print(f"\n  Batch Size Effect on Arithmetic Intensity")
    print(f"  (QKV Linear projection, hidden={hidden_dim}, T={seq_len})")

    # A100 FP16 ridge point
    ridge_ai = 312 / (2.0 * 1000) * 1e3  # ~156 FLOPs/byte

    print(f"\n  {'Batch':>8}  {'AI (F/B)':>10}  {'Region':>12}  {'% Peak':>10}")
    print(f"  {'-' * 8}  {'-' * 10}  {'-' * 12}  {'-' * 10}")

    for B in batch_sizes:
        flops = 2 * B * seq_len * hidden_dim * hidden_dim * 3
        bw = (hidden_dim * hidden_dim * 3 + B * seq_len * hidden_dim * 4) * 2  # fp16
        ai = flops / bw
        region = "COMPUTE" if ai > ridge_ai else "MEMORY"
        achievable_pct = min(1.0, ai / ridge_ai) * 100

        print(f"  {B:>8}  {ai:>10.1f}  {region:>12}  {achievable_pct:>9.1f}%")

    print(f"\n  Ridge point (A100 FP16): {ridge_ai:.0f} FLOPs/byte")
    print(f"  At batch=1, even large matmuls are memory-bound!")
    print(f"  Larger batch → higher AI → compute-bound → better GPU utilization")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("  Chapter 6: Roofline Model")
    print("=" * 70)

    print("\n  [1] Roofline analysis (batch=1, inference)")
    ops = profile_transformer_ops(batch_size=1, seq_len=512)
    text_roofline_plot("A100-80GB", ops)

    print("\n  [2] Roofline analysis (batch=32, inference)")
    ops = profile_transformer_ops(batch_size=32, seq_len=512)
    text_roofline_plot("A100-80GB", ops)

    print("\n  [3] Batch size effect on arithmetic intensity")
    batch_size_effect()

    print("\n  Key takeaways:")
    print("  - At batch=1 (typical inference): most ops are memory-bound")
    print("  - At large batch: linear layers become compute-bound")
    print("  - Softmax and norm ops are ALWAYS memory-bound")
    print("  - This explains why quantization helps so much at small batch")
    print("    (fewer bytes to load) but less at large batch (already compute-bound)")
