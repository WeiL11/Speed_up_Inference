"""
Chapter 6 — Hardware-Aware Design
File: memory_hierarchy.py

GPU memory hierarchy: L1/L2/HBM bandwidth and latency numbers.

Understanding the memory hierarchy is crucial for optimizing inference:
  - Registers:  fastest, ~TB/s, limited per thread
  - L1/SMEM:    shared memory, ~20 TB/s, 128-228 KB per SM
  - L2 Cache:   ~4-6 TB/s, 40-50 MB total (A100)
  - HBM (DRAM): ~2 TB/s (A100), ~3.35 TB/s (H100), main memory

The key insight: most LLM inference ops are MEMORY-BOUND, meaning they
spend more time loading data from HBM than doing arithmetic. This explains
why techniques like FlashAttention (reduce HBM reads) and quantization
(smaller data = less to load) give such big speedups.

Contents:
  - GPU_SPECS         : hardware specs for common GPUs
  - arithmetic_intensity : compute arithmetic intensity for each layer type
  - bandwidth_demo    : measure actual achievable bandwidth
  - hierarchy_analysis: show where time is spent for each operation
"""

import sys
import os
import time
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch


# ---------------------------------------------------------------------------
# GPU specifications
# ---------------------------------------------------------------------------

GPU_SPECS = {
    "A100-80GB": {
        "compute_fp32_tflops": 19.5,
        "compute_fp16_tflops": 312,    # with tensor cores
        "compute_bf16_tflops": 312,
        "compute_int8_tops": 624,
        "hbm_bandwidth_tb_s": 2.0,
        "hbm_size_gb": 80,
        "l2_cache_mb": 40,
        "sm_count": 108,
        "smem_per_sm_kb": 164,         # configurable up to 164 KB
    },
    "H100-80GB": {
        "compute_fp32_tflops": 67,
        "compute_fp16_tflops": 990,
        "compute_bf16_tflops": 990,
        "compute_int8_tops": 1979,
        "hbm_bandwidth_tb_s": 3.35,
        "hbm_size_gb": 80,
        "l2_cache_mb": 50,
        "sm_count": 132,
        "smem_per_sm_kb": 228,
    },
    "RTX-4090": {
        "compute_fp32_tflops": 82.6,
        "compute_fp16_tflops": 330.3,
        "compute_bf16_tflops": 330.3,
        "compute_int8_tops": 660.6,
        "hbm_bandwidth_tb_s": 1.008,   # GDDR6X
        "hbm_size_gb": 24,
        "l2_cache_mb": 72,
        "sm_count": 128,
        "smem_per_sm_kb": 128,
    },
}


# ---------------------------------------------------------------------------
# Arithmetic intensity analysis
# ---------------------------------------------------------------------------

def arithmetic_intensity(
    hidden_dim: int = 1024,
    num_heads: int = 8,
    seq_len: int = 512,
    batch_size: int = 1,
    dtype_bytes: int = 2,
):
    """
    Compute arithmetic intensity (FLOPs / bytes) for key operations.

    Arithmetic intensity determines whether an operation is compute-bound
    or memory-bound via the roofline model:
      - High AI (> machine's ops:byte ratio): compute-bound
      - Low AI  (< machine's ops:byte ratio): memory-bound

    For A100 FP16: peak = 312 TFLOPS / 2 TB/s = 156 FLOPs/byte
    Operations with AI < 156 are memory-bound on A100.
    """
    hD = hidden_dim // num_heads

    print(f"\n  Arithmetic Intensity Analysis")
    print(f"  hidden={hidden_dim}, nH={num_heads}, T={seq_len}, B={batch_size}")
    print(f"  dtype={dtype_bytes} bytes per element")

    ops = []

    # 1. Linear layer (e.g., Q/K/V projection)
    # FLOPs: 2 * B * T * in * out  (matmul)
    # Bytes: (in * out + B * T * in + B * T * out) * dtype_bytes
    in_dim = hidden_dim
    out_dim = hidden_dim
    flops = 2 * batch_size * seq_len * in_dim * out_dim
    bytes_moved = (in_dim * out_dim + batch_size * seq_len * (in_dim + out_dim)) * dtype_bytes
    ai = flops / bytes_moved
    ops.append(("Linear (QKV proj)", flops, bytes_moved, ai))

    # 2. Attention score: Q @ K^T
    # FLOPs: 2 * B * nH * T * T * hD
    # Bytes: (B * nH * T * hD * 2 + B * nH * T * T) * dtype_bytes
    flops = 2 * batch_size * num_heads * seq_len * seq_len * hD
    bytes_moved = (batch_size * num_heads * seq_len * hD * 2
                   + batch_size * num_heads * seq_len * seq_len) * dtype_bytes
    ai = flops / bytes_moved
    ops.append(("Attn QK^T", flops, bytes_moved, ai))

    # 3. Softmax (elementwise)
    # FLOPs: ~5 * B * nH * T * T  (exp, sum, div)
    # Bytes: 2 * B * nH * T * T * dtype_bytes  (read + write)
    flops = 5 * batch_size * num_heads * seq_len * seq_len
    bytes_moved = 2 * batch_size * num_heads * seq_len * seq_len * dtype_bytes
    ai = flops / bytes_moved
    ops.append(("Softmax", flops, bytes_moved, ai))

    # 4. Attention output: weights @ V
    # Same as QK^T
    flops = 2 * batch_size * num_heads * seq_len * seq_len * hD
    bytes_moved = (batch_size * num_heads * seq_len * seq_len
                   + batch_size * num_heads * seq_len * hD
                   + batch_size * num_heads * seq_len * hD) * dtype_bytes
    ai = flops / bytes_moved
    ops.append(("Attn wV", flops, bytes_moved, ai))

    # 5. MLP (gate + up + down projections + SiLU)
    intermediate = int(hidden_dim * 8 / 3)
    intermediate = ((intermediate + 63) // 64) * 64
    flops_mlp = 2 * batch_size * seq_len * hidden_dim * intermediate * 3  # 3 projections
    bytes_mlp = (hidden_dim * intermediate * 3  # weights
                 + batch_size * seq_len * (hidden_dim + intermediate * 2 + hidden_dim)) * dtype_bytes
    ai_mlp = flops_mlp / bytes_mlp
    ops.append(("MLP (gated)", flops_mlp, bytes_mlp, ai_mlp))

    # 6. RMSNorm (elementwise)
    flops = 4 * batch_size * seq_len * hidden_dim
    bytes_moved = (2 * batch_size * seq_len * hidden_dim + hidden_dim) * dtype_bytes
    ai = flops / bytes_moved
    ops.append(("RMSNorm", flops, bytes_moved, ai))

    # Print results
    print(f"\n  {'Operation':<20}  {'FLOPs':>14}  {'Bytes':>14}  {'AI (F/B)':>10}  {'Bound':>12}")
    print(f"  {'-' * 20}  {'-' * 14}  {'-' * 14}  {'-' * 10}  {'-' * 12}")

    # A100 FP16 threshold: 312 TFLOPS / 2 TB/s = 156 FLOPs/byte
    threshold = 156  # A100 FP16

    for name, flops, bw, ai in ops:
        bound = "COMPUTE" if ai > threshold else "MEMORY"
        print(f"  {name:<20}  {flops:>14,}  {bw:>14,}  {ai:>10.1f}  {bound:>12}")

    print(f"\n  Threshold (A100 FP16): {threshold} FLOPs/byte")
    print(f"  Operations below threshold are memory-bound → benefit from:")
    print(f"    - Quantization (fewer bytes to move)")
    print(f"    - Operator fusion (fewer HBM round-trips)")
    print(f"    - FlashAttention (reduce HBM access)")


# ---------------------------------------------------------------------------
# Bandwidth measurement
# ---------------------------------------------------------------------------

def measure_bandwidth(device: str = None, sizes_mb: list = None):
    """Measure actual achievable memory bandwidth on current GPU."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device != "cuda":
        print("\n  Bandwidth measurement requires CUDA.")
        return

    if sizes_mb is None:
        sizes_mb = [1, 10, 100, 500, 1000]

    print(f"\n  Memory Bandwidth Measurement")
    print(f"  GPU: {torch.cuda.get_device_name()}")

    print(f"\n  {'Size (MB)':>12}  {'Copy BW (GB/s)':>16}  {'% of peak':>12}")
    print(f"  {'-' * 12}  {'-' * 16}  {'-' * 12}")

    # Get theoretical peak from known specs
    gpu_name = torch.cuda.get_device_name().lower()
    peak_bw = 2000  # default GB/s
    for name, specs in GPU_SPECS.items():
        if any(part in gpu_name for part in name.lower().split("-")):
            peak_bw = specs["hbm_bandwidth_tb_s"] * 1000
            break

    for size_mb in sizes_mb:
        num_elements = size_mb * 1024 * 1024 // 4  # float32
        a = torch.randn(num_elements, device=device)
        b = torch.empty_like(a)

        # Warmup
        for _ in range(5):
            b.copy_(a)
        torch.cuda.synchronize()

        # Timed runs
        iterations = 20
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            b.copy_(a)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        bytes_per_iter = num_elements * 4 * 2  # read + write
        total_bytes = bytes_per_iter * iterations
        bw_gb_s = total_bytes / elapsed / 1e9
        pct_peak = bw_gb_s / peak_bw * 100

        print(f"  {size_mb:>12}  {bw_gb_s:>16.1f}  {pct_peak:>11.1f}%")

    print(f"\n  Theoretical peak: ~{peak_bw:.0f} GB/s")
    print(f"  Achievable: ~80-90% of peak for large transfers")
    print(f"  Small transfers are latency-bound (kernel launch overhead)")


# ---------------------------------------------------------------------------
# GPU info
# ---------------------------------------------------------------------------

def print_gpu_hierarchy():
    """Print memory hierarchy info for the current GPU."""
    if not torch.cuda.is_available():
        print("\n  No CUDA GPU available. Showing A100 reference specs.")
        specs = GPU_SPECS["A100-80GB"]
        print(f"\n  A100-80GB Memory Hierarchy:")
        print(f"    Registers:  per thread, ~TB/s")
        print(f"    L1/SMEM:    {specs['smem_per_sm_kb']} KB/SM × {specs['sm_count']} SMs")
        print(f"    L2 Cache:   {specs['l2_cache_mb']} MB")
        print(f"    HBM:        {specs['hbm_size_gb']} GB @ {specs['hbm_bandwidth_tb_s']} TB/s")
        return

    props = torch.cuda.get_device_properties(0)
    print(f"\n  Current GPU: {props.name}")
    print(f"  Compute capability: {props.major}.{props.minor}")
    print(f"  SM count: {props.multi_processor_count}")
    print(f"  Total memory: {props.total_mem / (1024**3):.1f} GB")

    # Try to match known specs
    gpu_name = props.name.lower()
    matched = None
    for name, specs in GPU_SPECS.items():
        if any(part in gpu_name for part in name.lower().split("-")):
            matched = (name, specs)
            break

    if matched:
        name, specs = matched
        print(f"\n  Known specs for {name}:")
        print(f"    FP32 compute:    {specs['compute_fp32_tflops']:.1f} TFLOPS")
        print(f"    FP16 compute:    {specs['compute_fp16_tflops']:.0f} TFLOPS (tensor cores)")
        print(f"    HBM bandwidth:   {specs['hbm_bandwidth_tb_s']:.2f} TB/s")
        print(f"    L2 cache:        {specs['l2_cache_mb']} MB")
        print(f"    SMEM per SM:     {specs['smem_per_sm_kb']} KB")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("  Chapter 6: GPU Memory Hierarchy")
    print("=" * 70)

    print("\n  [1] GPU info and memory hierarchy")
    print_gpu_hierarchy()

    print("\n  [2] Arithmetic intensity analysis")
    arithmetic_intensity(batch_size=1, seq_len=512)
    print()
    arithmetic_intensity(batch_size=32, seq_len=512)

    print("\n  [3] Memory bandwidth measurement")
    measure_bandwidth()

    print("\n  Key takeaways:")
    print("  - LLM inference at small batch is overwhelmingly memory-bound")
    print("  - Linear layers become compute-bound at large batch sizes")
    print("  - Elementwise ops (softmax, norm) are always memory-bound")
    print("  - Optimization strategy: reduce bytes moved (quantize, fuse, flash)")
