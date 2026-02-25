"""
Chapter 2 — Efficient Attention
File: benchmark.py

Unified benchmark comparing all attention implementations:
  1. Naive attention    (O(N²) memory, explicit matmul)
  2. JIT attention      (O(N²) memory, fused ops)
  3. FlashAttention-2   (O(N) memory, tiled online softmax)
  4. F.sdpa             (PyTorch built-in, auto-selects backend)

Sweeps across sequence lengths (512 → 8192) measuring:
  - Wall-clock latency (ms)
  - Peak GPU memory (MB)
"""

import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F

from utils.benchmarking import benchmark_fn
from naive_attention import naive_attention
from jit_attention import jit_attention
from triton_flash_attention import flash_attention, HAS_TRITON


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    seq_lens: list,
    batch: int = 2,
    num_heads: int = 8,
    head_dim: int = 64,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
    warmup: int = 3,
    steps: int = 10,
):
    """Run all attention variants across sequence lengths."""
    has_cuda = torch.cuda.is_available() and "cuda" in device

    variants = {
        "Naive":     lambda q, k, v: naive_attention(q, k, v, causal=True),
        "JIT":       lambda q, k, v: jit_attention(q, k, v, causal=True),
        "Flash":     lambda q, k, v: flash_attention(q, k, v, causal=True),
        "F.sdpa":    lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True),
    }

    # Header
    print(f"\n{'=' * 90}")
    print(f"  Chapter 2: Attention Benchmark")
    print(f"  device={device}, batch={batch}, nH={num_heads}, hD={head_dim}, dtype={dtype}")
    print(f"  Triton available: {HAS_TRITON}")
    print(f"{'=' * 90}")

    # ---- Latency comparison ----
    print(f"\n  LATENCY (ms)")
    header = f"  {'SeqLen':>8}"
    for name in variants:
        header += f"  {name:>12}"
    print(header)
    print(f"  {'-' * 8}" + f"  {'-' * 12}" * len(variants))

    latency_results = {}

    for T in seq_lens:
        if not has_cuda and T > 1024:
            continue

        q = torch.randn(batch, num_heads, T, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch, num_heads, T, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch, num_heads, T, head_dim, device=device, dtype=dtype)

        row = f"  {T:>8,}"
        latency_results[T] = {}

        for name, fn in variants.items():
            try:
                result = benchmark_fn(
                    lambda: fn(q, k, v),
                    warmup_steps=warmup,
                    measure_steps=steps,
                    name=name,
                    sync_cuda=has_cuda,
                    track_memory=False,
                )
                row += f"  {result.mean_ms:>12.2f}"
                latency_results[T][name] = result.mean_ms
            except Exception as e:
                row += f"  {'ERROR':>12}"
                latency_results[T][name] = None

        print(row)

    # ---- Memory comparison ----
    if has_cuda:
        print(f"\n  PEAK MEMORY (MB)")
        header = f"  {'SeqLen':>8}"
        for name in variants:
            header += f"  {name:>12}"
        print(header)
        print(f"  {'-' * 8}" + f"  {'-' * 12}" * len(variants))

        for T in seq_lens:
            q = torch.randn(batch, num_heads, T, head_dim, device=device, dtype=dtype)
            k = torch.randn(batch, num_heads, T, head_dim, device=device, dtype=dtype)
            v = torch.randn(batch, num_heads, T, head_dim, device=device, dtype=dtype)

            row = f"  {T:>8,}"
            for name, fn in variants.items():
                try:
                    torch.cuda.reset_peak_memory_stats()
                    # Warmup
                    for _ in range(2):
                        _ = fn(q, k, v)
                        torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    _ = fn(q, k, v)
                    torch.cuda.synchronize()
                    mem_mb = torch.cuda.max_memory_allocated() / (1024**2)
                    row += f"  {mem_mb:>12.1f}"
                except Exception:
                    row += f"  {'ERROR':>12}"
            print(row)

    # ---- Speedup summary ----
    print(f"\n  SPEEDUP vs Naive")
    header = f"  {'SeqLen':>8}"
    for name in variants:
        if name != "Naive":
            header += f"  {name:>12}"
    print(header)
    print(f"  {'-' * 8}" + f"  {'-' * 12}" * (len(variants) - 1))

    for T in seq_lens:
        if T not in latency_results:
            continue
        naive_ms = latency_results[T].get("Naive")
        if naive_ms is None:
            continue
        row = f"  {T:>8,}"
        for name in variants:
            if name == "Naive":
                continue
            ms = latency_results[T].get(name)
            if ms and ms > 0:
                row += f"  {naive_ms / ms:>11.2f}x"
            else:
                row += f"  {'N/A':>12}"
        print(row)

    print(f"\n  Key insight: FlashAttention achieves O(N) memory vs O(N²) for naive,")
    print(f"  and typically >2x speedup for long sequences due to reduced HBM traffic.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chapter 2: Attention Benchmark")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seq_lens", type=int, nargs="+", default=None)
    args = parser.parse_args()

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    if args.seq_lens is None:
        if "cuda" in args.device:
            args.seq_lens = [128, 256, 512, 1024, 2048, 4096]
        else:
            args.seq_lens = [64, 128, 256, 512]

    run_benchmark(
        seq_lens=args.seq_lens,
        batch=args.batch,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        dtype=dtype,
        device=args.device,
        warmup=args.warmup,
        steps=args.steps,
    )
