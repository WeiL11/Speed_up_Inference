"""
Chapter 2 — Efficient Attention
File: jit_attention.py

torch.jit.script version of scaled dot-product attention.

The JIT compiler fuses elementwise operations (scale, mask fill, softmax)
into fewer GPU kernels, reducing kernel launch overhead and memory traffic.
The full N×N matrix is still materialized — the algorithmic complexity is
identical to naive_attention — but constant-factor overhead is reduced.

Contents:
  - jit_attention()           : @torch.jit.script fused attention
  - JITSelfAttention          : nn.Module wrapping jit_attention
  - benchmark_jit_vs_naive()  : side-by-side comparison
"""

import sys
import math
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.model_loader import TransformerConfig
from utils.benchmarking import benchmark_fn, BenchmarkResult


# ---------------------------------------------------------------------------
# JIT-compiled attention
# ---------------------------------------------------------------------------

@torch.jit.script
def jit_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    """
    JIT-compiled scaled dot-product attention.

    Same algorithm as naive_attention but decorated with @torch.jit.script,
    which lets the PyTorch JIT compiler:
      1. Fuse elementwise ops (divide by scale, mask fill) into a single kernel
      2. Eliminate Python interpreter overhead in the hot loop
      3. Optimize memory access patterns for the fused operations

    Args:
        q: (B, nH, T, hD) — queries
        k: (B, nH, T, hD) — keys
        v: (B, nH, T, hD) — values
        causal: apply causal mask

    Returns:
        output: (B, nH, T, hD)

    Memory: O(B * nH * T²) — still materializes full attention matrix.
    """
    B, nH, T, hD = q.shape
    scale = math.sqrt(float(hD))

    # Q @ K^T → (B, nH, T, T)
    scores = torch.matmul(q, k.transpose(-2, -1)) / scale

    if causal:
        # Build causal mask (upper triangle = True where j > i)
        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=q.device),
            diagonal=1,
        )
        scores = scores.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )

    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)
    return output


# ---------------------------------------------------------------------------
# Non-JIT reference (for fair comparison)
# ---------------------------------------------------------------------------

def eager_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    """Identical logic to jit_attention but without @torch.jit.script."""
    B, nH, T, hD = q.shape
    scale = math.sqrt(float(hD))
    scores = torch.matmul(q, k.transpose(-2, -1)) / scale

    if causal:
        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=q.device),
            diagonal=1,
        )
        scores = scores.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )

    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)
    return output


# ---------------------------------------------------------------------------
# nn.Module wrapper
# ---------------------------------------------------------------------------

class JITSelfAttention(nn.Module):
    """
    Self-attention layer using JIT-compiled attention kernel.

    Drop-in replacement for NaiveSelfAttention with identical output
    but (potentially) lower latency due to JIT fusion.
    """

    def __init__(self, config: TransformerConfig, causal: bool = True):
        super().__init__()
        self.num_heads    = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim     = config.head_dim
        self.kv_groups    = config.kv_groups
        self.causal       = causal

        C   = config.hidden_dim
        nH  = config.num_heads
        nKV = config.num_kv_heads
        hD  = config.head_dim

        self.q_proj = nn.Linear(C, nH  * hD, bias=False)
        self.k_proj = nn.Linear(C, nKV * hD, bias=False)
        self.v_proj = nn.Linear(C, nKV * hD, bias=False)
        self.o_proj = nn.Linear(nH * hD, C,  bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        nH  = self.num_heads
        nKV = self.num_kv_heads
        hD  = self.head_dim

        q = self.q_proj(x).view(B, T, nH,  hD).transpose(1, 2)
        k = self.k_proj(x).view(B, T, nKV, hD).transpose(1, 2)
        v = self.v_proj(x).view(B, T, nKV, hD).transpose(1, 2)

        if self.kv_groups > 1:
            k = k.repeat_interleave(self.kv_groups, dim=1)
            v = v.repeat_interleave(self.kv_groups, dim=1)

        context = jit_attention(q, k, v, self.causal)
        context = context.transpose(1, 2).contiguous().view(B, T, nH * hD)
        return self.o_proj(context)


# ---------------------------------------------------------------------------
# JIT inspection: what does the compiler fuse?
# ---------------------------------------------------------------------------

def inspect_jit_graph():
    """Show the optimized JIT graph to see which ops are fused."""
    print("\n  JIT Graph (optimized):")
    print("  " + "-" * 60)
    graph = jit_attention.graph_for(
        torch.randn(1, 1, 4, 4),
        torch.randn(1, 1, 4, 4),
        torch.randn(1, 1, 4, 4),
        True,
    )
    # Print a simplified view
    graph_str = str(graph)
    for line in graph_str.split("\n")[:30]:
        print(f"  {line}")
    if graph_str.count("\n") > 30:
        print(f"  ... ({graph_str.count(chr(10)) - 30} more lines)")
    print("  " + "-" * 60)


# ---------------------------------------------------------------------------
# Benchmark: JIT vs eager (non-JIT) attention
# ---------------------------------------------------------------------------

def benchmark_jit_vs_naive(
    seq_lens: list = None,
    batch: int = 4,
    num_heads: int = 8,
    head_dim: int = 64,
    device: str = None,
    dtype: torch.dtype = torch.float32,
    warmup: int = 5,
    steps: int = 20,
) -> list:
    """
    Compare JIT-compiled attention vs eager (non-JIT) attention.

    Returns list of dicts with timing for both variants at each seq length.
    """
    if seq_lens is None:
        seq_lens = [128, 256, 512, 1024, 2048]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    has_cuda = torch.cuda.is_available() and "cuda" in device

    print(f"\n{'=' * 72}")
    print(f"  JIT vs Eager Attention Benchmark")
    print(f"  device={device}, batch={batch}, nH={num_heads}, hD={head_dim}, dtype={dtype}")
    print(f"{'=' * 72}")
    print(f"\n  {'SeqLen':>8}  {'Eager (ms)':>12}  {'JIT (ms)':>12}  {'Speedup':>10}")
    print(f"  {'-' * 8}  {'-' * 12}  {'-' * 12}  {'-' * 10}")

    results = []

    for T in seq_lens:
        if not has_cuda and T > 1024:
            print(f"  {T:>8}   (skipped — CPU only)")
            continue

        q = torch.randn(batch, num_heads, T, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch, num_heads, T, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch, num_heads, T, head_dim, device=device, dtype=dtype)

        # Benchmark eager
        def run_eager():
            return eager_attention(q, k, v, causal=True)

        eager_result = benchmark_fn(
            run_eager, warmup_steps=warmup, measure_steps=steps,
            name=f"eager_T{T}", sync_cuda=has_cuda, track_memory=False,
        )

        # Benchmark JIT
        def run_jit():
            return jit_attention(q, k, v, causal=True)

        jit_result = benchmark_fn(
            run_jit, warmup_steps=warmup, measure_steps=steps,
            name=f"jit_T{T}", sync_cuda=has_cuda, track_memory=False,
        )

        speedup = eager_result.mean_ms / jit_result.mean_ms if jit_result.mean_ms > 0 else 0

        print(f"  {T:>8,}  {eager_result.mean_ms:>12.2f}  {jit_result.mean_ms:>12.2f}  {speedup:>9.2f}x")

        results.append({
            "seq_len": T,
            "eager_ms": eager_result.mean_ms,
            "jit_ms": jit_result.mean_ms,
            "speedup": speedup,
        })

    return results


# ---------------------------------------------------------------------------
# Correctness verification
# ---------------------------------------------------------------------------

def _verify_jit_attention():
    """Verify JIT attention matches F.scaled_dot_product_attention."""
    B, nH, T, hD = 2, 4, 32, 16
    q = torch.randn(B, nH, T, hD)
    k = torch.randn(B, nH, T, hD)
    v = torch.randn(B, nH, T, hD)

    out_jit = jit_attention(q, k, v, causal=True)
    out_ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    max_diff = (out_jit - out_ref).abs().max().item()
    ok = max_diff < 1e-5
    print(f"  Verification: max |jit - F.sdpa| = {max_diff:.2e}  "
          f"({'PASS' if ok else 'FAIL'})")
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 72)
    print("  Chapter 2: JIT-Compiled Attention")
    print("=" * 72)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Running on: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")

    print("\n  [1] Correctness check vs F.scaled_dot_product_attention")
    _verify_jit_attention()

    print("\n  [2] JIT graph inspection")
    inspect_jit_graph()

    print("\n  [3] JIT vs Eager benchmark")
    seq_lens = [128, 256, 512, 1024, 2048] if device == "cuda" else [64, 128, 256, 512]
    benchmark_jit_vs_naive(seq_lens=seq_lens, device=device)

    print("\n  Key takeaway: JIT fusion reduces kernel launch overhead for")
    print("  elementwise ops (scale, mask, softmax). The algorithmic complexity")
    print("  is still O(N²) — FlashAttention-2 is needed to fix that.")
