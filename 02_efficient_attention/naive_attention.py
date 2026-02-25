"""
Chapter 2 — Efficient Attention
File: naive_attention.py

Naive scaled dot-product attention implemented FROM SCRATCH using pure PyTorch
matmuls. The full N×N attention matrix is explicitly materialized in memory.

This is the baseline implementation that demonstrates O(N²) memory scaling,
which FlashAttention-2 (triton_flash_attention.py) eliminates.

Contents:
  - naive_attention()      : pure functional attention (no F.scaled_dot_product_attention)
  - NaiveSelfAttention     : nn.Module wrapping naive_attention
  - benchmark()            : sweep over sequence lengths showing O(N²) scaling
"""

import sys
import math
import time
from pathlib import Path
from typing import Optional

# Allow imports from utils/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.model_loader import TransformerConfig
from utils.benchmarking import benchmark_fn, BenchmarkResult, cuda_memory_tracker


# ---------------------------------------------------------------------------
# Core naive attention function
# ---------------------------------------------------------------------------

def naive_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Naive scaled dot-product attention — materializes the full N×N matrix.

    Args:
        q: Query tensor  of shape (B, nH, T, hD)
        k: Key tensor    of shape (B, nH, T, hD)  (or (B, nKV, T, hD) before expand)
        v: Value tensor  of shape (B, nH, T, hD)
        causal: If True, apply causal (autoregressive) mask so position i
                cannot attend to position j > i.
        attn_mask: Optional additive mask of shape broadcastable to (B, nH, T, T).
                   Use -inf where attention should be blocked.

    Returns:
        output: (B, nH, T, hD)

    Memory: O(B * nH * T²) — the attention weight matrix is fully materialized.
    FLOPs:  O(B * nH * T² * hD) — two matmuls of this cost.
    """
    # q, k, v: (B, nH, T, hD)
    B, nH, T, hD = q.shape

    # -------------------------------------------------------------------
    # Step 1: Attention scores
    # scores[b, h, i, j] = dot(q[b,h,i,:], k[b,h,j,:]) / sqrt(hD)
    # q: (B, nH, T, hD) @ k^T: (B, nH, hD, T) -> (B, nH, T, T)
    # -------------------------------------------------------------------
    scale = math.sqrt(hD)
    scores = torch.matmul(q, k.transpose(-2, -1)) / scale   # (B, nH, T, T)

    # -------------------------------------------------------------------
    # Step 2: Apply causal mask
    # Upper triangle (j > i) is set to -inf so softmax drives those
    # positions to 0. We generate the mask on-the-fly to avoid storing it.
    # -------------------------------------------------------------------
    if causal:
        # torch.triu returns upper-triangular part; diagonal=1 means j > i
        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=q.device),
            diagonal=1,
        )
        # Expand mask to broadcast: (1, 1, T, T) -> (B, nH, T, T)
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    # -------------------------------------------------------------------
    # Step 3: Optional additive mask (e.g. padding mask)
    # -------------------------------------------------------------------
    if attn_mask is not None:
        scores = scores + attn_mask

    # -------------------------------------------------------------------
    # Step 4: Softmax over the last dimension (key dimension)
    # Converts raw scores to probability distribution over keys.
    # -------------------------------------------------------------------
    attn_weights = torch.softmax(scores, dim=-1)   # (B, nH, T, T)

    # -------------------------------------------------------------------
    # Step 5: Weighted sum of values
    # attn_weights: (B, nH, T, T) @ v: (B, nH, T, hD) -> (B, nH, T, hD)
    # -------------------------------------------------------------------
    output = torch.matmul(attn_weights, v)          # (B, nH, T, hD)

    return output


# ---------------------------------------------------------------------------
# nn.Module wrapper
# ---------------------------------------------------------------------------

class NaiveSelfAttention(nn.Module):
    """
    Self-attention layer that uses naive_attention (materializes full N×N matrix).

    Designed as a drop-in replacement for SimpleTransformer's SelfAttention
    for comparison purposes.  Supports MHA and GQA.

    Args:
        config: TransformerConfig with hidden_dim, num_heads, num_kv_heads, head_dim.
        causal: Whether to apply causal masking.
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
        """
        Args:
            x: (B, T, C)
        Returns:
            (B, T, C)
        """
        B, T, _ = x.shape
        nH  = self.num_heads
        nKV = self.num_kv_heads
        hD  = self.head_dim

        q = self.q_proj(x).view(B, T, nH,  hD).transpose(1, 2)   # (B, nH,  T, hD)
        k = self.k_proj(x).view(B, T, nKV, hD).transpose(1, 2)   # (B, nKV, T, hD)
        v = self.v_proj(x).view(B, T, nKV, hD).transpose(1, 2)   # (B, nKV, T, hD)

        # GQA: expand K/V to match num_heads
        if self.kv_groups > 1:
            k = k.repeat_interleave(self.kv_groups, dim=1)        # (B, nH, T, hD)
            v = v.repeat_interleave(self.kv_groups, dim=1)

        # Naive attention: materializes (B, nH, T, T)
        context = naive_attention(q, k, v, causal=self.causal)    # (B, nH, T, hD)

        # Reshape and project back
        context = context.transpose(1, 2).contiguous().view(B, T, nH * hD)
        return self.o_proj(context)


# ---------------------------------------------------------------------------
# Benchmark: time & memory vs sequence length
# ---------------------------------------------------------------------------

def benchmark(
    seq_lens: list = None,
    batch: int = 4,
    num_heads: int = 8,
    head_dim: int = 64,
    device: str = None,
    dtype: torch.dtype = torch.float32,
    warmup: int = 3,
    steps: int = 10,
) -> list:
    """
    Benchmark naive_attention across multiple sequence lengths.

    Shows:
      - Mean latency (ms) vs T
      - Peak memory (MB) vs T
      - Demonstrates O(N²) scaling clearly

    Args:
        seq_lens : list of sequence lengths to test
        batch    : batch size
        num_heads: number of attention heads
        head_dim : dimension per head
        device   : torch device string (auto-detects CUDA if None)
        dtype    : tensor dtype
        warmup   : warmup iterations before timing
        steps    : timed iterations per configuration

    Returns:
        List of dicts with keys: seq_len, mean_ms, memory_mb
    """
    if seq_lens is None:
        seq_lens = [128, 256, 512, 1024, 2048]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    has_cuda = torch.cuda.is_available() and "cuda" in device

    print(f"\n{'=' * 65}")
    print(f"  Naive Attention Benchmark")
    print(f"  device={device}, batch={batch}, nH={num_heads}, hD={head_dim}")
    print(f"  dtype={dtype}")
    print(f"{'=' * 65}")
    print(f"\n  {'SeqLen':>8}  {'Mean (ms)':>12}  {'Memory (MB)':>14}  {'Ratio(T²)':>12}")
    print(f"  {'-' * 8}  {'-' * 12}  {'-' * 14}  {'-' * 12}")

    results = []
    prev_ms = None

    for T in seq_lens:
        # Build fresh tensors for each run
        def make_tensors():
            return (
                torch.randn(batch, num_heads, T, head_dim, device=device, dtype=dtype),
                torch.randn(batch, num_heads, T, head_dim, device=device, dtype=dtype),
                torch.randn(batch, num_heads, T, head_dim, device=device, dtype=dtype),
            )

        # Skip very large T on CPU to avoid very long waits
        if not has_cuda and T > 1024:
            print(f"  {T:>8}   (skipped — CPU only, T>1024 too slow)")
            continue

        # Warmup
        q, k, v = make_tensors()
        for _ in range(warmup):
            _ = naive_attention(q, k, v, causal=True)
            if has_cuda:
                torch.cuda.synchronize()

        # Memory reset
        if has_cuda:
            torch.cuda.reset_peak_memory_stats()

        # Timed runs
        q, k, v = make_tensors()
        times_ms = []
        for _ in range(steps):
            if has_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = naive_attention(q, k, v, causal=True)
            if has_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

        import statistics
        mean_ms = statistics.mean(times_ms)

        # Peak memory
        if has_cuda:
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            # Estimate analytically: Q+K+V+(B,nH,T,T)+out
            bpe = 4  # float32
            mem_mb = (
                batch * num_heads * T * head_dim * 4   # Q, K, V, out
                + batch * num_heads * T * T             # attn_weights
            ) * bpe / (1024 ** 2)

        # Quadratic scaling ratio: compare to previous T (doubled each time)
        if prev_ms is not None and prev_ms > 0:
            ratio = mean_ms / prev_ms
            ratio_str = f"{ratio:.2f}x"
        else:
            ratio_str = "  --"
        prev_ms = mean_ms

        print(f"  {T:>8,}  {mean_ms:>12.2f}  {mem_mb:>14.1f}  {ratio_str:>12}")
        results.append({
            "seq_len": T,
            "mean_ms": mean_ms,
            "memory_mb": mem_mb,
        })

    print(f"\n  Expected ratio for O(T²): ~4.0x when T doubles.")
    print(f"  (Deviations occur for small T due to kernel launch overhead.)\n")

    return results


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

def _verify_naive_attention():
    """Verify naive_attention matches F.scaled_dot_product_attention."""
    B, nH, T, hD = 2, 4, 32, 16
    q = torch.randn(B, nH, T, hD)
    k = torch.randn(B, nH, T, hD)
    v = torch.randn(B, nH, T, hD)

    out_naive = naive_attention(q, k, v, causal=True)

    # Reference: PyTorch's fused SDPA
    out_ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    max_diff = (out_naive - out_ref).abs().max().item()
    print(f"  Verification: max |naive - F.sdpa| = {max_diff:.2e}  "
          f"({'PASS' if max_diff < 1e-5 else 'FAIL'})")
    return max_diff < 1e-5


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("  Chapter 2: Naive Attention (O(N²) Baseline)")
    print("=" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Running on: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")

    print("\n  [1] Correctness check vs F.scaled_dot_product_attention")
    ok = _verify_naive_attention()

    print("\n  [2] Shape demonstration  (B=1, nH=2, T=8, hD=4)")
    B, nH, T, hD = 1, 2, 8, 4
    q = torch.randn(B, nH, T, hD)
    k = torch.randn(B, nH, T, hD)
    v = torch.randn(B, nH, T, hD)
    scale = math.sqrt(hD)
    scores = torch.matmul(q, k.transpose(-2, -1)) / scale
    print(f"  scores shape:  {tuple(scores.shape)}  (B, nH, T, T)")
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    scores_masked = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    attn_w = torch.softmax(scores_masked, dim=-1)
    print(f"  attn_w shape:  {tuple(attn_w.shape)}  (B, nH, T, T) — FULL N×N materialized")
    out = torch.matmul(attn_w, v)
    print(f"  output shape:  {tuple(out.shape)}  (B, nH, T, hD)")

    print("\n  [3] O(N²) scaling benchmark")
    # Use shorter seq_lens if no CUDA
    seq_lens = [128, 256, 512, 1024, 2048] if device == "cuda" else [64, 128, 256, 512]
    benchmark(
        seq_lens=seq_lens,
        batch=4,
        num_heads=8,
        head_dim=64,
        device=device,
    )
