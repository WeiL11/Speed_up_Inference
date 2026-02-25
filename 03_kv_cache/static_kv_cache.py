"""
Chapter 3 — KV Cache Management
File: static_kv_cache.py

Pre-allocated static KV cache — allocates the FULL max_seq_len buffer once
at initialization. No dynamic allocation during generation.

Advantages over naive (append-and-cat) approach:
  - Zero allocation after init → no GC pauses, deterministic latency
  - Contiguous memory layout → better GPU memory access patterns
  - Predictable peak memory → easier capacity planning

Disadvantages:
  - Wastes memory for short sequences (allocated but unused)
  - max_seq_len must be known in advance
  - No sharing across requests with different lengths

Contents:
  - StaticKVCache      : fixed-buffer cache with position tracking
  - CachedSelfAttention: attention module using StaticKVCache
  - decode_with_cache  : prefill + autoregressive generation demo
  - efficiency_demo    : compare allocation behavior vs naive
"""

import sys
import time
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.model_loader import TransformerConfig, create_model
from utils.benchmarking import benchmark_fn


# ---------------------------------------------------------------------------
# Static KV Cache
# ---------------------------------------------------------------------------

class StaticKVCache:
    """
    Pre-allocated KV cache with fixed max_seq_len buffer.

    The full (batch, num_kv_heads, max_seq_len, head_dim) tensor is allocated
    once. A position counter tracks how many tokens have been stored.

    Args:
        batch_size: batch dimension
        num_kv_heads: number of KV heads
        max_seq_len: maximum sequence length (pre-allocated)
        head_dim: dimension per head
        device: torch device
        dtype: tensor dtype
    """

    def __init__(
        self,
        batch_size: int,
        num_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        self.max_seq_len = max_seq_len
        self.pos = 0  # next write position

        # Pre-allocate full buffers — zeros, overwritten during generation
        self.k_cache = torch.zeros(
            batch_size, num_kv_heads, max_seq_len, head_dim,
            device=device, dtype=dtype,
        )
        self.v_cache = torch.zeros(
            batch_size, num_kv_heads, max_seq_len, head_dim,
            device=device, dtype=dtype,
        )

    def update(self, k_new: torch.Tensor, v_new: torch.Tensor):
        """
        Write new K/V into the cache at the current position.

        Args:
            k_new: (B, nKV, new_tokens, hD)
            v_new: (B, nKV, new_tokens, hD)

        Returns:
            (k_valid, v_valid): views of the cache up to current position
        """
        new_tokens = k_new.shape[2]
        end_pos = self.pos + new_tokens
        assert end_pos <= self.max_seq_len, (
            f"Cache overflow: pos={self.pos} + new={new_tokens} > max={self.max_seq_len}"
        )

        # Write into pre-allocated buffer (no allocation!)
        self.k_cache[:, :, self.pos:end_pos, :] = k_new
        self.v_cache[:, :, self.pos:end_pos, :] = v_new
        self.pos = end_pos

        # Return valid portion (view, no copy)
        return self.k_cache[:, :, :self.pos, :], self.v_cache[:, :, :self.pos, :]

    def reset(self):
        """Reset position counter (buffer remains allocated)."""
        self.pos = 0

    @property
    def seq_len(self) -> int:
        return self.pos

    def memory_mb(self) -> float:
        """Total allocated memory in MB (including unused portion)."""
        bpe = self.k_cache.element_size()
        total_bytes = 2 * self.k_cache.numel() * bpe  # K + V
        return total_bytes / (1024 ** 2)

    def utilization(self) -> float:
        """Fraction of allocated cache actually in use."""
        return self.pos / self.max_seq_len if self.max_seq_len > 0 else 0.0


# ---------------------------------------------------------------------------
# Cached self-attention using static cache
# ---------------------------------------------------------------------------

class CachedSelfAttention(nn.Module):
    """
    Attention layer that uses StaticKVCache for autoregressive decoding.

    During prefill: processes all tokens at once, fills cache.
    During decode: processes one token, reads full cache.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.num_heads    = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim     = config.head_dim
        self.kv_groups    = config.kv_groups

        C   = config.hidden_dim
        nH  = config.num_heads
        nKV = config.num_kv_heads
        hD  = config.head_dim

        self.q_proj = nn.Linear(C, nH  * hD, bias=False)
        self.k_proj = nn.Linear(C, nKV * hD, bias=False)
        self.v_proj = nn.Linear(C, nKV * hD, bias=False)
        self.o_proj = nn.Linear(nH * hD, C,  bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cache: StaticKVCache = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        nH  = self.num_heads
        nKV = self.num_kv_heads
        hD  = self.head_dim

        q = self.q_proj(x).view(B, T, nH,  hD).transpose(1, 2)
        k = self.k_proj(x).view(B, T, nKV, hD).transpose(1, 2)
        v = self.v_proj(x).view(B, T, nKV, hD).transpose(1, 2)

        if cache is not None:
            k, v = cache.update(k, v)

        # GQA expansion
        if self.kv_groups > 1:
            k = k.repeat_interleave(self.kv_groups, dim=1)
            v = v.repeat_interleave(self.kv_groups, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=(cache is None or T > 1))
        out = out.transpose(1, 2).contiguous().view(B, T, nH * hD)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# Generation demo
# ---------------------------------------------------------------------------

def decode_with_cache(
    batch_size: int = 2,
    prefill_len: int = 64,
    decode_steps: int = 128,
    max_seq_len: int = 256,
    hidden_dim: int = 512,
    num_heads: int = 8,
    num_kv_heads: int = 8,
    device: str = None,
):
    """
    Demonstrate prefill + autoregressive decode with static KV cache.

    Returns timing information for each decode step to show constant-time
    per-step allocation (unlike naive cache which gets slower).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    has_cuda = "cuda" in device

    config = TransformerConfig(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        num_layers=1,
        max_seq_len=max_seq_len,
    )
    hD = config.head_dim

    attn = CachedSelfAttention(config).to(device)
    attn.eval()

    cache = StaticKVCache(batch_size, num_kv_heads, max_seq_len, hD, device)

    print(f"\n  Static KV Cache Demo")
    print(f"  batch={batch_size}, prefill={prefill_len}, decode={decode_steps}")
    print(f"  max_seq_len={max_seq_len}, cache memory={cache.memory_mb():.1f} MB")
    print(f"  {'Step':>6}  {'Time (ms)':>10}  {'Cache pos':>10}  {'Util %':>8}")
    print(f"  {'-' * 6}  {'-' * 10}  {'-' * 10}  {'-' * 8}")

    step_times = []

    with torch.no_grad():
        # Prefill
        x = torch.randn(batch_size, prefill_len, hidden_dim, device=device)
        if has_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = attn(x, cache)
        if has_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        prefill_ms = (t1 - t0) * 1000
        print(f"  {'PF':>6}  {prefill_ms:>10.2f}  {cache.seq_len:>10}  {cache.utilization()*100:>7.1f}%")

        # Autoregressive decode
        for step in range(decode_steps):
            x = torch.randn(batch_size, 1, hidden_dim, device=device)
            if has_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = attn(x, cache)
            if has_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            ms = (t1 - t0) * 1000
            step_times.append(ms)

            if step % 32 == 0 or step == decode_steps - 1:
                print(f"  {step:>6}  {ms:>10.3f}  {cache.seq_len:>10}  {cache.utilization()*100:>7.1f}%")

    print(f"\n  Decode step stats:")
    import statistics
    print(f"    Mean:  {statistics.mean(step_times):.3f} ms")
    print(f"    Stdev: {statistics.stdev(step_times):.3f} ms")
    print(f"    Min:   {min(step_times):.3f} ms")
    print(f"    Max:   {max(step_times):.3f} ms")
    print(f"\n  Key point: step time is near-constant (no reallocation).")

    return step_times


# ---------------------------------------------------------------------------
# Efficiency analysis: static vs naive allocation cost
# ---------------------------------------------------------------------------

def efficiency_demo(device: str = None):
    """Compare allocation patterns of static vs naive cache."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    B, nKV, hD = 2, 8, 64
    max_seq = 512

    print(f"\n  Static vs Naive Allocation Cost")
    print(f"  B={B}, nKV={nKV}, hD={hD}, max_seq={max_seq}")

    # Static: one allocation up front
    t0 = time.perf_counter()
    cache = StaticKVCache(B, nKV, max_seq, hD, device)
    for step in range(max_seq):
        k_new = torch.randn(B, nKV, 1, hD, device=device)
        v_new = torch.randn(B, nKV, 1, hD, device=device)
        cache.update(k_new, v_new)
    if torch.cuda.is_available() and "cuda" in device:
        torch.cuda.synchronize()
    static_ms = (time.perf_counter() - t0) * 1000

    # Naive: concat every step (O(N²) total allocation)
    k_list, v_list = [], []
    t0 = time.perf_counter()
    for step in range(max_seq):
        k_new = torch.randn(B, nKV, 1, hD, device=device)
        v_new = torch.randn(B, nKV, 1, hD, device=device)
        k_list.append(k_new)
        v_list.append(v_new)
        # Concatenate every step (this is the expensive part)
        k_full = torch.cat(k_list, dim=2)
        v_full = torch.cat(v_list, dim=2)
    if torch.cuda.is_available() and "cuda" in device:
        torch.cuda.synchronize()
    naive_ms = (time.perf_counter() - t0) * 1000

    print(f"\n  Static cache ({max_seq} steps): {static_ms:.1f} ms")
    print(f"  Naive concat ({max_seq} steps): {naive_ms:.1f} ms")
    if naive_ms > 0:
        print(f"  Speedup: {naive_ms / static_ms:.1f}x")
    print(f"\n  Static: O(1) per step (write into pre-allocated buffer)")
    print(f"  Naive:  O(N) per step → O(N²) total (copy entire cache each step)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("  Chapter 3: Static KV Cache (Pre-Allocated)")
    print("=" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Running on: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")

    print("\n  [1] Cache properties")
    cache = StaticKVCache(2, 8, 2048, 64, device)
    print(f"  Allocated memory: {cache.memory_mb():.1f} MB")
    print(f"  Utilization: {cache.utilization()*100:.0f}% (empty)")
    k = torch.randn(2, 8, 64, 64, device=device)
    v = torch.randn(2, 8, 64, 64, device=device)
    cache.update(k, v)
    print(f"  After 64 tokens: {cache.utilization()*100:.1f}%")

    print("\n  [2] Decode with static cache")
    decode_with_cache(device=device)

    print("\n  [3] Allocation efficiency: static vs naive")
    efficiency_demo(device=device)
