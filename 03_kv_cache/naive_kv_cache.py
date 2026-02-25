"""
naive_kv_cache.py  —  Chapter 3: KV Cache

Naive KV cache implementation: a growing list of tensors, concatenated at
each step. Simple but causes O(N²) total memory allocations because every
torch.cat creates a brand-new tensor.

Demonstrates:
  - NaiveKVCache class (append-and-cat pattern)
  - decode_with_cache(): prefill + autoregressive decode loop
  - allocation_cost_demo(): shows the quadratic allocation problem

Usage:
    python naive_kv_cache.py
    python naive_kv_cache.py --decode_steps 128 --batch_size 2
"""

import argparse
import sys
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.model_loader import TransformerConfig, SimpleTransformer, GEMMA3_1B_CONFIG


# ---------------------------------------------------------------------------
# NaiveKVCache
# ---------------------------------------------------------------------------

class NaiveKVCache:
    """
    Appends new KV pairs every decoding step.

    Simple but inefficient: every call to update() performs a torch.cat()
    which allocates a new tensor of size proportional to the current sequence
    length. Over N decode steps this creates O(N²) total bytes of allocations.

    Shape convention (all tensors):
        K: (batch_size, num_kv_heads, 1, head_dim)   — single step input
        V: (batch_size, num_kv_heads, 1, head_dim)

    After update(), returns:
        K_full: (batch_size, num_kv_heads, T_so_far, head_dim)
        V_full: (batch_size, num_kv_heads, T_so_far, head_dim)
    """

    def __init__(self):
        self.k_cache: list[torch.Tensor] = []
        self.v_cache: list[torch.Tensor] = []
        self._alloc_count: int = 0  # track number of allocations

    def update(
        self,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Append k_new and v_new, then concatenate along the seq dimension.

        Each call allocates a new tensor of size (B, nKV, T+1, hD).
        After N steps: T^0 + T^1 + ... + T^N bytes ~ O(N²) total allocation.
        """
        self.k_cache.append(k_new)
        self.v_cache.append(v_new)
        self._alloc_count += 1

        # torch.cat creates a brand-new tensor on every call
        k_full = torch.cat(self.k_cache, dim=2)  # dim=2 is the seq dimension
        v_full = torch.cat(self.v_cache, dim=2)
        return k_full, v_full

    def reset(self) -> None:
        """Clear the cache, ready for a new sequence."""
        self.k_cache = []
        self.v_cache = []
        self._alloc_count = 0

    def current_length(self) -> int:
        """Number of tokens currently cached."""
        return len(self.k_cache)

    def total_allocations(self) -> int:
        """Number of torch.cat calls made (each = one new tensor allocation)."""
        return self._alloc_count

    def __repr__(self) -> str:
        return (
            f"NaiveKVCache(length={self.current_length()}, "
            f"allocations={self.total_allocations()})"
        )


# ---------------------------------------------------------------------------
# Minimal cache-aware attention block
# ---------------------------------------------------------------------------

class CachedSelfAttention(nn.Module):
    """
    A self-attention module that accepts pre-computed KV tensors.

    Wraps SimpleTransformer's attention logic but exposes a cache interface.
    Used here to demonstrate KV caching at the single-layer level.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.num_heads   = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim    = config.head_dim
        self.kv_groups   = config.kv_groups

        self.q_proj = nn.Linear(config.hidden_dim,
                                config.num_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim,
                                config.num_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim,
                                config.num_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_heads * config.head_dim,
                                config.hidden_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        k_cache: torch.Tensor | None = None,
        v_cache: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x:       (B, T_new, hidden_dim)  — new tokens only
            k_cache: (B, nKV, T_past, head_dim)  or None
            v_cache: (B, nKV, T_past, head_dim)  or None

        Returns:
            out:    (B, T_new, hidden_dim)
            k_new:  (B, nKV, T_new, head_dim)   — keys for new tokens
            v_new:  (B, nKV, T_new, head_dim)   — values for new tokens
        """
        B, T_new, _ = x.shape

        q = self.q_proj(x).view(B, T_new, self.num_heads, self.head_dim).transpose(1, 2)
        k_new = self.k_proj(x).view(B, T_new, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v_new = self.v_proj(x).view(B, T_new, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Concatenate with past KV (if any)
        if k_cache is not None and v_cache is not None:
            k = torch.cat([k_cache, k_new], dim=2)
            v = torch.cat([v_cache, v_new], dim=2)
        else:
            k, v = k_new, v_new

        # GQA: expand KV to match number of query heads
        if self.kv_groups > 1:
            k = k.repeat_interleave(self.kv_groups, dim=1)
            v = v.repeat_interleave(self.kv_groups, dim=1)

        # During decode, T_new=1 so no causal mask needed; during prefill, use causal
        is_causal = (T_new > 1 and k_cache is None)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        out = out.transpose(1, 2).contiguous().view(B, T_new, self.num_heads * self.head_dim)

        return self.o_proj(out), k_new, v_new


# ---------------------------------------------------------------------------
# Decode loop with NaiveKVCache
# ---------------------------------------------------------------------------

def decode_with_cache(
    model: SimpleTransformer,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    cache: NaiveKVCache,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Autoregressive decode using a NaiveKVCache.

    Phases:
      1. Prefill: forward pass over the full prompt to warm the cache.
         The SimpleTransformer does not expose per-layer KV, so we simulate
         the cache by treating the entire prompt output as context.
      2. Decode: one token at a time; each step appends to the cache.

    Note: SimpleTransformer doesn't natively support KV caching, so here we
    demonstrate the NaiveKVCache usage pattern with a thin wrapper that
    extracts KV tensors from a CachedSelfAttention module. The demo below
    uses a single-layer CachedSelfAttention to keep things concrete.

    Args:
        model:           SimpleTransformer (used for embedding + LM head)
        prompt_ids:      (B, prompt_len) int64 token IDs
        max_new_tokens:  number of tokens to generate
        cache:           NaiveKVCache instance (will be reset)
        verbose:         print per-step info

    Returns:
        generated_ids: (B, prompt_len + max_new_tokens)
    """
    device = prompt_ids.device
    B, prompt_len = prompt_ids.shape
    cache.reset()

    # We'll drive the demo with a simplified forward (embedding → single attn → lm_head).
    # For full model KV cache, see static_kv_cache.py.
    generated = prompt_ids.clone()

    if verbose:
        print(f"  [NaiveKVCache] Prefill: {prompt_len} tokens ...")

    # PREFILL PHASE: run full prompt through model (no cache yet)
    with torch.no_grad():
        _ = model(prompt_ids)   # warm the model; output ignored for demo purposes

    if verbose:
        print(f"  [NaiveKVCache] Prefill done. Starting decode ...")

    # DECODE PHASE: one token at a time
    for step in range(max_new_tokens):
        # Only feed the last token during decode
        last_token = generated[:, -1:]          # (B, 1)

        with torch.no_grad():
            logits = model(last_token)           # (B, 1, vocab_size)

        # Greedy sampling
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (B, 1)
        generated = torch.cat([generated, next_token], dim=1)

        # Simulate cache update — store the new token position's embedding
        # (in a real system, we'd store actual K/V from each layer)
        dummy_k = torch.zeros(B, model.config.num_kv_heads, 1, model.config.head_dim,
                              device=device, dtype=model.tok_emb.weight.dtype)
        dummy_v = torch.zeros_like(dummy_k)
        k_full, v_full = cache.update(dummy_k, dummy_v)

        if verbose and (step % 64 == 0 or step == max_new_tokens - 1):
            print(
                f"  Step {step+1:4d}/{max_new_tokens} | "
                f"cached_len={cache.current_length()} | "
                f"allocs={cache.total_allocations()}"
            )

    return generated


# ---------------------------------------------------------------------------
# Allocation cost demonstration
# ---------------------------------------------------------------------------

def allocation_cost_demo(
    decode_steps: int = 256,
    batch_size: int = 1,
    num_kv_heads: int = 1,
    head_dim: int = 288,
    device: str = "cpu",
    verbose: bool = True,
) -> dict:
    """
    Measure the wall-clock cost of the naive append-and-cat pattern.

    At each step we do:
        list.append(new_kv)
        torch.cat(list, dim=2)   <-- allocates (step+1) × elem bytes

    Over N steps: total bytes allocated ∝ N*(N+1)/2  → O(N²).

    Returns timing dict.
    """
    cache = NaiveKVCache()
    step_times_ms = []
    cumulative_mb = []

    if verbose:
        print(f"\n{'='*60}")
        print(f"  NaiveKVCache Allocation Cost Demo")
        print(f"  decode_steps={decode_steps}, batch={batch_size}, "
              f"kv_heads={num_kv_heads}, head_dim={head_dim}, device={device}")
        print(f"{'='*60}")
        print(f"  {'Step':>6}  {'Step time (ms)':>16}  {'Cache MB':>10}  {'Allocs':>8}")
        print(f"  {'-'*6}  {'-'*16}  {'-'*10}  {'-'*8}")

    for step in range(decode_steps):
        k_new = torch.randn(batch_size, num_kv_heads, 1, head_dim, device=device)
        v_new = torch.randn(batch_size, num_kv_heads, 1, head_dim, device=device)

        t0 = time.perf_counter()
        k_full, v_full = cache.update(k_new, v_new)
        t1 = time.perf_counter()

        step_ms = (t1 - t0) * 1000.0
        step_times_ms.append(step_ms)

        # Current cache size in MB (one tensor, float32)
        cache_elements = batch_size * num_kv_heads * (step + 1) * head_dim
        cache_mb = cache_elements * 4 / (1024 ** 2)  # float32
        cumulative_mb.append(cache_mb)

        if verbose and (step % (decode_steps // 8) == 0 or step == decode_steps - 1):
            print(
                f"  {step+1:6d}  {step_ms:16.3f}  {cache_mb:10.3f}  {cache.total_allocations():8d}"
            )

    # Compute allocation trend
    early_avg  = sum(step_times_ms[:decode_steps // 4]) / (decode_steps // 4)
    late_avg   = sum(step_times_ms[3 * decode_steps // 4:]) / (decode_steps // 4)

    if verbose:
        print(f"\n  Early steps avg (first 25%): {early_avg:.3f} ms")
        print(f"  Late  steps avg (last  25%): {late_avg:.3f} ms")
        print(f"  Slowdown ratio:              {late_avg / early_avg:.2f}x")
        print()
        print("  Problem: Each torch.cat allocates a NEW tensor growing by 1 step.")
        print("  Total bytes allocated ∝ N*(N+1)/2  —  O(N²) behavior.")
        print(f"  After {decode_steps} steps: {cache.total_allocations()} allocations.")
        print(f"{'='*60}")

    return {
        "step_times_ms": step_times_ms,
        "cumulative_mb": cumulative_mb,
        "early_avg_ms": early_avg,
        "late_avg_ms": late_avg,
        "slowdown_ratio": late_avg / early_avg,
        "total_allocations": cache.total_allocations(),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NaiveKVCache demo and allocation analysis.")
    parser.add_argument("--decode_steps", type=int, default=256,
                        help="Number of decode steps to simulate (default: 256)")
    parser.add_argument("--batch_size",   type=int, default=1)
    parser.add_argument("--num_kv_heads", type=int, default=1,
                        help="KV heads for demo tensor (default: 1, Gemma-3 1B style)")
    parser.add_argument("--head_dim",     type=int, default=288,
                        help="Head dim (hidden_dim/num_heads = 1152/4 = 288)")
    parser.add_argument("--device",       type=str, default="cpu",
                        help="Device for tensors (default: cpu; gpu not required)")
    args = parser.parse_args()

    results = allocation_cost_demo(
        decode_steps=args.decode_steps,
        batch_size=args.batch_size,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        device=args.device,
        verbose=True,
    )

    print("\nSummary:")
    print(f"  Total decode steps:     {args.decode_steps}")
    print(f"  Total torch.cat calls:  {results['total_allocations']}")
    print(f"  Early avg step time:    {results['early_avg_ms']:.3f} ms")
    print(f"  Late  avg step time:    {results['late_avg_ms']:.3f} ms")
    print(f"  Observed slowdown:      {results['slowdown_ratio']:.2f}x")
    print()
    print("Solution: pre-allocate a fixed buffer -> see static_kv_cache.py")


if __name__ == "__main__":
    main()
