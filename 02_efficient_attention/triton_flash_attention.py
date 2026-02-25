"""
Chapter 2 — Efficient Attention
File: triton_flash_attention.py

FlashAttention-2 implemented FROM SCRATCH in Triton.

Key ideas from Dao 2023 (FlashAttention-2):
  - Tile Q/K/V so the full N×N attention matrix is NEVER materialized in HBM
  - Online softmax: maintain running max + running sum for numerically stable
    softmax computed incrementally over K/V tiles
  - O(N) HBM memory instead of O(N²)
  - Significant wall-clock speedup from reduced HBM reads/writes

This file provides:
  - flash_attention_fwd_kernel : Triton kernel (forward pass)
  - flash_attention()          : Python wrapper dispatching the kernel
  - FlashSelfAttention         : nn.Module wrapper
  - benchmark()                : compare FlashAttention vs naive vs F.sdpa

Fallback: If Triton is not installed, a pure-PyTorch tiled implementation
is provided that demonstrates the same algorithmic idea (online softmax +
tiling) without GPU kernel-level optimization.
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
from utils.benchmarking import benchmark_fn

# Try to import Triton
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ===========================================================================
# Triton FlashAttention-2 kernel (forward only)
# ===========================================================================

if HAS_TRITON:

    @triton.jit
    def flash_attention_fwd_kernel(
        Q_ptr, K_ptr, V_ptr, Out_ptr,
        stride_qb, stride_qh, stride_qt, stride_qd,
        stride_kb, stride_kh, stride_kt, stride_kd,
        stride_vb, stride_vh, stride_vt, stride_vd,
        stride_ob, stride_oh, stride_ot, stride_od,
        seq_len: tl.constexpr,
        head_dim: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_D: tl.constexpr,
        causal: tl.constexpr,
    ):
        """
        FlashAttention-2 forward kernel.

        Each program instance computes one tile of the output for one
        (batch, head, query_block) triple.

        Algorithm (per query tile):
          1. Load Q tile from HBM to SRAM
          2. For each K/V tile:
             a. Load K tile, compute S = Q @ K^T / sqrt(d)
             b. Apply causal mask (if needed)
             c. Online softmax update: track running max and running sum
             d. Load V tile, accumulate O += softmax(S) @ V
          3. Write final O tile back to HBM

        Memory: O(BLOCK_T * head_dim) per program — never O(N²).
        """
        # Which (batch, head, query_block) this program handles
        pid_bh = tl.program_id(0)  # batch * num_heads + head
        pid_t  = tl.program_id(1)  # query block index

        batch_idx = pid_bh // stride_qh  # inferred from strides
        head_idx  = pid_bh % (stride_qb // stride_qh) if stride_qh > 0 else 0

        # Offsets for this query tile
        q_start = pid_t * BLOCK_T
        offs_t = q_start + tl.arange(0, BLOCK_T)   # query positions
        offs_d = tl.arange(0, BLOCK_D)              # head dim

        # Pointers to Q[batch, head, q_start:q_start+BLOCK_T, :]
        q_ptrs = (Q_ptr
                  + pid_bh * stride_qh * (stride_qb // stride_qh)
                  + offs_t[:, None] * stride_qt
                  + offs_d[None, :] * stride_qd)

        # Actually, let's use a simpler flat addressing approach
        # Q is laid out as (B, nH, T, hD) contiguous
        q_base = pid_bh * seq_len * head_dim
        q_ptrs = Q_ptr + q_base + offs_t[:, None] * head_dim + offs_d[None, :]

        # Load Q tile — mask out-of-bounds positions
        q_mask = offs_t[:, None] < seq_len
        q = tl.load(q_ptrs, mask=q_mask, other=0.0)

        # Scale factor
        scale = 1.0 / tl.sqrt(float(head_dim))
        q = q * scale

        # Initialize accumulators for online softmax
        # m_i = running row-wise max of scores (for numerical stability)
        # l_i = running row-wise sum of exp(scores - m_i)
        # o_i = running weighted sum of V
        m_i = tl.full([BLOCK_T], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_T], dtype=tl.float32)
        o_i = tl.zeros([BLOCK_T, BLOCK_D], dtype=tl.float32)

        # Iterate over K/V tiles
        kv_base = pid_bh * seq_len * head_dim
        num_kv_blocks = tl.cdiv(seq_len, BLOCK_T)

        # For causal: only iterate up to the block containing position q_start + BLOCK_T - 1
        if causal:
            kv_limit = tl.minimum(num_kv_blocks, tl.cdiv(q_start + BLOCK_T, BLOCK_T))
        else:
            kv_limit = num_kv_blocks

        for kv_block in range(0, num_kv_blocks):
            if kv_block >= kv_limit:
                break

            kv_start = kv_block * BLOCK_T
            offs_kv = kv_start + tl.arange(0, BLOCK_T)

            # Load K tile: (BLOCK_T, BLOCK_D)
            k_ptrs = K_ptr + kv_base + offs_kv[:, None] * head_dim + offs_d[None, :]
            k_mask = offs_kv[:, None] < seq_len
            k = tl.load(k_ptrs, mask=k_mask, other=0.0)

            # S = Q @ K^T → (BLOCK_T, BLOCK_T)
            s = tl.dot(q, tl.trans(k))

            # Causal mask: set s[i, j] = -inf where offs_t[i] < offs_kv[j]
            if causal:
                causal_mask = offs_t[:, None] < offs_kv[None, :]
                s = tl.where(causal_mask, float("-inf"), s)

            # Out-of-bounds mask
            oob_mask = offs_kv[None, :] >= seq_len
            s = tl.where(oob_mask, float("-inf"), s)

            # --- Online softmax update ---
            # New row max
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            # Correction factor for previous accumulation
            alpha = tl.exp(m_i - m_new)
            # exp(s - m_new) for current tile
            p = tl.exp(s - m_new[:, None])
            # Update running sum
            l_i = l_i * alpha + tl.sum(p, axis=1)
            # Rescale previous output accumulator
            o_i = o_i * alpha[:, None]

            # Load V tile and accumulate
            v_ptrs = V_ptr + kv_base + offs_kv[:, None] * head_dim + offs_d[None, :]
            v_mask = offs_kv[:, None] < seq_len
            v = tl.load(v_ptrs, mask=v_mask, other=0.0)

            o_i += tl.dot(p.to(v.dtype), v)
            m_i = m_new

        # Final normalization: o = o_i / l_i
        o_i = o_i / l_i[:, None]

        # Write output
        o_ptrs = Out_ptr + q_base + offs_t[:, None] * head_dim + offs_d[None, :]
        o_mask = offs_t[:, None] < seq_len
        tl.store(o_ptrs, o_i.to(Out_ptr.dtype.element_ty), mask=o_mask)


# ===========================================================================
# Pure-PyTorch tiled attention (fallback / educational)
# ===========================================================================

def tiled_flash_attention_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int = 64,
    causal: bool = True,
) -> torch.Tensor:
    """
    Pure-PyTorch implementation of the FlashAttention tiling algorithm.

    This demonstrates the ALGORITHM (online softmax + tiling) without
    actual kernel-level optimization. It's useful for understanding the
    math but won't be faster than naive attention in PyTorch.

    Args:
        q, k, v: (B, nH, T, hD)
        block_size: tile size for sequence dimension
        causal: apply causal mask

    Returns:
        output: (B, nH, T, hD)
    """
    B, nH, T, hD = q.shape
    scale = math.sqrt(hD)
    q = q / scale

    # Accumulators (always float32 for numerical stability)
    o = torch.zeros_like(q, dtype=torch.float32)
    m = torch.full((B, nH, T), float("-inf"), device=q.device, dtype=torch.float32)
    l = torch.zeros((B, nH, T), device=q.device, dtype=torch.float32)

    # Iterate over K/V blocks
    num_kv_blocks = math.ceil(T / block_size)

    for kv_block_idx in range(num_kv_blocks):
        kv_start = kv_block_idx * block_size
        kv_end = min(kv_start + block_size, T)

        k_block = k[:, :, kv_start:kv_end, :]  # (B, nH, bk, hD)
        v_block = v[:, :, kv_start:kv_end, :]

        # S = Q @ K_block^T → (B, nH, T, bk)
        s = torch.matmul(q.float(), k_block.float().transpose(-2, -1))

        # Causal mask
        if causal:
            q_pos = torch.arange(T, device=q.device).unsqueeze(1)
            k_pos = torch.arange(kv_start, kv_end, device=q.device).unsqueeze(0)
            causal_mask = q_pos < k_pos
            s.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # --- Online softmax ---
        m_new = torch.maximum(m, s.max(dim=-1).values)
        alpha = torch.exp(m - m_new)
        p = torch.exp(s - m_new.unsqueeze(-1))

        l = l * alpha + p.sum(dim=-1)
        o = o * alpha.unsqueeze(-1) + torch.matmul(p, v_block.float())
        m = m_new

    # Normalize
    o = o / l.unsqueeze(-1)
    return o.to(q.dtype)


# ===========================================================================
# Python wrapper dispatching to Triton or fallback
# ===========================================================================

def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    block_size: int = 64,
) -> torch.Tensor:
    """
    FlashAttention-2 forward pass.

    Uses Triton kernel when available, falls back to PyTorch tiled
    implementation otherwise.

    Args:
        q, k, v: (B, nH, T, hD) — must be contiguous
        causal: apply causal attention mask
        block_size: tile size (must be power of 2 for Triton)

    Returns:
        output: (B, nH, T, hD) — O(N) memory, not O(N²)
    """
    B, nH, T, hD = q.shape

    if HAS_TRITON and q.is_cuda and hD <= 128:
        # Use Triton kernel
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        out = torch.empty_like(q)

        BLOCK_T = min(block_size, T)
        BLOCK_D = hD  # head dim must fit in one tile

        grid = (B * nH, triton.cdiv(T, BLOCK_T))

        flash_attention_fwd_kernel[grid](
            q, k, v, out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            seq_len=T,
            head_dim=BLOCK_D,
            BLOCK_T=BLOCK_T,
            BLOCK_D=BLOCK_D,
            causal=causal,
        )
        return out
    else:
        return tiled_flash_attention_pytorch(q, k, v, block_size, causal)


# ===========================================================================
# nn.Module wrapper
# ===========================================================================

class FlashSelfAttention(nn.Module):
    """
    Self-attention using FlashAttention-2 (Triton or tiled PyTorch fallback).

    Memory: O(N) instead of O(N²) — enables much longer sequences.
    """

    def __init__(self, config: TransformerConfig, causal: bool = True, block_size: int = 64):
        super().__init__()
        self.num_heads    = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim     = config.head_dim
        self.kv_groups    = config.kv_groups
        self.causal       = causal
        self.block_size   = block_size

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

        context = flash_attention(q, k, v, self.causal, self.block_size)
        context = context.transpose(1, 2).contiguous().view(B, T, nH * hD)
        return self.o_proj(context)


# ===========================================================================
# Correctness verification
# ===========================================================================

def _verify_flash_attention():
    """Verify flash_attention matches F.scaled_dot_product_attention."""
    B, nH, T, hD = 2, 4, 64, 32
    q = torch.randn(B, nH, T, hD)
    k = torch.randn(B, nH, T, hD)
    v = torch.randn(B, nH, T, hD)

    if torch.cuda.is_available():
        q, k, v = q.cuda(), k.cuda(), v.cuda()

    out_flash = flash_attention(q, k, v, causal=True)
    out_ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    max_diff = (out_flash.float() - out_ref.float()).abs().max().item()
    # Tiled approach may have slightly higher error due to float32 accumulation order
    tol = 1e-3 if HAS_TRITON else 1e-4
    ok = max_diff < tol
    print(f"  Verification: max |flash - F.sdpa| = {max_diff:.2e}  "
          f"({'PASS' if ok else 'FAIL'})  (tol={tol:.0e})")
    return ok


# ===========================================================================
# Benchmark: Flash vs Naive vs F.sdpa
# ===========================================================================

def benchmark_all(
    seq_lens: list = None,
    batch: int = 2,
    num_heads: int = 8,
    head_dim: int = 64,
    device: str = None,
    warmup: int = 3,
    steps: int = 10,
):
    """Compare FlashAttention, naive attention, and F.sdpa across seq lengths."""
    from naive_attention import naive_attention

    if seq_lens is None:
        seq_lens = [128, 256, 512, 1024, 2048]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    has_cuda = "cuda" in device and torch.cuda.is_available()

    print(f"\n{'=' * 80}")
    print(f"  FlashAttention-2 vs Naive vs F.sdpa")
    print(f"  device={device}, batch={batch}, nH={num_heads}, hD={head_dim}")
    print(f"  Backend: {'Triton kernel' if HAS_TRITON and has_cuda else 'PyTorch tiled (fallback)'}")
    print(f"{'=' * 80}")
    print(f"\n  {'SeqLen':>8}  {'Naive (ms)':>12}  {'Flash (ms)':>12}  {'F.sdpa (ms)':>13}  {'Speedup':>10}")
    print(f"  {'-' * 8}  {'-' * 12}  {'-' * 12}  {'-' * 13}  {'-' * 10}")

    for T in seq_lens:
        if not has_cuda and T > 512:
            print(f"  {T:>8}   (skipped)")
            continue

        q = torch.randn(batch, num_heads, T, head_dim, device=device)
        k = torch.randn(batch, num_heads, T, head_dim, device=device)
        v = torch.randn(batch, num_heads, T, head_dim, device=device)

        def run_naive():
            return naive_attention(q, k, v, causal=True)

        def run_flash():
            return flash_attention(q, k, v, causal=True)

        def run_sdpa():
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)

        r_naive = benchmark_fn(run_naive, warmup_steps=warmup, measure_steps=steps,
                               name="naive", sync_cuda=has_cuda, track_memory=False)
        r_flash = benchmark_fn(run_flash, warmup_steps=warmup, measure_steps=steps,
                               name="flash", sync_cuda=has_cuda, track_memory=False)
        r_sdpa = benchmark_fn(run_sdpa, warmup_steps=warmup, measure_steps=steps,
                              name="sdpa", sync_cuda=has_cuda, track_memory=False)

        speedup = r_naive.mean_ms / r_flash.mean_ms if r_flash.mean_ms > 0 else 0

        print(f"  {T:>8,}  {r_naive.mean_ms:>12.2f}  {r_flash.mean_ms:>12.2f}"
              f"  {r_sdpa.mean_ms:>13.2f}  {speedup:>9.2f}x")

    print(f"\n  Note: F.sdpa uses PyTorch's built-in FlashAttention/efficient backend.")
    print(f"  Our Triton kernel demonstrates the algorithm; F.sdpa is more optimized.\n")


# ===========================================================================
# Memory comparison
# ===========================================================================

def memory_comparison(device: str = None):
    """Show memory savings of flash attention vs naive for long sequences."""
    from naive_attention import naive_attention

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    has_cuda = "cuda" in device

    if not has_cuda:
        print("\n  Memory comparison requires CUDA. Showing theoretical analysis instead.")
        print(f"\n  {'SeqLen':>8}  {'Naive O(N²) MB':>16}  {'Flash O(N) MB':>16}  {'Savings':>10}")
        print(f"  {'-' * 8}  {'-' * 16}  {'-' * 16}  {'-' * 10}")
        B, nH, hD = 2, 8, 64
        for T in [512, 1024, 2048, 4096, 8192]:
            bpe = 4  # float32
            naive_mb = B * nH * T * T * bpe / (1024**2)
            flash_mb = B * nH * T * hD * 4 * bpe / (1024**2)  # Q+K+V+O
            print(f"  {T:>8,}  {naive_mb:>16.1f}  {flash_mb:>16.1f}  {naive_mb/flash_mb:>9.1f}x")
        return

    print(f"\n  Memory comparison (measured on GPU):")
    print(f"  {'SeqLen':>8}  {'Naive peak MB':>16}  {'Flash peak MB':>16}  {'Savings':>10}")
    print(f"  {'-' * 8}  {'-' * 16}  {'-' * 16}  {'-' * 10}")

    B, nH, hD = 2, 8, 64
    for T in [256, 512, 1024, 2048]:
        q = torch.randn(B, nH, T, hD, device=device)
        k = torch.randn(B, nH, T, hD, device=device)
        v = torch.randn(B, nH, T, hD, device=device)

        # Naive
        torch.cuda.reset_peak_memory_stats()
        _ = naive_attention(q, k, v, causal=True)
        torch.cuda.synchronize()
        naive_mb = torch.cuda.max_memory_allocated() / (1024**2)

        # Flash
        torch.cuda.reset_peak_memory_stats()
        _ = flash_attention(q, k, v, causal=True)
        torch.cuda.synchronize()
        flash_mb = torch.cuda.max_memory_allocated() / (1024**2)

        ratio = naive_mb / flash_mb if flash_mb > 0 else 0
        print(f"  {T:>8,}  {naive_mb:>16.1f}  {flash_mb:>16.1f}  {ratio:>9.1f}x")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("  Chapter 2: FlashAttention-2 (Triton / Tiled PyTorch)")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Running on: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  Triton available: {HAS_TRITON}")

    print("\n  [1] Correctness check")
    _verify_flash_attention()

    print("\n  [2] Algorithm walkthrough (tiled online softmax)")
    print("  The key insight: instead of materializing the full N×N attention matrix,")
    print("  we process Q/K/V in tiles and maintain a running softmax via:")
    print("    m_new = max(m_old, max(current_tile_scores))")
    print("    alpha = exp(m_old - m_new)  # correction factor")
    print("    l = l * alpha + sum(exp(scores - m_new))")
    print("    O = O * alpha + softmax_tile @ V_tile")
    print("  This gives exact results with O(N) memory instead of O(N²).")

    print("\n  [3] Memory comparison")
    memory_comparison(device)

    print("\n  [4] Speed benchmark")
    seq_lens = [128, 256, 512, 1024] if device == "cuda" else [64, 128, 256]
    benchmark_all(seq_lens=seq_lens, device=device)
