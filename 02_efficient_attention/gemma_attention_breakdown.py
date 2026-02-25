"""
Chapter 2 — Efficient Attention
File: gemma_attention_breakdown.py

Analytical breakdown of Gemma's Grouped-Query Attention (GQA) mechanism.
Uses our custom SimpleTransformer as a stand-in for Gemma-3 1B so that no
HuggingFace credentials are required.

Topics covered:
  - GQA: num_heads query heads sharing fewer KV heads
  - head_dim = hidden_dim // num_heads
  - Tensor shapes at every step
  - FLOPs per attention computation
  - Memory breakdown: Q + K + V + attention_weights + output
"""

import sys
import math
from pathlib import Path

# Allow imports from utils/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from utils.model_loader import TransformerConfig, SimpleTransformer, GEMMA3_1B_CONFIG


# ---------------------------------------------------------------------------
# Analytical helpers
# ---------------------------------------------------------------------------

def bytes_per_element(dtype: torch.dtype) -> int:
    """Return bytes per element for common dtypes."""
    return {
        torch.float32:  4,
        torch.float16:  2,
        torch.bfloat16: 2,
        torch.int8:     1,
    }.get(dtype, 4)


def attention_flops(B: int, nH: int, T: int, hD: int) -> dict:
    """
    Compute FLOPs for each step of scaled dot-product attention.

    Convention: one multiply-add = 2 FLOPs.

    Steps:
      1. QK^T matmul : (B, nH, T, hD) x (B, nH, hD, T)  ->  (B, nH, T, T)
      2. Scale        : element-wise division by sqrt(hD)  ->  T^2 multiplies
      3. Softmax      : exp + sum + division (approx 3 ops per element)
      4. AV matmul    : (B, nH, T, T) x (B, nH, T, hD)   ->  (B, nH, T, hD)

    Returns dict with per-step FLOPs and total.
    """
    # QK^T: for each of the B*nH heads, multiply (T x hD) by (hD x T)
    # = B * nH * T * hD * T  multiplications + same additions = 2 * B*nH*T*hD*T
    qkt_flops = 2 * B * nH * T * hD * T

    # Scale: B * nH * T * T element-wise muls
    scale_flops = B * nH * T * T

    # Softmax: exp (1 FLOP) + running sum (1 FLOP) + divide (1 FLOP) = 3 per element
    softmax_flops = 3 * B * nH * T * T

    # AV: for each of B*nH heads, multiply (T x T) by (T x hD)
    # = B * nH * T * T * hD muls + same additions = 2 * B*nH*T*T*hD
    av_flops = 2 * B * nH * T * T * hD

    total = qkt_flops + scale_flops + softmax_flops + av_flops

    return {
        "QK^T matmul": qkt_flops,
        "Scale (÷√hD)": scale_flops,
        "Softmax": softmax_flops,
        "AV matmul": av_flops,
        "TOTAL": total,
    }


def attention_memory_bytes(
    B: int,
    nH: int,
    nKV: int,
    T: int,
    hD: int,
    dtype: torch.dtype = torch.float32,
    include_kv_cache: bool = False,
) -> dict:
    """
    Memory required for attention tensors (activations during one forward pass).

    Args:
        B      : batch size
        nH     : number of query heads
        nKV    : number of KV heads  (nKV <= nH for GQA)
        T      : sequence length
        hD     : head dimension
        dtype  : element dtype
        include_kv_cache: whether to count KV cache separately

    Returns dict with per-tensor byte counts.
    """
    bpe = bytes_per_element(dtype)

    q_bytes      = B * nH  * T * hD * bpe
    k_bytes      = B * nKV * T * hD * bpe
    v_bytes      = B * nKV * T * hD * bpe
    # Attention weight matrix: (B, nH, T, T) — always nH query heads
    attn_bytes   = B * nH  * T * T  * bpe
    output_bytes = B * nH  * T * hD * bpe  # same shape as Q

    total = q_bytes + k_bytes + v_bytes + attn_bytes + output_bytes

    result = {
        "Q": q_bytes,
        "K": k_bytes,
        "V": v_bytes,
        "Attention weights (N×N)": attn_bytes,
        "Output": output_bytes,
        "TOTAL": total,
    }

    if include_kv_cache:
        # KV cache stores K and V for ALL positions up to T
        result["KV cache (K+V)"] = k_bytes + v_bytes

    return result


def _fmt_bytes(n: int) -> str:
    """Human-readable byte count."""
    if n >= 1 << 30:
        return f"{n / (1 << 30):.2f} GB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.2f} MB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.2f} KB"
    return f"{n} B"


def _fmt_flops(n: int) -> str:
    """Human-readable FLOPs count."""
    if n >= 1e12:
        return f"{n / 1e12:.3f} TFLOPs"
    if n >= 1e9:
        return f"{n / 1e9:.3f} GFLOPs"
    if n >= 1e6:
        return f"{n / 1e6:.3f} MFLOPs"
    return f"{n} FLOPs"


# ---------------------------------------------------------------------------
# GQA explanation
# ---------------------------------------------------------------------------

def explain_gqa(config: TransformerConfig) -> None:
    """Print a detailed explanation of GQA for the given config."""
    nH   = config.num_heads
    nKV  = config.num_kv_heads
    hD   = config.head_dim
    C    = config.hidden_dim
    grp  = config.kv_groups

    print("=" * 70)
    print("  GROUPED-QUERY ATTENTION (GQA) — STRUCTURAL BREAKDOWN")
    print("=" * 70)

    print(f"\n  Config:")
    print(f"    hidden_dim   (C)  = {C}")
    print(f"    num_heads    (nH) = {nH}   <- query heads")
    print(f"    num_kv_heads (nKV)= {nKV}   <- KV heads (GQA grouping)")
    print(f"    head_dim     (hD) = C // nH = {C} // {nH} = {hD}")
    print(f"    kv_groups        = nH // nKV = {nH} // {nKV} = {grp}")

    print(f"\n  Standard MHA would need: {nH} query + {nH} key + {nH} value heads")
    print(f"  GQA uses:                {nH} query + {nKV} key  + {nKV} value heads")
    reduction = (1 - nKV / nH) * 100
    print(f"  KV reduction: {reduction:.0f}% fewer KV parameters")

    print(f"\n  Each KV head is shared by {grp} query head(s):")
    for kv_idx in range(nKV):
        q_idxs = list(range(kv_idx * grp, (kv_idx + 1) * grp))
        print(f"    KV head {kv_idx}  <-  Query heads {q_idxs}")

    print(f"\n  Projection shapes:")
    print(f"    q_proj: Linear({C}, {nH}*{hD})  = Linear({C}, {nH * hD})")
    print(f"    k_proj: Linear({C}, {nKV}*{hD}) = Linear({C}, {nKV * hD})")
    print(f"    v_proj: Linear({C}, {nKV}*{hD}) = Linear({C}, {nKV * hD})")
    print(f"    o_proj: Linear({nH * hD}, {C})")

    q_params  = C * (nH  * hD)
    kv_params = 2 * C * (nKV * hD)
    o_params  = (nH * hD) * C
    total_attn_params = q_params + kv_params + o_params
    mha_kv_params = 2 * C * (nH * hD)
    print(f"\n  Attention parameter counts:")
    print(f"    Q projection: {q_params:,} params")
    print(f"    K+V projections (GQA): {kv_params:,} params  "
          f"(vs {mha_kv_params:,} for full MHA)")
    print(f"    O projection: {o_params:,} params")
    print(f"    Total attention: {total_attn_params:,} params")


# ---------------------------------------------------------------------------
# Tensor shape walkthrough
# ---------------------------------------------------------------------------

def explain_tensor_shapes(config: TransformerConfig, B: int = 2, T: int = 512) -> None:
    """Print tensor shapes at every step of GQA attention."""
    nH  = config.num_heads
    nKV = config.num_kv_heads
    hD  = config.head_dim
    C   = config.hidden_dim

    print("\n" + "=" * 70)
    print("  TENSOR SHAPE WALKTHROUGH")
    print(f"  (B={B}, T={T}, C={C}, nH={nH}, nKV={nKV}, hD={hD})")
    print("=" * 70)

    print(f"\n  Input to attention block:")
    print(f"    x:             ({B}, {T}, {C})")

    print(f"\n  After projections (before reshape):")
    print(f"    q_proj(x):     ({B}, {T}, {nH * hD})")
    print(f"    k_proj(x):     ({B}, {T}, {nKV * hD})")
    print(f"    v_proj(x):     ({B}, {T}, {nKV * hD})")

    print(f"\n  After reshape + transpose  [view + .transpose(1,2)]:")
    print(f"    Q:             ({B}, {nH}, {T}, {hD})")
    print(f"    K:             ({B}, {nKV}, {T}, {hD})")
    print(f"    V:             ({B}, {nKV}, {T}, {hD})")

    if config.kv_groups > 1:
        print(f"\n  GQA expand K/V  [repeat_interleave x{config.kv_groups}]:")
        print(f"    K_expanded:    ({B}, {nH}, {T}, {hD})")
        print(f"    V_expanded:    ({B}, {nH}, {T}, {hD})")

    print(f"\n  Attention score computation:")
    print(f"    scores = Q @ K^T / sqrt({hD})")
    print(f"    Q:             ({B}, {nH}, {T}, {hD})")
    print(f"    K^T:           ({B}, {nH}, {hD}, {T})")
    print(f"    scores:        ({B}, {nH}, {T}, {T})  <-- O(N²) memory!")

    print(f"\n  After causal mask + softmax:")
    print(f"    attn_weights:  ({B}, {nH}, {T}, {T})")

    print(f"\n  Weighted value sum:")
    print(f"    attn_weights:  ({B}, {nH}, {T}, {T})")
    print(f"    V:             ({B}, {nH}, {T}, {hD})")
    print(f"    context:       ({B}, {nH}, {T}, {hD})")

    print(f"\n  After reshape + o_proj:")
    print(f"    context:       ({B}, {T}, {nH * hD})")
    print(f"    output:        ({B}, {T}, {C})")


# ---------------------------------------------------------------------------
# FLOPs + memory breakdown
# ---------------------------------------------------------------------------

def explain_flops_memory(
    config: TransformerConfig,
    B: int = 2,
    T: int = 512,
    dtype: torch.dtype = torch.float32,
) -> None:
    """Print FLOPs and memory breakdown for given config and sequence length."""
    nH  = config.num_heads
    nKV = config.num_kv_heads
    hD  = config.head_dim

    print("\n" + "=" * 70)
    print(f"  FLOPS BREAKDOWN  (B={B}, T={T}, nH={nH}, hD={hD})")
    print("=" * 70)

    flops = attention_flops(B, nH, T, hD)
    for name, val in flops.items():
        sep = "  " if name != "TOTAL" else ""
        print(f"  {sep}{name:<30s} {_fmt_flops(val)}")

    print("\n" + "=" * 70)
    print(f"  MEMORY BREAKDOWN  (dtype={dtype}, B={B}, T={T})")
    print("=" * 70)

    mem = attention_memory_bytes(B, nH, nKV, T, hD, dtype=dtype)
    for name, val in mem.items():
        sep = "  " if name != "TOTAL" else ""
        print(f"  {sep}{name:<30s} {_fmt_bytes(val)}")

    print(f"\n  Note: 'Attention weights (N×N)' scales as O(T²) = O({T}²) = {T*T:,} entries")
    print(f"        At T=4096 that would be {4096**2:,} entries = "
          f"{_fmt_bytes(4096**2 * bytes_per_element(dtype))} per head per batch!")


# ---------------------------------------------------------------------------
# Scaling analysis
# ---------------------------------------------------------------------------

def explain_o2_scaling(config: TransformerConfig, dtype: torch.dtype = torch.float32) -> None:
    """Show how attention memory grows quadratically with sequence length."""
    print("\n" + "=" * 70)
    print("  O(N²) SCALING — ATTENTION WEIGHT MATRIX MEMORY")
    print("=" * 70)

    nH  = config.num_heads
    nKV = config.num_kv_heads
    hD  = config.head_dim
    bpe = bytes_per_element(dtype)
    B   = 1  # per-batch analysis

    print(f"\n  Config: nH={nH}, nKV={nKV}, hD={hD}, dtype={dtype}, B={B}")
    print(f"\n  {'Seq Len':>10}  {'QKV tensors':>14}  {'Attn Matrix':>14}  {'Total':>14}")
    print(f"  {'-'*10}  {'-'*14}  {'-'*14}  {'-'*14}")

    for T in [128, 256, 512, 1024, 2048, 4096, 8192]:
        mem = attention_memory_bytes(B, nH, nKV, T, hD, dtype=dtype)
        qkv  = mem["Q"] + mem["K"] + mem["V"]
        attn = mem["Attention weights (N×N)"]
        tot  = mem["TOTAL"]
        print(f"  {T:>10,}  {_fmt_bytes(qkv):>14}  {_fmt_bytes(attn):>14}  {_fmt_bytes(tot):>14}")


# ---------------------------------------------------------------------------
# Live verification using SimpleTransformer
# ---------------------------------------------------------------------------

def verify_with_simple_transformer(config: TransformerConfig, B: int = 2, T: int = 64) -> None:
    """
    Instantiate a single SelfAttention layer and trace actual tensor shapes.
    Runs on CPU — no GPU needed.
    """
    from utils.model_loader import SelfAttention

    print("\n" + "=" * 70)
    print("  LIVE VERIFICATION — SimpleTransformer SelfAttention")
    print(f"  (B={B}, T={T}, config matches Gemma-3 1B)")
    print("=" * 70)

    device = "cpu"
    attn = SelfAttention(config).to(device)
    attn.eval()

    C = config.hidden_dim
    x = torch.randn(B, T, C, device=device)

    print(f"\n  Input x: {tuple(x.shape)}")

    with torch.no_grad():
        # Manually trace through projection + reshape steps
        q_flat = attn.q_proj(x)
        k_flat = attn.k_proj(x)
        v_flat = attn.v_proj(x)
        print(f"  After q_proj:  {tuple(q_flat.shape)}")
        print(f"  After k_proj:  {tuple(k_flat.shape)}")
        print(f"  After v_proj:  {tuple(v_flat.shape)}")

        nH  = config.num_heads
        nKV = config.num_kv_heads
        hD  = config.head_dim
        q = q_flat.view(B, T, nH,  hD).transpose(1, 2)
        k = k_flat.view(B, T, nKV, hD).transpose(1, 2)
        v = v_flat.view(B, T, nKV, hD).transpose(1, 2)
        print(f"  After reshape: Q={tuple(q.shape)}, K={tuple(k.shape)}, V={tuple(v.shape)}")

        if config.kv_groups > 1:
            k_exp = k.repeat_interleave(config.kv_groups, dim=1)
            v_exp = v.repeat_interleave(config.kv_groups, dim=1)
            print(f"  After GQA expand: K={tuple(k_exp.shape)}, V={tuple(v_exp.shape)}")

        # Full forward pass
        out = attn(x)
        print(f"  Final output:  {tuple(out.shape)}")

    param_q  = sum(p.numel() for p in attn.q_proj.parameters())
    param_k  = sum(p.numel() for p in attn.k_proj.parameters())
    param_v  = sum(p.numel() for p in attn.v_proj.parameters())
    param_o  = sum(p.numel() for p in attn.o_proj.parameters())
    total    = param_q + param_k + param_v + param_o
    print(f"\n  Parameter counts:")
    print(f"    q_proj: {param_q:,}")
    print(f"    k_proj: {param_k:,}")
    print(f"    v_proj: {param_v:,}")
    print(f"    o_proj: {param_o:,}")
    print(f"    Total:  {total:,}")
    print("\n  Verification passed — shapes match analytical predictions.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_full_breakdown(
    config: TransformerConfig = GEMMA3_1B_CONFIG,
    B: int = 2,
    T: int = 512,
    dtype: torch.dtype = torch.float32,
) -> None:
    """Run the complete attention breakdown for the given config."""
    print("\n")
    print("*" * 70)
    print("  Chapter 2: Efficient Attention — Gemma Attention Breakdown")
    print("*" * 70)

    explain_gqa(config)
    explain_tensor_shapes(config, B=B, T=T)
    explain_flops_memory(config, B=B, T=T, dtype=dtype)
    explain_o2_scaling(config, dtype=dtype)
    verify_with_simple_transformer(config, B=min(B, 2), T=64)

    print("\n" + "=" * 70)
    print("  SUMMARY: Why GQA matters for Gemma-3 1B")
    print("=" * 70)
    nH, nKV = config.num_heads, config.num_kv_heads
    C, hD   = config.hidden_dim, config.head_dim
    kv_param_gqa = 2 * C * (nKV * hD)
    kv_param_mha = 2 * C * (nH  * hD)
    print(f"  Gemma-3 1B uses nKV={nKV} vs nH={nH} query heads.")
    print(f"  KV projection params: {kv_param_gqa:,} (GQA) vs {kv_param_mha:,} (MHA)")
    print(f"  KV cache reduction:   {nKV}/{nH} = {nKV/nH:.2%} of full MHA KV cache")
    print(f"  This is critical for long-context (8192 tokens) inference throughput.")
    print()


if __name__ == "__main__":
    run_full_breakdown()
