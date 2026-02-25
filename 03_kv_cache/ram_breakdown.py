"""
ram_breakdown.py  —  Chapter 3: KV Cache

Analytically compute and display where GPU RAM goes during LLM inference.
No GPU required; all calculations are arithmetic.

Usage:
    python ram_breakdown.py
    python ram_breakdown.py --hidden_dim 1152 --num_layers 26 --num_heads 4 \
        --num_kv_heads 1 --seq_len 2048 --batch_size 8 --dtype float16
"""

import argparse
import sys
import os

# Allow running from any working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.model_loader import TransformerConfig, GEMMA3_1B_CONFIG


# ---------------------------------------------------------------------------
# Dtype helpers
# ---------------------------------------------------------------------------

DTYPE_BYTES = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int8": 1,
    "int4": 0.5,
}


def dtype_bytes(dtype_str: str) -> float:
    """Return bytes per element for the given dtype string."""
    if dtype_str not in DTYPE_BYTES:
        raise ValueError(
            f"Unknown dtype '{dtype_str}'. Choose from {list(DTYPE_BYTES.keys())}"
        )
    return DTYPE_BYTES[dtype_str]


# ---------------------------------------------------------------------------
# Component breakdown functions
# ---------------------------------------------------------------------------

def compute_model_weights_mb(
    config: TransformerConfig,
    bytes_per_param: float,
) -> dict:
    """
    Analytically compute model weight memory in MB.

    Returns a dict with per-component breakdown and total.
    """
    hd = config.hidden_dim
    nh = config.num_heads
    nkv = config.num_kv_heads
    hd_per_head = config.head_dim  # hidden_dim // num_heads
    inter = config.intermediate_dim
    vocab = config.vocab_size
    nl = config.num_layers

    # Token embedding: vocab_size × hidden_dim
    embed_params = vocab * hd

    per_layer = {}

    # Attention projections (no bias)
    # Q: hidden_dim × (num_heads * head_dim)  = hidden_dim × hidden_dim
    q_params = hd * (nh * hd_per_head)
    # K: hidden_dim × (num_kv_heads * head_dim)
    k_params = hd * (nkv * hd_per_head)
    # V: hidden_dim × (num_kv_heads * head_dim)
    v_params = hd * (nkv * hd_per_head)
    # O: (num_heads * head_dim) × hidden_dim = hidden_dim × hidden_dim
    o_params = (nh * hd_per_head) * hd

    attn_params = q_params + k_params + v_params + o_params

    # MLP: gate_proj + up_proj + down_proj (no bias)
    gate_params = hd * inter
    up_params   = hd * inter
    down_params = inter * hd
    mlp_params  = gate_params + up_params + down_params

    # 2 RMSNorm layers per block: each has hidden_dim parameters (scale only)
    rms_params = 2 * hd

    layer_params = attn_params + mlp_params + rms_params

    # Final norm
    final_norm_params = hd

    # LM head
    if config.tie_weights:
        lm_head_params = 0  # shared with embedding
    else:
        lm_head_params = vocab * hd

    total_params = (
        embed_params
        + nl * layer_params
        + final_norm_params
        + lm_head_params
    )

    def _mb(p):
        return p * bytes_per_param / (1024 ** 2)

    return {
        "embedding_mb": _mb(embed_params),
        "per_layer_attn_mb": _mb(attn_params),
        "per_layer_mlp_mb": _mb(mlp_params),
        "per_layer_rms_mb": _mb(rms_params),
        "per_layer_total_mb": _mb(layer_params),
        "all_layers_mb": _mb(nl * layer_params),
        "final_norm_mb": _mb(final_norm_params),
        "lm_head_mb": _mb(lm_head_params),
        "total_mb": _mb(total_params),
        "total_params": total_params,
        "tie_weights": config.tie_weights,
    }


def compute_kv_cache_mb(
    config: TransformerConfig,
    seq_len: int,
    batch_size: int,
    bytes_per_param: float,
) -> dict:
    """
    Compute KV cache memory in MB for given seq_len and batch_size.

    KV cache shape per layer: 2 × (B, num_kv_heads, seq_len, head_dim)
    """
    nl = config.num_layers
    nkv = config.num_kv_heads
    hd_per_head = config.head_dim

    # Per layer: K tensor + V tensor
    # Each: batch_size × num_kv_heads × seq_len × head_dim
    per_layer_elements = 2 * batch_size * nkv * seq_len * hd_per_head
    per_layer_mb = per_layer_elements * bytes_per_param / (1024 ** 2)
    total_mb = per_layer_mb * nl

    return {
        "per_layer_mb": per_layer_mb,
        "total_mb": total_mb,
        "num_layers": nl,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "num_kv_heads": nkv,
        "head_dim": hd_per_head,
    }


def compute_activations_mb(
    config: TransformerConfig,
    seq_len: int,
    batch_size: int,
    bytes_per_param: float,
    num_layers_stored: int = None,
) -> dict:
    """
    Estimate peak activation memory during a forward pass.

    During training, activations from each layer are stored for backward.
    During inference, only a rolling window is needed — but peak still occurs
    at the largest layer.

    Components:
      - Residual stream: stored per layer (for backward); shape (B, T, hidden_dim)
      - Attention matrix (naive SDPA): (B, num_heads, T, T)
    """
    hd = config.hidden_dim
    nh = config.num_heads
    nl = config.num_layers

    if num_layers_stored is None:
        num_layers_stored = nl  # worst case: all layers for backward

    # Residual stream per layer
    residual_per_layer = batch_size * seq_len * hd
    residual_total = residual_per_layer * num_layers_stored

    # Attention score matrix per layer (naive O(T²) attention)
    # Shape: (B, num_heads, T, T)
    attn_matrix_per_layer = batch_size * nh * seq_len * seq_len
    attn_matrix_total = attn_matrix_per_layer  # only one layer at a time at peak

    total_elements = residual_total + attn_matrix_total

    def _mb(e):
        return e * bytes_per_param / (1024 ** 2)

    return {
        "residual_per_layer_mb": _mb(residual_per_layer),
        "residual_all_layers_mb": _mb(residual_total),
        "attn_matrix_peak_mb": _mb(attn_matrix_per_layer),
        "total_approx_mb": _mb(total_elements),
        "note": (
            "Residual stored for all layers (training). "
            "Attention matrix peaks at O(B*nh*T^2) per layer."
        ),
    }


def compute_optimizer_state_mb(
    model_weights_info: dict,
    bytes_per_param: float,
) -> dict:
    """
    Compute AdamW optimizer state memory.

    AdamW stores:
      - Gradients:   1× model params (same dtype as model)
      - Momentum m:  1× model params (float32)
      - Variance v:  1× model params (float32)

    Typically optimizer states are kept in float32 regardless of model dtype.
    """
    total_params = model_weights_info["total_params"]

    # Gradients: same dtype as model
    grad_mb = total_params * bytes_per_param / (1024 ** 2)

    # AdamW moments: always float32
    momentum_mb = total_params * 4 / (1024 ** 2)
    variance_mb = total_params * 4 / (1024 ** 2)

    total_mb = grad_mb + momentum_mb + variance_mb

    return {
        "gradients_mb": grad_mb,
        "momentum_mb": momentum_mb,
        "variance_mb": variance_mb,
        "total_mb": total_mb,
        "note": "AdamW momentum & variance kept in float32 regardless of model dtype.",
    }


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def _hline(width=72):
    return "-" * width


def _dline(width=72):
    return "=" * width


def print_model_weights_table(weights_info: dict, config: TransformerConfig):
    print()
    print(_dline())
    print("  MODEL WEIGHTS BREAKDOWN")
    print(_dline())
    print(f"  {'Component':<40} {'MB':>10}")
    print(_hline())
    print(f"  {'Token Embedding':<40} {weights_info['embedding_mb']:>10.1f}")
    print(f"  {'Per-Layer Attention (Q+K+V+O)':<40} {weights_info['per_layer_attn_mb']:>10.1f}")
    print(f"  {'Per-Layer MLP (gate+up+down)':<40} {weights_info['per_layer_mlp_mb']:>10.1f}")
    print(f"  {'Per-Layer RMSNorm (x2)':<40} {weights_info['per_layer_rms_mb']:>10.4f}")
    print(f"  {'Per-Layer Total':<40} {weights_info['per_layer_total_mb']:>10.1f}")
    print(f"  {f'All {config.num_layers} Layers':<40} {weights_info['all_layers_mb']:>10.1f}")
    print(f"  {'Final RMSNorm':<40} {weights_info['final_norm_mb']:>10.4f}")
    lm_head_note = " (tied)" if weights_info["tie_weights"] else ""
    print(f"  {f'LM Head{lm_head_note}':<40} {weights_info['lm_head_mb']:>10.1f}")
    print(_hline())
    print(f"  {'TOTAL':<40} {weights_info['total_mb']:>10.1f}")
    print(f"  {'Total Parameters':<40} {weights_info['total_params']:>10,}")
    print(_dline())


def print_kv_cache_table(
    config: TransformerConfig,
    seq_lens: list,
    batch_sizes: list,
    bytes_per_param: float,
    model_total_mb: float,
):
    print()
    print(_dline())
    print("  KV CACHE SIZE (MB)  —  per (batch_size, seq_len)")
    print(f"  Model weights: {model_total_mb:.1f} MB  |  dtype: {bytes_per_param*8:.0f}-bit")
    print(_dline())

    # Header
    header = f"  {'batch \\ seq':<12}"
    for sl in seq_lens:
        header += f" {sl:>10}"
    print(header)
    print(_hline())

    for bs in batch_sizes:
        row = f"  batch={bs:<6}"
        for sl in seq_lens:
            info = compute_kv_cache_mb(config, sl, bs, bytes_per_param)
            kv_mb = info["total_mb"]
            marker = " (*)" if kv_mb > model_total_mb else "    "
            row += f" {kv_mb:>7.1f}{marker}"
        print(row)

    print()
    print("  (*) KV cache exceeds model weights")
    print(_dline())


def print_activation_table(
    config: TransformerConfig,
    seq_lens: list,
    batch_sizes: list,
    bytes_per_param: float,
):
    print()
    print(_dline())
    print("  PEAK ACTIVATIONS (MB, training)  —  residual + attn matrix")
    print(_dline())

    header = f"  {'batch \\ seq':<12}"
    for sl in seq_lens:
        header += f" {sl:>10}"
    print(header)
    print(_hline())

    for bs in batch_sizes:
        row = f"  batch={bs:<6}"
        for sl in seq_lens:
            info = compute_activations_mb(config, sl, bs, bytes_per_param)
            row += f" {info['total_approx_mb']:>10.1f}"
        print(row)
    print(_dline())


def print_optimizer_table(weights_info: dict, bytes_per_param: float):
    opt = compute_optimizer_state_mb(weights_info, bytes_per_param)
    print()
    print(_dline())
    print("  OPTIMIZER STATE (AdamW, training only)")
    print(_dline())
    print(f"  {'Component':<40} {'MB':>10}")
    print(_hline())
    print(f"  {'Gradients (model dtype)':<40} {opt['gradients_mb']:>10.1f}")
    print(f"  {'Momentum m (float32)':<40} {opt['momentum_mb']:>10.1f}")
    print(f"  {'Variance v (float32)':<40} {opt['variance_mb']:>10.1f}")
    print(_hline())
    print(f"  {'TOTAL Optimizer State':<40} {opt['total_mb']:>10.1f}")
    print(f"  {opt['note']}")
    print(_dline())


def print_doubling_analysis(
    config: TransformerConfig,
    base_seq: int,
    base_batch: int,
    bytes_per_param: float,
):
    """Show how KV cache scales when seq_len or batch doubles."""
    print()
    print(_dline())
    print("  KV CACHE SCALING  —  Doubling Analysis")
    print(_dline())

    base = compute_kv_cache_mb(config, base_seq, base_batch, bytes_per_param)
    print(f"  Base: seq_len={base_seq}, batch={base_batch}  ->  {base['total_mb']:.2f} MB")

    seq2 = compute_kv_cache_mb(config, base_seq * 2, base_batch, bytes_per_param)
    print(
        f"  Seq ×2 (seq_len={base_seq*2}, batch={base_batch}): "
        f"{seq2['total_mb']:.2f} MB  "
        f"(ratio={seq2['total_mb']/base['total_mb']:.2f}x)"
    )

    bat2 = compute_kv_cache_mb(config, base_seq, base_batch * 2, bytes_per_param)
    print(
        f"  Batch×2 (seq_len={base_seq}, batch={base_batch*2}): "
        f"{bat2['total_mb']:.2f} MB  "
        f"(ratio={bat2['total_mb']/base['total_mb']:.2f}x)"
    )
    print()
    print("  Observation: KV cache grows LINEARLY with both seq_len and batch_size.")
    print("  At large scales, KV cache dominates over model weights.")
    print(_dline())


def print_full_summary(
    config: TransformerConfig,
    seq_lens: list,
    batch_sizes: list,
    bytes_per_param: float,
):
    """Print a combined RAM summary for given config and sweep parameters."""
    print()
    print(_dline(72))
    print("  GPU RAM BREAKDOWN  —  LLM Inference")
    print(f"  Model: hidden={config.hidden_dim}, layers={config.num_layers}, "
          f"heads={config.num_heads}, kv_heads={config.num_kv_heads}")
    print(f"  intermediate_dim={config.intermediate_dim}, "
          f"vocab={config.vocab_size}, dtype={bytes_per_param*8:.0f}-bit")
    print(_dline(72))

    weights_info = compute_model_weights_mb(config, bytes_per_param)
    print_model_weights_table(weights_info, config)
    print_kv_cache_table(config, seq_lens, batch_sizes, bytes_per_param, weights_info["total_mb"])
    print_activation_table(config, seq_lens, batch_sizes, bytes_per_param)
    print_optimizer_table(weights_info, bytes_per_param)
    print_doubling_analysis(config, base_seq=1024, base_batch=4, bytes_per_param=bytes_per_param)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analytically compute GPU RAM breakdown for LLM inference."
    )
    parser.add_argument("--hidden_dim",   type=int,   default=1152,
                        help="Model hidden dimension (default: 1152, Gemma-3 1B)")
    parser.add_argument("--num_layers",   type=int,   default=26,
                        help="Number of transformer layers")
    parser.add_argument("--num_heads",    type=int,   default=4,
                        help="Number of attention heads")
    parser.add_argument("--num_kv_heads", type=int,   default=1,
                        help="Number of KV heads (GQA). Set equal to num_heads for MHA.")
    parser.add_argument("--intermediate_dim", type=int, default=6912,
                        help="MLP intermediate dimension (0=auto)")
    parser.add_argument("--vocab_size",   type=int,   default=256_000)
    parser.add_argument("--seq_len",      type=int,   default=None,
                        help="Single seq_len to show (default: sweep 512,1024,2048,4096)")
    parser.add_argument("--batch_size",   type=int,   default=None,
                        help="Single batch_size to show (default: sweep 1,4,8,16)")
    parser.add_argument("--dtype",        type=str,   default="float16",
                        choices=list(DTYPE_BYTES.keys()),
                        help="Data type for parameters and KV cache")
    args = parser.parse_args()

    config = TransformerConfig(
        vocab_size=args.vocab_size,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        intermediate_dim=args.intermediate_dim,
        num_layers=args.num_layers,
        tie_weights=True,
    )

    bpp = dtype_bytes(args.dtype)

    seq_lens   = [args.seq_len]   if args.seq_len   else [512, 1024, 2048, 4096]
    batch_sizes = [args.batch_size] if args.batch_size else [1, 4, 8, 16]

    print_full_summary(config, seq_lens, batch_sizes, bpp)


if __name__ == "__main__":
    main()
