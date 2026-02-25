"""
Chapter 7 — Quantization
File: naive_quantization.py

Quantization theory and FROM-SCRATCH implementations.

Quantization reduces model weights from floating-point (16/32-bit) to
lower-precision integers (8-bit, 4-bit), dramatically reducing:
  - Memory footprint (2-8x)
  - Memory bandwidth usage (key for memory-bound inference)

This file implements three quantization schemes from scratch:
  1. Absmax INT8    : symmetric, per-tensor or per-channel
  2. Zero-point INT8: asymmetric, handles non-zero-centered distributions
  3. Per-group INT4 : 4-bit quantization with group-wise scaling

Contents:
  - absmax_quantize/dequantize     : symmetric INT8
  - zeropoint_quantize/dequantize  : asymmetric INT8
  - group_quantize_int4/dequantize : per-group INT4 with packing
  - error_analysis                 : MSE, max abs error comparison
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn


# ===========================================================================
# 1. Absmax Quantization (Symmetric INT8)
# ===========================================================================

def absmax_quantize(
    tensor: torch.Tensor,
    per_channel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetric absmax quantization to INT8.

    Maps the range [-absmax, absmax] to [-127, 127].

    Formula:
      scale = absmax / 127
      quantized = round(tensor / scale)
      quantized = clamp(quantized, -128, 127)

    Args:
        tensor: input floating-point tensor
        per_channel: if True, compute scale per output channel (dim=0)

    Returns:
        (quantized_int8, scale) — scale needed for dequantization
    """
    if per_channel and tensor.ndim >= 2:
        # Per-channel: scale per row (output channel dimension)
        absmax = tensor.abs().amax(dim=list(range(1, tensor.ndim)), keepdim=True)
    else:
        absmax = tensor.abs().max()

    # Avoid division by zero
    absmax = absmax.clamp(min=1e-8)
    scale = absmax / 127.0

    quantized = torch.round(tensor / scale).clamp(-128, 127).to(torch.int8)
    return quantized, scale


def absmax_dequantize(
    quantized: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Dequantize INT8 absmax back to floating-point."""
    return quantized.float() * scale


# ===========================================================================
# 2. Zero-Point Quantization (Asymmetric INT8)
# ===========================================================================

def zeropoint_quantize(
    tensor: torch.Tensor,
    per_channel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Asymmetric zero-point quantization to INT8.

    Maps the range [min, max] to [0, 255] (unsigned) by introducing a
    zero-point offset. Better than absmax for tensors with non-zero mean.

    Formula:
      scale = (max - min) / 255
      zero_point = round(-min / scale)
      quantized = round(tensor / scale) + zero_point
      quantized = clamp(quantized, 0, 255)

    Args:
        tensor: input floating-point tensor
        per_channel: per-channel quantization

    Returns:
        (quantized_uint8, scale, zero_point)
    """
    if per_channel and tensor.ndim >= 2:
        reduce_dims = list(range(1, tensor.ndim))
        t_min = tensor.amin(dim=reduce_dims, keepdim=True)
        t_max = tensor.amax(dim=reduce_dims, keepdim=True)
    else:
        t_min = tensor.min()
        t_max = tensor.max()

    scale = (t_max - t_min) / 255.0
    scale = scale.clamp(min=1e-8)
    zero_point = torch.round(-t_min / scale).clamp(0, 255)

    quantized = torch.round(tensor / scale) + zero_point
    quantized = quantized.clamp(0, 255).to(torch.uint8)

    return quantized, scale, zero_point


def zeropoint_dequantize(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
) -> torch.Tensor:
    """Dequantize zero-point INT8 back to floating-point."""
    return (quantized.float() - zero_point.float()) * scale


# ===========================================================================
# 3. Per-Group INT4 Quantization
# ===========================================================================

def group_quantize_int4(
    tensor: torch.Tensor,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-group symmetric INT4 quantization.

    Divides each row into groups of `group_size` elements, computes a
    per-group scale, and quantizes to 4-bit integers [-8, 7].

    Two 4-bit values are packed into one uint8 for memory efficiency.

    Args:
        tensor: 2D weight tensor (out_features, in_features)
        group_size: number of elements per quantization group

    Returns:
        (packed_uint8, scales)
        packed_uint8: shape (out, in_features // group_size, group_size // 2)
        scales: shape (out, in_features // group_size)
    """
    assert tensor.ndim == 2, "INT4 quantization requires 2D tensor"
    out_features, in_features = tensor.shape
    assert in_features % group_size == 0, (
        f"in_features {in_features} must be divisible by group_size {group_size}"
    )

    num_groups = in_features // group_size

    # Reshape to (out, num_groups, group_size)
    tensor_grouped = tensor.reshape(out_features, num_groups, group_size)

    # Per-group scale
    absmax = tensor_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = absmax / 7.0  # INT4 signed: [-8, 7]

    # Quantize
    quantized = torch.round(tensor_grouped / scale).clamp(-8, 7).to(torch.int8)

    # Pack two int4 values into one uint8
    # Low nibble: even indices, high nibble: odd indices
    assert group_size % 2 == 0
    q_even = quantized[:, :, 0::2] & 0x0F  # low nibble
    q_odd = (quantized[:, :, 1::2] & 0x0F) << 4  # high nibble
    packed = (q_even | q_odd).to(torch.uint8)

    return packed, scale.squeeze(-1)


def group_dequantize_int4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """Dequantize packed INT4 back to floating-point."""
    out_features = packed.shape[0]
    num_groups = packed.shape[1]
    half_group = packed.shape[2]

    # Unpack
    low_nibble = (packed & 0x0F).to(torch.int8)
    high_nibble = ((packed >> 4) & 0x0F).to(torch.int8)

    # Sign-extend 4-bit to 8-bit
    low_nibble = torch.where(low_nibble > 7, low_nibble - 16, low_nibble)
    high_nibble = torch.where(high_nibble > 7, high_nibble - 16, high_nibble)

    # Interleave back
    dequantized = torch.zeros(
        out_features, num_groups, group_size,
        device=packed.device, dtype=torch.float32,
    )
    dequantized[:, :, 0::2] = low_nibble.float()
    dequantized[:, :, 1::2] = high_nibble.float()

    # Apply scale
    dequantized = dequantized * scales.unsqueeze(-1)

    return dequantized.reshape(out_features, num_groups * group_size)


# ===========================================================================
# Error analysis
# ===========================================================================

def error_analysis(weight: torch.Tensor, group_size: int = 128):
    """
    Compare quantization error across all three methods.

    Measures:
      - MSE (mean squared error)
      - Max absolute error
      - Signal-to-noise ratio (SNR in dB)
    """
    print(f"\n  Quantization Error Analysis")
    print(f"  Weight shape: {tuple(weight.shape)}")
    print(f"  Weight stats: mean={weight.mean():.4f}, std={weight.std():.4f}, "
          f"min={weight.min():.4f}, max={weight.max():.4f}")

    methods = {}

    # Absmax INT8
    q, s = absmax_quantize(weight, per_channel=True)
    recon = absmax_dequantize(q, s)
    methods["Absmax INT8 (per-ch)"] = (recon, 8)

    q, s = absmax_quantize(weight, per_channel=False)
    recon = absmax_dequantize(q, s)
    methods["Absmax INT8 (per-tensor)"] = (recon, 8)

    # Zero-point INT8
    q, s, zp = zeropoint_quantize(weight, per_channel=True)
    recon = zeropoint_dequantize(q, s, zp)
    methods["Zero-point INT8 (per-ch)"] = (recon, 8)

    # INT4
    if weight.ndim == 2 and weight.shape[1] % group_size == 0:
        packed, scales = group_quantize_int4(weight, group_size)
        recon = group_dequantize_int4(packed, scales, group_size)
        methods[f"Group INT4 (g={group_size})"] = (recon, 4)

        # Smaller group for comparison
        if weight.shape[1] % 32 == 0:
            packed, scales = group_quantize_int4(weight, 32)
            recon = group_dequantize_int4(packed, scales, 32)
            methods["Group INT4 (g=32)"] = (recon, 4)

    # Print results
    print(f"\n  {'Method':<28}  {'Bits':>5}  {'MSE':>12}  {'Max Err':>10}  {'SNR (dB)':>10}")
    print(f"  {'-' * 28}  {'-' * 5}  {'-' * 12}  {'-' * 10}  {'-' * 10}")

    w_float = weight.float()
    signal_power = w_float.pow(2).mean().item()

    for name, (recon, bits) in methods.items():
        error = (w_float - recon.float())
        mse = error.pow(2).mean().item()
        max_err = error.abs().max().item()
        snr = 10 * math.log10(signal_power / mse) if mse > 0 else float("inf")

        print(f"  {name:<28}  {bits:>5}  {mse:>12.6f}  {max_err:>10.6f}  {snr:>10.1f}")

    # Memory savings
    print(f"\n  Memory Savings:")
    base_bytes = weight.numel() * weight.element_size()
    for bits_label, bits in [("FP32", 32), ("FP16", 16), ("INT8", 8), ("INT4", 4)]:
        total_bytes = weight.numel() * bits / 8
        ratio = base_bytes / total_bytes
        print(f"    {bits_label}: {total_bytes / (1024**2):.1f} MB  ({ratio:.1f}x smaller than FP32)")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  Chapter 7: Naive Quantization (From Scratch)")
    print("=" * 70)

    # Create a realistic weight tensor (simulating a linear layer)
    torch.manual_seed(42)
    weight = torch.randn(1024, 1024) * 0.02  # typical init scale

    print("\n  [1] Absmax INT8 demo")
    q, s = absmax_quantize(weight)
    recon = absmax_dequantize(q, s)
    print(f"  Original: dtype={weight.dtype}, shape={tuple(weight.shape)}")
    print(f"  Quantized: dtype={q.dtype}, scale={s.item():.6f}")
    print(f"  Memory: {weight.numel() * 4 / 1024:.0f} KB → {q.numel() / 1024:.0f} KB (4x reduction)")

    print("\n  [2] Zero-point INT8 demo")
    # Add a bias to show zero-point advantage
    biased_weight = weight + 0.01
    q, s, zp = zeropoint_quantize(biased_weight)
    recon = zeropoint_dequantize(q, s, zp)
    print(f"  Zero-point: {zp.item():.0f}")
    print(f"  Scale: {s.item():.6f}")

    print("\n  [3] Per-group INT4 demo")
    packed, scales = group_quantize_int4(weight, group_size=128)
    recon = group_dequantize_int4(packed, scales, group_size=128)
    print(f"  Packed shape: {tuple(packed.shape)} (2 values per byte)")
    print(f"  Scales shape: {tuple(scales.shape)} (one per group)")
    print(f"  Memory: {weight.numel() * 4 / 1024:.0f} KB → "
          f"{(packed.numel() + scales.numel() * 4) / 1024:.0f} KB")

    print("\n  [4] Error analysis (all methods)")
    error_analysis(weight)

    print("\n  Key takeaways:")
    print("  - INT8: 4x memory reduction, minimal quality loss")
    print("  - INT4: 8x memory reduction, more noticeable degradation")
    print("  - Per-channel/per-group scaling significantly improves accuracy")
    print("  - Smaller groups = better accuracy but more scale overhead")
