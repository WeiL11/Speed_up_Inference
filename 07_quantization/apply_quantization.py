"""
Chapter 7 — Quantization
File: apply_quantization.py

Apply quantization to model layers and measure impact on:
  - Latency
  - Memory usage
  - Output quality (vs FP32 reference)

Replaces nn.Linear layers with quantized versions using the naive
quantization methods from naive_quantization.py.

Contents:
  - QuantizedLinear    : INT8 linear layer (dequantize-on-the-fly)
  - INT4Linear         : INT4 packed linear layer
  - quantize_model()   : replace all Linear layers in a model
  - measure_quality()  : compare outputs vs FP32 reference
"""

import sys
import os
import time
import math
import copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, str(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.model_loader import create_model, generate_random_batch, TransformerConfig
from utils.benchmarking import benchmark_fn
from naive_quantization import (
    absmax_quantize, absmax_dequantize,
    group_quantize_int4, group_dequantize_int4,
)


# ---------------------------------------------------------------------------
# INT8 Quantized Linear
# ---------------------------------------------------------------------------

class QuantizedLinear(nn.Module):
    """
    INT8 quantized linear layer.

    Stores weights as INT8 + per-channel scale. During forward pass,
    dequantizes to floating-point and performs the matmul.

    This is "weight-only quantization" — activations stay in FP16/FP32.
    For full INT8 inference (weights + activations), you'd also quantize
    the input tensor, but that requires calibration data.
    """

    def __init__(self, in_features: int, out_features: int, weight_fp: torch.Tensor):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Quantize weights to INT8
        q_weight, scale = absmax_quantize(weight_fp, per_channel=True)
        self.register_buffer("weight_int8", q_weight)
        self.register_buffer("scale", scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize weight on-the-fly
        weight_fp = absmax_dequantize(self.weight_int8, self.scale).to(x.dtype)
        return F.linear(x, weight_fp)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, dtype=INT8"


# ---------------------------------------------------------------------------
# INT4 Quantized Linear
# ---------------------------------------------------------------------------

class INT4Linear(nn.Module):
    """
    INT4 quantized linear layer with per-group scaling.

    Stores weights as packed INT4 (2 values per byte) + per-group scales.
    Dequantizes during forward pass.
    """

    def __init__(
        self, in_features: int, out_features: int,
        weight_fp: torch.Tensor, group_size: int = 128,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        # Pad in_features to be divisible by group_size
        padded_in = ((in_features + group_size - 1) // group_size) * group_size
        if padded_in != in_features:
            weight_fp = F.pad(weight_fp, (0, padded_in - in_features))
        self._padded_in = padded_in

        packed, scales = group_quantize_int4(weight_fp, group_size)
        self.register_buffer("packed", packed)
        self.register_buffer("scales", scales)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_fp = group_dequantize_int4(
            self.packed, self.scales, self.group_size
        ).to(x.dtype)

        # Trim padding if needed
        if self._padded_in != self.in_features:
            weight_fp = weight_fp[:, :self.in_features]

        return F.linear(x, weight_fp)

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"dtype=INT4, group_size={self.group_size}")


# ---------------------------------------------------------------------------
# Model quantization
# ---------------------------------------------------------------------------

def quantize_model(
    model: nn.Module,
    bits: int = 8,
    group_size: int = 128,
    skip_patterns: list = None,
) -> nn.Module:
    """
    Replace all nn.Linear layers with quantized versions.

    Args:
        model: model to quantize (modified in-place)
        bits: 8 for INT8, 4 for INT4
        group_size: group size for INT4 quantization
        skip_patterns: list of substrings — skip layers whose name contains these

    Returns:
        The model with quantized layers
    """
    if skip_patterns is None:
        skip_patterns = []

    replaced = 0
    skipped = 0

    for name, module in model.named_modules():
        for attr_name, child in list(module.named_children()):
            if not isinstance(child, nn.Linear):
                continue

            full_name = f"{name}.{attr_name}" if name else attr_name

            # Check skip patterns
            if any(pattern in full_name for pattern in skip_patterns):
                skipped += 1
                continue

            weight = child.weight.data.clone()
            in_f, out_f = child.in_features, child.out_features

            if bits == 8:
                q_layer = QuantizedLinear(in_f, out_f, weight)
            elif bits == 4:
                q_layer = INT4Linear(in_f, out_f, weight, group_size)
            else:
                raise ValueError(f"Unsupported bits: {bits}")

            q_layer = q_layer.to(weight.device)
            setattr(module, attr_name, q_layer)
            replaced += 1

    print(f"  Quantized {replaced} Linear layers to INT{bits} "
          f"(skipped {skipped})")
    return model


# ---------------------------------------------------------------------------
# Memory comparison
# ---------------------------------------------------------------------------

def memory_comparison(
    num_layers: int = 6,
    hidden_dim: int = 1024,
    num_heads: int = 8,
    device: str = None,
):
    """Compare model size across quantization levels."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n  Memory Comparison ({num_layers} layers, hidden={hidden_dim})")

    configs = [
        ("FP32", torch.float32, None),
        ("FP16", torch.float16, None),
        ("INT8", torch.float32, 8),
        ("INT4", torch.float32, 4),
    ]

    print(f"\n  {'Config':<10}  {'Param MB':>10}  {'Ratio':>8}")
    print(f"  {'-' * 10}  {'-' * 10}  {'-' * 8}")

    fp32_mb = None
    for name, dtype, quant_bits in configs:
        model = create_model(
            num_layers=num_layers, hidden_dim=hidden_dim,
            num_heads=num_heads, dtype=dtype, device="cpu",
        )

        if quant_bits:
            model = quantize_model(model, bits=quant_bits, skip_patterns=["tok_emb", "lm_head"])

        # Calculate actual memory
        total_bytes = 0
        for param in model.parameters():
            total_bytes += param.numel() * param.element_size()
        for buf in model.buffers():
            total_bytes += buf.numel() * buf.element_size()

        mb = total_bytes / (1024 ** 2)
        if fp32_mb is None:
            fp32_mb = mb
        ratio = fp32_mb / mb if mb > 0 else 0

        print(f"  {name:<10}  {mb:>10.1f}  {ratio:>7.1f}x")
        del model


# ---------------------------------------------------------------------------
# Latency comparison
# ---------------------------------------------------------------------------

def latency_comparison(
    num_layers: int = 4,
    hidden_dim: int = 1024,
    num_heads: int = 8,
    batch_size: int = 8,
    seq_len: int = 256,
    device: str = None,
    warmup: int = 5,
    steps: int = 20,
):
    """Benchmark latency for FP32, FP16, INT8, INT4 models."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    has_cuda = "cuda" in device

    input_ids = generate_random_batch(batch_size, seq_len, device=device)

    print(f"\n  Latency Comparison (B={batch_size}, T={seq_len})")
    print(f"\n  {'Config':<10}  {'Latency (ms)':>14}  {'Speedup':>10}")
    print(f"  {'-' * 10}  {'-' * 14}  {'-' * 10}")

    fp32_ms = None

    for name, dtype, quant_bits in [
        ("FP32", torch.float32, None),
        ("FP16", torch.float16, None),
        ("INT8", torch.float32, 8),
        ("INT4", torch.float32, 4),
    ]:
        model = create_model(
            num_layers=num_layers, hidden_dim=hidden_dim,
            num_heads=num_heads, dtype=dtype, device=device,
        )
        model.eval()

        if quant_bits:
            model = quantize_model(model, bits=quant_bits, skip_patterns=["tok_emb", "lm_head"])

        with torch.no_grad():
            r = benchmark_fn(
                lambda: model(input_ids),
                warmup_steps=warmup, measure_steps=steps,
                name=name, sync_cuda=has_cuda,
            )

        if fp32_ms is None:
            fp32_ms = r.mean_ms
        speedup = fp32_ms / r.mean_ms if r.mean_ms > 0 else 0

        print(f"  {name:<10}  {r.mean_ms:>14.2f}  {speedup:>9.2f}x")
        del model
        if has_cuda:
            torch.cuda.empty_cache()

    print(f"\n  Note: naive dequantize-on-the-fly adds overhead vs native INT8 kernels.")
    print(f"  Production systems (bitsandbytes, GPTQ) use optimized CUDA kernels.")


# ---------------------------------------------------------------------------
# Quality comparison
# ---------------------------------------------------------------------------

def quality_comparison(
    num_layers: int = 4,
    hidden_dim: int = 1024,
    num_heads: int = 8,
    batch_size: int = 2,
    seq_len: int = 64,
    device: str = None,
):
    """Compare output quality of quantized models vs FP32 reference."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create reference model
    model_ref = create_model(
        num_layers=num_layers, hidden_dim=hidden_dim,
        num_heads=num_heads, dtype=torch.float32, device=device,
    )
    model_ref.eval()
    input_ids = generate_random_batch(batch_size, seq_len, device=device)

    with torch.no_grad():
        out_ref = model_ref(input_ids).float()

    print(f"\n  Output Quality vs FP32 Reference")
    print(f"\n  {'Config':<15}  {'MSE':>12}  {'Max Err':>12}  {'Cosine Sim':>12}")
    print(f"  {'-' * 15}  {'-' * 12}  {'-' * 12}  {'-' * 12}")

    for name, quant_bits in [("FP16", None), ("INT8", 8), ("INT4 (g=128)", 4), ("INT4 (g=32)", 4)]:
        model = copy.deepcopy(model_ref)

        if name == "FP16":
            model = model.half()
        elif quant_bits:
            gs = 32 if "g=32" in name else 128
            model = quantize_model(model, bits=quant_bits, group_size=gs,
                                   skip_patterns=["tok_emb", "lm_head"])

        model.eval()
        with torch.no_grad():
            out = model(input_ids).float()

        error = out - out_ref
        mse = error.pow(2).mean().item()
        max_err = error.abs().max().item()
        cos_sim = F.cosine_similarity(
            out.reshape(-1).unsqueeze(0),
            out_ref.reshape(-1).unsqueeze(0),
        ).item()

        print(f"  {name:<15}  {mse:>12.6f}  {max_err:>12.4f}  {cos_sim:>12.6f}")
        del model

    del model_ref
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("  Chapter 7: Apply Quantization to Model")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Running on: {device}")

    print("\n  [1] Memory comparison")
    memory_comparison(device=device)

    print("\n  [2] Latency comparison")
    latency_comparison(device=device)

    print("\n  [3] Output quality comparison")
    quality_comparison(device=device)

    print("\n  Key takeaways:")
    print("  - INT8 weight-only: ~4x memory saving, minimal quality loss")
    print("  - INT4: ~8x memory saving, noticeable but manageable degradation")
    print("  - Naive dequant-on-fly is slow; production uses fused INT8/INT4 kernels")
    print("  - Skip embedding/LM head layers for best quality preservation")
