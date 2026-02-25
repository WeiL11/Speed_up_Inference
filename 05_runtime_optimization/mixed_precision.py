"""
Chapter 5 — Runtime Optimization
File: mixed_precision.py

Mixed precision inference: FP16, BF16, TF32, and AMP autocast comparison.

Key concepts:
  - FP32 (float32): full precision, largest memory footprint, slowest
  - FP16 (float16): half precision, 2x memory savings, uses tensor cores
  - BF16 (bfloat16): same range as FP32 but less mantissa precision,
    preferred on Ampere+ GPUs for training/inference stability
  - TF32: NVIDIA tensor core mode that uses FP32 inputs but rounds
    mantissa to 10 bits internally, giving near-FP32 accuracy with
    tensor core speed (enabled by default on Ampere+)
  - AMP (Automatic Mixed Precision): PyTorch autocast that selects
    optimal precision per operation

Contents:
  - bench_dtype()       : benchmark model at different dtypes
  - bench_tf32()        : show TF32 tensor core mode impact
  - bench_amp()         : PyTorch AMP autocast comparison
  - accuracy_analysis() : measure numerical difference between precisions
"""

import sys
import os
import time
import math
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

from utils.model_loader import TransformerConfig, create_model, generate_random_batch
from utils.benchmarking import benchmark_fn


# ---------------------------------------------------------------------------
# Dtype benchmark
# ---------------------------------------------------------------------------

def bench_dtype(
    num_layers: int = 4,
    hidden_dim: int = 1024,
    num_heads: int = 8,
    batch_size: int = 8,
    seq_len: int = 256,
    device: str = None,
    warmup: int = 5,
    steps: int = 20,
):
    """Benchmark model forward pass at different precisions."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    has_cuda = "cuda" in device

    dtypes_to_test = [torch.float32]
    if has_cuda:
        dtypes_to_test.extend([torch.float16, torch.bfloat16])

    print(f"\n{'=' * 70}")
    print(f"  Dtype Comparison: {num_layers}L, hidden={hidden_dim}, nH={num_heads}")
    print(f"  batch={batch_size}, seq_len={seq_len}, device={device}")
    print(f"{'=' * 70}")

    print(f"\n  {'Dtype':<12}  {'Latency (ms)':>14}  {'Memory (MB)':>13}  {'Speedup':>10}  {'Mem Save':>10}")
    print(f"  {'-' * 12}  {'-' * 14}  {'-' * 13}  {'-' * 10}  {'-' * 10}")

    fp32_ms = None
    fp32_mem = None

    for dtype in dtypes_to_test:
        dtype_name = str(dtype).replace("torch.", "")

        model = create_model(
            num_layers=num_layers, hidden_dim=hidden_dim,
            num_heads=num_heads, dtype=dtype, device=device,
        )
        model.eval()

        input_ids = generate_random_batch(batch_size, seq_len, device=device)

        if has_cuda:
            torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            result = benchmark_fn(
                lambda: model(input_ids),
                warmup_steps=warmup, measure_steps=steps,
                name=dtype_name, sync_cuda=has_cuda,
            )

        if has_cuda:
            mem_mb = torch.cuda.max_memory_allocated() / (1024**2)
        else:
            # Estimate from model size
            bpe = {torch.float32: 4, torch.float16: 2, torch.bfloat16: 2}
            mem_mb = sum(p.numel() for p in model.parameters()) * bpe.get(dtype, 4) / (1024**2)

        if fp32_ms is None:
            fp32_ms = result.mean_ms
            fp32_mem = mem_mb

        speedup = fp32_ms / result.mean_ms if result.mean_ms > 0 else 0
        mem_save = (1 - mem_mb / fp32_mem) * 100 if fp32_mem > 0 else 0

        print(f"  {dtype_name:<12}  {result.mean_ms:>14.2f}  {mem_mb:>13.1f}  "
              f"{speedup:>9.2f}x  {mem_save:>9.1f}%")

        del model
        if has_cuda:
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# TF32 benchmark (Ampere+ GPUs)
# ---------------------------------------------------------------------------

def bench_tf32(
    hidden_dim: int = 1024,
    batch_size: int = 8,
    seq_len: int = 512,
    device: str = None,
    warmup: int = 5,
    steps: int = 20,
):
    """Show impact of TF32 tensor core mode."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device != "cuda":
        print("\n  TF32 benchmark requires CUDA (Ampere+ GPU).")
        return

    print(f"\n  TF32 Tensor Core Mode")
    print(f"  (Available on Ampere+ GPUs: A100, A6000, RTX 30xx, RTX 40xx)")

    model = create_model(
        num_layers=4, hidden_dim=hidden_dim, num_heads=8,
        dtype=torch.float32, device=device,
    )
    model.eval()
    input_ids = generate_random_batch(batch_size, seq_len, device=device)

    results = {}

    for tf32_enabled in [False, True]:
        torch.backends.cuda.matmul.allow_tf32 = tf32_enabled
        torch.backends.cudnn.allow_tf32 = tf32_enabled
        label = "TF32 ON" if tf32_enabled else "TF32 OFF"

        with torch.no_grad():
            result = benchmark_fn(
                lambda: model(input_ids),
                warmup_steps=warmup, measure_steps=steps,
                name=label, sync_cuda=True,
            )
        results[label] = result.mean_ms

    # Restore default
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    off_ms = results.get("TF32 OFF", 1)
    on_ms = results.get("TF32 ON", 1)
    speedup = off_ms / on_ms if on_ms > 0 else 0

    print(f"  TF32 OFF: {off_ms:.2f} ms")
    print(f"  TF32 ON:  {on_ms:.2f} ms")
    print(f"  Speedup:  {speedup:.2f}x")
    print(f"\n  TF32 rounds FP32 mantissa to 10 bits internally,")
    print(f"  enabling tensor core acceleration with negligible accuracy loss.")

    del model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# AMP (Automatic Mixed Precision) benchmark
# ---------------------------------------------------------------------------

def bench_amp(
    hidden_dim: int = 1024,
    batch_size: int = 8,
    seq_len: int = 256,
    device: str = None,
    warmup: int = 5,
    steps: int = 20,
):
    """Compare eager FP32, manual FP16, and AMP autocast."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    has_cuda = "cuda" in device

    if not has_cuda:
        print("\n  AMP benchmark requires CUDA.")
        return

    model = create_model(
        num_layers=4, hidden_dim=hidden_dim, num_heads=8,
        dtype=torch.float32, device=device,
    )
    model.eval()
    input_ids = generate_random_batch(batch_size, seq_len, device=device)

    print(f"\n  AMP Autocast Comparison")

    # FP32 baseline
    with torch.no_grad():
        r_fp32 = benchmark_fn(
            lambda: model(input_ids),
            warmup_steps=warmup, measure_steps=steps,
            name="fp32", sync_cuda=True,
        )

    # AMP with float16
    with torch.no_grad():
        def amp_fp16():
            with torch.amp.autocast("cuda", dtype=torch.float16):
                return model(input_ids)
        r_amp16 = benchmark_fn(
            amp_fp16, warmup_steps=warmup, measure_steps=steps,
            name="amp_fp16", sync_cuda=True,
        )

    # AMP with bfloat16
    with torch.no_grad():
        def amp_bf16():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                return model(input_ids)
        r_amp_bf16 = benchmark_fn(
            amp_bf16, warmup_steps=warmup, measure_steps=steps,
            name="amp_bf16", sync_cuda=True,
        )

    print(f"\n  {'Mode':<16}  {'Latency (ms)':>14}  {'Speedup':>10}")
    print(f"  {'-' * 16}  {'-' * 14}  {'-' * 10}")
    print(f"  {'FP32':<16}  {r_fp32.mean_ms:>14.2f}  {'1.00x':>10}")
    print(f"  {'AMP (FP16)':<16}  {r_amp16.mean_ms:>14.2f}  "
          f"{r_fp32.mean_ms/r_amp16.mean_ms:>9.2f}x")
    print(f"  {'AMP (BF16)':<16}  {r_amp_bf16.mean_ms:>14.2f}  "
          f"{r_fp32.mean_ms/r_amp_bf16.mean_ms:>9.2f}x")

    print(f"\n  AMP automatically selects optimal precision per operation:")
    print(f"  - MatMul/Linear: FP16/BF16 (tensor cores)")
    print(f"  - Softmax/LayerNorm: FP32 (numerical stability)")
    print(f"  - Result: near-FP16 speed with FP32-level stability")

    del model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Accuracy analysis
# ---------------------------------------------------------------------------

def accuracy_analysis(
    hidden_dim: int = 1024,
    batch_size: int = 2,
    seq_len: int = 64,
    device: str = None,
):
    """Measure numerical differences between precision modes."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_fp32 = create_model(
        num_layers=2, hidden_dim=hidden_dim, num_heads=8,
        dtype=torch.float32, device=device,
    )
    model_fp32.eval()
    input_ids = generate_random_batch(batch_size, seq_len, device=device)

    with torch.no_grad():
        out_fp32 = model_fp32(input_ids).float()

        # FP16
        model_fp16 = create_model(
            num_layers=2, hidden_dim=hidden_dim, num_heads=8,
            dtype=torch.float16, device=device,
        )
        # Copy weights
        model_fp16.load_state_dict(
            {k: v.half() for k, v in model_fp32.state_dict().items()}
        )
        model_fp16.eval()
        out_fp16 = model_fp16(input_ids).float()

        # BF16
        model_bf16 = create_model(
            num_layers=2, hidden_dim=hidden_dim, num_heads=8,
            dtype=torch.bfloat16, device=device,
        )
        model_bf16.load_state_dict(
            {k: v.bfloat16() for k, v in model_fp32.state_dict().items()}
        )
        model_bf16.eval()
        out_bf16 = model_bf16(input_ids).float()

    print(f"\n  Numerical Accuracy vs FP32 Reference")
    print(f"  (2-layer model, hidden={hidden_dim})")

    for name, out in [("FP16", out_fp16), ("BF16", out_bf16)]:
        diff = (out - out_fp32).abs()
        rel_diff = diff / (out_fp32.abs() + 1e-8)
        print(f"\n  {name}:")
        print(f"    Max abs error:  {diff.max().item():.6f}")
        print(f"    Mean abs error: {diff.mean().item():.6f}")
        print(f"    Max rel error:  {rel_diff.max().item():.6f}")
        print(f"    Mean rel error: {rel_diff.mean().item():.6f}")

    del model_fp32, model_fp16, model_bf16
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("  Chapter 5: Mixed Precision Inference")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Running on: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")

    print("\n  [1] Dtype comparison (FP32 vs FP16 vs BF16)")
    bench_dtype(device=device)

    print("\n  [2] TF32 tensor core mode")
    bench_tf32(device=device)

    print("\n  [3] AMP autocast comparison")
    bench_amp(device=device)

    print("\n  [4] Numerical accuracy analysis")
    accuracy_analysis(device=device)

    print("\n  Summary:")
    print("  - FP16/BF16: ~2x memory savings, tensor core speedup")
    print("  - BF16 preferred: same dynamic range as FP32, more stable")
    print("  - TF32: free speedup on Ampere+, enabled by default")
    print("  - AMP: best of both worlds (fast ops in FP16, sensitive ops in FP32)")
