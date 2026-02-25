"""
Chapter 7 — Quantization
File: bitsandbytes_quant.py

Survey and comparison of production quantization libraries:
  1. bitsandbytes (LLM.int8, NF4)
  2. GPTQ (post-training, calibration-based)
  3. AWQ (activation-aware weight quantization)
  4. GGUF (llama.cpp format)

This file provides educational overviews and, where the libraries are
installed, live benchmarks. If a library isn't available, theoretical
comparisons are shown instead.

Contents:
  - survey()           : overview of each method with pros/cons
  - bnb_demo()         : bitsandbytes INT8/NF4 demo (if installed)
  - comparison_table() : theoretical and practical comparison
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Check available libraries
# ---------------------------------------------------------------------------

def _check_library(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False

HAS_BNB = _check_library("bitsandbytes")
HAS_GPTQ = _check_library("auto_gptq")
HAS_AWQ = _check_library("awq")


# ---------------------------------------------------------------------------
# Survey of quantization methods
# ---------------------------------------------------------------------------

def survey():
    """Print detailed survey of production quantization methods."""
    print(f"""
  ================================================================
  PRODUCTION QUANTIZATION METHODS — SURVEY
  ================================================================

  1. BITSANDBYTES (Tim Dettmers)
  ────────────────────────────────
  Methods:
    - LLM.int8(): Mixed INT8/FP16 quantization
      - Decomposes weight matrix into outlier (FP16) and regular (INT8) parts
      - Outlier channels (>6σ) stay in FP16 to preserve quality
      - Regular channels use absmax INT8
      - Result: ~0% quality loss, ~50% memory reduction

    - NF4 (4-bit NormalFloat): Used in QLoRA
      - Uses information-theoretically optimal 4-bit data type
      - Based on the assumption that weights are normally distributed
      - Double quantization: quantize the quantization constants too
      - Result: <1% quality loss, ~75% memory reduction

  Pros: Easy to use (one flag in HuggingFace), no calibration needed
  Cons: Slower than native FP16 (dequantization overhead), no INT4 matmul

  Installation: pip install bitsandbytes

  2. GPTQ (Frantar et al. 2023)
  ────────────────────────────────
  Method: Post-training quantization with calibration data
    - Quantizes weights one column at a time
    - Uses Hessian information to minimize output error
    - Compensates for each column's quantization error in remaining columns
    - Typically INT4 with per-group scaling (group_size=128)

  Pros: Best INT4 quality (uses calibration data), fast inference with
        optimized CUDA kernels (ExLlama, Marlin)
  Cons: Requires calibration dataset and quantization time (~hours)

  Installation: pip install auto-gptq

  3. AWQ (Activation-Aware Weight Quantization, Lin et al. 2023)
  ────────────────────────────────
  Method: Identifies salient weight channels based on activation magnitudes
    - Key insight: 1% of weights are far more important than the rest
    - Protects salient channels (scale them up before quantization)
    - Per-channel scaling to equalize weight ranges
    - Typically INT4 with per-group scaling

  Pros: Better quality than naive INT4, activation-aware protection,
        fast inference with optimized kernels
  Cons: Requires calibration data, newer (less battle-tested than GPTQ)

  Installation: pip install autoawq

  4. GGUF (llama.cpp, Georgi Gerganov)
  ────────────────────────────────
  Method: CPU-optimized quantization format for llama.cpp
    - Multiple quantization types: Q2_K through Q8_0
    - Block quantization with per-block scales and mins
    - K-quants use importance-based mixed precision per layer
    - Optimized for CPU inference (AVX, ARM NEON)

  Formats:
    Q4_0: 4-bit uniform, 32-element blocks
    Q4_K_M: 4-bit k-quant medium (mixed precision per layer)
    Q5_K_M: 5-bit k-quant medium
    Q8_0: 8-bit uniform

  Pros: CPU inference champion, many format options, huge community
  Cons: CPU-only (no GPU kernels), quantization quality varies by format

  Tool: python -m llama_cpp.convert (from llama-cpp-python)
""")


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def comparison_table():
    """Print structured comparison of all methods."""
    print(f"\n  {'Method':<20}  {'Bits':>5}  {'Calibration':>13}  {'Speed':>8}  "
          f"{'Quality':>9}  {'Target':>8}")
    print(f"  {'-' * 20}  {'-' * 5}  {'-' * 13}  {'-' * 8}  {'-' * 9}  {'-' * 8}")

    rows = [
        ("BNB LLM.int8()", "8", "No", "Medium", "Excellent", "GPU"),
        ("BNB NF4", "4", "No", "Medium", "Very Good", "GPU"),
        ("GPTQ", "4", "Yes (128+)", "Fast", "Best INT4", "GPU"),
        ("AWQ", "4", "Yes (128+)", "Fast", "Very Good", "GPU"),
        ("GGUF Q4_K_M", "4-5", "No", "Fast", "Good", "CPU"),
        ("GGUF Q8_0", "8", "No", "Fast", "Excellent", "CPU"),
        ("Naive INT8", "8", "No", "Slow*", "Good", "Any"),
        ("Naive INT4", "4", "No", "Slow*", "Fair", "Any"),
    ]

    for name, bits, calib, speed, quality, target in rows:
        print(f"  {name:<20}  {bits:>5}  {calib:>13}  {speed:>8}  "
              f"{quality:>9}  {target:>8}")

    print(f"\n  * Naive methods use dequantize-on-the-fly (no optimized kernels)")
    print(f"  Production methods use fused INT4/INT8 GEMM kernels for real speedup.")


# ---------------------------------------------------------------------------
# bitsandbytes demo (if available)
# ---------------------------------------------------------------------------

def bnb_demo():
    """Demo bitsandbytes quantization if installed."""
    if not HAS_BNB:
        print(f"\n  bitsandbytes not installed.")
        print(f"  Install with: pip install bitsandbytes")
        print(f"\n  Showing theoretical analysis instead:")
        print(f"\n  LLM.int8() for a 1B parameter model:")
        params = 1e9
        fp16_gb = params * 2 / 1e9
        int8_gb = params * 1 / 1e9
        nf4_gb = params * 0.5 / 1e9
        print(f"    FP16: {fp16_gb:.1f} GB")
        print(f"    INT8: {int8_gb:.1f} GB ({fp16_gb/int8_gb:.1f}x reduction)")
        print(f"    NF4:  {nf4_gb:.1f} GB ({fp16_gb/nf4_gb:.1f}x reduction)")
        return

    import bitsandbytes as bnb

    print(f"\n  bitsandbytes {bnb.__version__} demo")

    # INT8 linear
    in_f, out_f = 1024, 2048
    linear_fp16 = nn.Linear(in_f, out_f, bias=False, dtype=torch.float16)

    linear_int8 = bnb.nn.Linear8bitLt(in_f, out_f, bias=False, has_fp16_weights=False)
    linear_int8.weight = bnb.nn.Int8Params(
        linear_fp16.weight.data.clone(), requires_grad=False
    )

    if torch.cuda.is_available():
        linear_fp16 = linear_fp16.cuda()
        linear_int8 = linear_int8.cuda()
        x = torch.randn(8, 256, in_f, device="cuda", dtype=torch.float16)

        # Benchmark
        from utils.benchmarking import benchmark_fn

        r_fp16 = benchmark_fn(lambda: linear_fp16(x), warmup_steps=10,
                              measure_steps=30, name="fp16", sync_cuda=True)
        r_int8 = benchmark_fn(lambda: linear_int8(x), warmup_steps=10,
                              measure_steps=30, name="int8", sync_cuda=True)

        print(f"  FP16 Linear: {r_fp16.mean_ms:.3f} ms")
        print(f"  INT8 Linear: {r_int8.mean_ms:.3f} ms")
        print(f"  Speedup: {r_fp16.mean_ms / r_int8.mean_ms:.2f}x")


# ---------------------------------------------------------------------------
# Choosing the right method
# ---------------------------------------------------------------------------

def recommendation_guide():
    """Print a decision guide for choosing quantization method."""
    print(f"""
  ================================================================
  QUANTIZATION DECISION GUIDE
  ================================================================

  Q: What's your deployment target?

  ┌─ GPU Inference
  │   ├─ Want easiest setup?
  │   │   └─ bitsandbytes NF4 (one HuggingFace flag)
  │   ├─ Want best quality?
  │   │   └─ GPTQ with calibration data
  │   ├─ Want fastest inference?
  │   │   └─ AWQ with Marlin kernels
  │   └─ Need dynamic quantization (no calibration)?
  │       └─ bitsandbytes LLM.int8()
  │
  └─ CPU Inference
      ├─ Want best quality/speed tradeoff?
      │   └─ GGUF Q4_K_M or Q5_K_M
      ├─ Want smallest model?
      │   └─ GGUF Q2_K
      └─ Want best quality?
          └─ GGUF Q8_0

  General rules:
    - Always start with FP16/BF16 (free 2x memory reduction)
    - INT8 gives excellent quality with 4x memory savings
    - INT4 gives good quality with 8x memory savings
    - Below INT4, quality degrades noticeably
    - Calibration-based methods (GPTQ, AWQ) beat naive quantization
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("  Chapter 7: Production Quantization Survey")
    print("=" * 70)

    print(f"\n  Available libraries:")
    print(f"    bitsandbytes: {'YES' if HAS_BNB else 'NO'}")
    print(f"    auto-gptq:    {'YES' if HAS_GPTQ else 'NO'}")
    print(f"    autoawq:      {'YES' if HAS_AWQ else 'NO'}")

    print("\n  [1] Survey of methods")
    survey()

    print("\n  [2] Comparison table")
    comparison_table()

    print("\n  [3] bitsandbytes demo")
    bnb_demo()

    print("\n  [4] Choosing the right method")
    recommendation_guide()
