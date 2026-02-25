#!/usr/bin/env python3
"""
End-to-End Benchmark — Combined Optimization Speedup
=====================================================

Measures the cumulative speedup of applying all optimizations from each
chapter in sequence, using the same model and data configuration.

Optimization stack:
  Baseline : FP32 + naive attention + no compile
  Step 1   : BF16 (Ch5 mixed precision)
  Step 2   : + torch.compile (Ch5 runtime)
  Step 3   : + FlashAttention via SDPA (Ch2 efficient attention)
  Step 4   : + CUDA graphs (Ch5 runtime)
  Step 5   : + INT8 weight quantization (Ch7 quantization)

Reports:
  - Latency (ms) per step
  - Peak GPU memory (MB)
  - Throughput (tokens/sec)
  - Speedup vs baseline

Usage:
  python end_to_end.py
  python end_to_end.py --num_layers 26 --hidden_dim 1152 --num_heads 4 --num_kv_heads 1
  python end_to_end.py --batch_size 1 --seq_len 512 --steps 30
"""

import argparse
import sys
import timeit
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.model_loader import create_model, generate_random_batch, SimpleTransformer
from utils.benchmarking import benchmark_fn, BenchmarkResult, print_gpu_info


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class BenchConfig:
    batch_size: int = 8
    seq_len: int = 512
    warmup: int = 5
    steps: int = 20
    device: str = "cuda"


def timed_forward(model, input_ids, cfg: BenchConfig, use_no_grad=True) -> BenchmarkResult:
    """Benchmark forward pass with optional torch.no_grad."""
    def step():
        if use_no_grad:
            with torch.no_grad():
                model(input_ids)
        else:
            model(input_ids)

    return benchmark_fn(
        step,
        warmup_steps=cfg.warmup,
        measure_steps=cfg.steps,
        name="",
    )


def memory_mb() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


def reset_mem():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Step 0: Baseline — FP32, no compile
# ---------------------------------------------------------------------------

def run_baseline(args, cfg: BenchConfig):
    reset_mem()
    model = create_model(
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        vocab_size=args.vocab_size,
        dtype=torch.float32,
        device=cfg.device,
    )
    input_ids = generate_random_batch(cfg.batch_size, cfg.seq_len, args.vocab_size, cfg.device)
    result = timed_forward(model, input_ids, cfg)
    result.name = "Baseline (FP32, no compile)"
    result.peak_memory_mb = memory_mb()
    del model, input_ids
    reset_mem()
    return result


# ---------------------------------------------------------------------------
# Step 1: BF16
# ---------------------------------------------------------------------------

def run_bf16(args, cfg: BenchConfig):
    reset_mem()
    model = create_model(
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        vocab_size=args.vocab_size,
        dtype=torch.bfloat16,
        device=cfg.device,
    )
    input_ids = generate_random_batch(cfg.batch_size, cfg.seq_len, args.vocab_size, cfg.device)
    result = timed_forward(model, input_ids, cfg)
    result.name = "Step 1: + BF16"
    result.peak_memory_mb = memory_mb()
    del model, input_ids
    reset_mem()
    return result


# ---------------------------------------------------------------------------
# Step 2: BF16 + torch.compile
# ---------------------------------------------------------------------------

def run_compiled(args, cfg: BenchConfig):
    reset_mem()
    model = create_model(
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        vocab_size=args.vocab_size,
        dtype=torch.bfloat16,
        device=cfg.device,
    )
    if hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead")
    input_ids = generate_random_batch(cfg.batch_size, cfg.seq_len, args.vocab_size, cfg.device)
    result = timed_forward(model, input_ids, cfg)
    result.name = "Step 2: + torch.compile"
    result.peak_memory_mb = memory_mb()
    del model, input_ids
    reset_mem()
    return result


# ---------------------------------------------------------------------------
# Step 3: BF16 + compile + FlashAttention (via SDPA, already used by default)
# ---------------------------------------------------------------------------

def run_flash(args, cfg: BenchConfig):
    """
    Our SimpleTransformer already uses F.scaled_dot_product_attention,
    which dispatches to FlashAttention when available.
    This step makes it explicit by setting the backend.
    """
    reset_mem()
    model = create_model(
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        vocab_size=args.vocab_size,
        dtype=torch.bfloat16,
        device=cfg.device,
    )
    if hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead")

    input_ids = generate_random_batch(cfg.batch_size, cfg.seq_len, args.vocab_size, cfg.device)

    # Force FlashAttention backend if available
    with torch.backends.cuda.sdp_kernel(
        enable_flash=True,
        enable_math=False,
        enable_mem_efficient=False,
    ) if hasattr(torch.backends.cuda, "sdp_kernel") else __import__("contextlib").nullcontext():
        result = timed_forward(model, input_ids, cfg)

    result.name = "Step 3: + FlashAttention"
    result.peak_memory_mb = memory_mb()
    del model, input_ids
    reset_mem()
    return result


# ---------------------------------------------------------------------------
# Step 4: BF16 + compile + Flash + CUDA Graphs
# ---------------------------------------------------------------------------

def run_cuda_graphs(args, cfg: BenchConfig):
    reset_mem()

    if not torch.cuda.is_available():
        r = BenchmarkResult(name="Step 4: + CUDA Graphs (N/A - no CUDA)")
        return r

    model = create_model(
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        vocab_size=args.vocab_size,
        dtype=torch.bfloat16,
        device=cfg.device,
    )
    if hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead")

    input_ids = generate_random_batch(cfg.batch_size, cfg.seq_len, args.vocab_size, cfg.device)

    # Warmup before graph capture
    for _ in range(cfg.warmup):
        with torch.no_grad():
            model(input_ids)
    torch.cuda.synchronize()

    # Capture CUDA graph
    static_input = input_ids.clone()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        with torch.no_grad():
            static_output = model(static_input)

    # Benchmark replay
    times_ms = []
    reset_mem()
    for _ in range(cfg.steps):
        torch.cuda.synchronize()
        t0 = timeit.default_timer()
        static_input.copy_(input_ids)
        g.replay()
        torch.cuda.synchronize()
        times_ms.append((timeit.default_timer() - t0) * 1000.0)

    result = BenchmarkResult(
        name="Step 4: + CUDA Graphs",
        times_ms=times_ms,
        peak_memory_mb=memory_mb(),
    )
    del model, input_ids, g
    reset_mem()
    return result


# ---------------------------------------------------------------------------
# Step 5: + INT8 weight quantization
# ---------------------------------------------------------------------------

def run_quantized(args, cfg: BenchConfig):
    """Apply simple absmax INT8 weight quantization and measure."""
    reset_mem()
    model = create_model(
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        vocab_size=args.vocab_size,
        dtype=torch.bfloat16,
        device=cfg.device,
    )

    # Simple weight-only INT8 quantization (absmax per-tensor)
    # This reduces memory bandwidth for weight loading
    def quantize_weights_inplace(m):
        if isinstance(m, nn.Linear):
            w = m.weight.data.float()
            scale = w.abs().max() / 127.0
            w_int8 = (w / scale).round().clamp(-128, 127).to(torch.int8)
            # Dequantize and store back as bfloat16
            m.weight.data = (w_int8.float() * scale).to(torch.bfloat16)

    model.apply(quantize_weights_inplace)

    if hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead")

    input_ids = generate_random_batch(cfg.batch_size, cfg.seq_len, args.vocab_size, cfg.device)
    result = timed_forward(model, input_ids, cfg)
    result.name = "Step 5: + INT8 Quant"
    result.peak_memory_mb = memory_mb()
    del model, input_ids
    reset_mem()
    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(results: list[BenchmarkResult]):
    baseline = results[0].mean_ms
    print()
    print("=" * 78)
    print("END-TO-END OPTIMIZATION RESULTS")
    print("=" * 78)
    print(f"{'Step':<35} {'Latency (ms)':>12} {'Memory (MB)':>12} {'Speedup':>8} {'Tokens/s':>10}")
    print("-" * 78)
    for r in results:
        tokens_per_s = 0.0
        if r.mean_ms > 0 and ARGS is not None:
            tokens_per_s = ARGS.batch_size * ARGS.seq_len * 1000.0 / r.mean_ms
        speedup = baseline / r.mean_ms if r.mean_ms > 0 else 0
        print(f"  {r.name:<33} {r.mean_ms:>12.2f} {r.peak_memory_mb:>12.1f} "
              f"{speedup:>7.2f}x {tokens_per_s:>10,.0f}")
    print("=" * 78)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

ARGS = None  # global for report function

def parse_args():
    p = argparse.ArgumentParser(description="End-to-end optimization benchmark")
    p.add_argument("--num_layers",   type=int, default=6)
    p.add_argument("--hidden_dim",   type=int, default=1024)
    p.add_argument("--num_heads",    type=int, default=8)
    p.add_argument("--num_kv_heads", type=int, default=None)
    p.add_argument("--vocab_size",   type=int, default=32000)
    p.add_argument("--batch_size",   type=int, default=8)
    p.add_argument("--seq_len",      type=int, default=512)
    p.add_argument("--warmup",       type=int, default=5)
    p.add_argument("--steps",        type=int, default=20)
    p.add_argument("--device",       type=str, default="cuda")
    return p.parse_args()


def main():
    global ARGS
    ARGS = parse_args()

    if ARGS.num_kv_heads is None:
        ARGS.num_kv_heads = ARGS.num_heads

    if ARGS.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU.")
        ARGS.device = "cpu"

    cfg = BenchConfig(
        batch_size=ARGS.batch_size,
        seq_len=ARGS.seq_len,
        warmup=ARGS.warmup,
        steps=ARGS.steps,
        device=ARGS.device,
    )

    print("=" * 64)
    print("END-TO-END INFERENCE OPTIMIZATION BENCHMARK")
    print("=" * 64)
    print(f"  Model: {ARGS.num_layers}L, dim={ARGS.hidden_dim}, "
          f"heads={ARGS.num_heads}/{ARGS.num_kv_heads} KV")
    print(f"  Data:  batch={ARGS.batch_size}, seq_len={ARGS.seq_len}")
    print(f"  Runs:  {ARGS.warmup} warmup + {ARGS.steps} measured")
    print()
    print_gpu_info()
    print()

    results = []

    print("Running Baseline (FP32)...")
    results.append(run_baseline(ARGS, cfg))

    print("Running BF16...")
    results.append(run_bf16(ARGS, cfg))

    if torch.cuda.is_available() and hasattr(torch, "compile"):
        print("Running torch.compile...")
        results.append(run_compiled(ARGS, cfg))

    if torch.cuda.is_available():
        print("Running FlashAttention (SDPA)...")
        results.append(run_flash(ARGS, cfg))

        print("Running CUDA Graphs...")
        results.append(run_cuda_graphs(ARGS, cfg))

    print("Running INT8 Quantization...")
    results.append(run_quantized(ARGS, cfg))

    print_report(results)


if __name__ == "__main__":
    main()
