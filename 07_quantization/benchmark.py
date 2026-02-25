"""
Chapter 7 — Quantization
File: benchmark.py

Unified quantization benchmark:
  - FP32 / FP16 / INT8 (naive) / INT4 (naive)
  - Memory, latency, and quality metrics
"""

import sys
import os
import time
import argparse
import copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, str(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.model_loader import create_model, generate_random_batch
from utils.benchmarking import benchmark_fn
from apply_quantization import quantize_model


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    num_layers: int = 4,
    hidden_dim: int = 1024,
    num_heads: int = 8,
    batch_size: int = 8,
    seq_len: int = 256,
    device: str = None,
    warmup: int = 5,
    steps: int = 20,
):
    """Comprehensive quantization benchmark."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    has_cuda = "cuda" in device

    print(f"\n{'=' * 80}")
    print(f"  Chapter 7: Quantization Benchmark")
    print(f"  {num_layers} layers, hidden={hidden_dim}, nH={num_heads}")
    print(f"  batch={batch_size}, seq_len={seq_len}, device={device}")
    print(f"{'=' * 80}")

    input_ids = generate_random_batch(batch_size, seq_len, device=device)

    # Create reference model for quality comparison
    model_ref = create_model(
        num_layers=num_layers, hidden_dim=hidden_dim,
        num_heads=num_heads, dtype=torch.float32, device=device,
    )
    model_ref.eval()

    with torch.no_grad():
        out_ref = model_ref(input_ids).float()

    configs = [
        ("FP32", torch.float32, None, None),
        ("FP16", torch.float16, None, None),
        ("INT8", torch.float32, 8, 128),
        ("INT4 (g=128)", torch.float32, 4, 128),
        ("INT4 (g=32)", torch.float32, 4, 32),
    ]

    results = []

    for name, dtype, quant_bits, group_size in configs:
        # Build model
        if quant_bits is None and dtype == torch.float32:
            model = model_ref
        elif quant_bits is None:
            model = create_model(
                num_layers=num_layers, hidden_dim=hidden_dim,
                num_heads=num_heads, dtype=dtype, device=device,
            )
            # Copy weights for fair quality comparison
            model.load_state_dict(
                {k: v.to(dtype) for k, v in model_ref.state_dict().items()}
            )
        else:
            model = copy.deepcopy(model_ref)
            model = quantize_model(model, bits=quant_bits, group_size=group_size or 128,
                                   skip_patterns=["tok_emb", "lm_head"])

        model.eval()

        # Memory
        total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        total_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
        mem_mb = total_bytes / (1024 ** 2)

        # Latency
        if has_cuda:
            torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            r = benchmark_fn(
                lambda: model(input_ids),
                warmup_steps=warmup, measure_steps=steps,
                name=name, sync_cuda=has_cuda,
            )

        gpu_mem = torch.cuda.max_memory_allocated() / (1024**2) if has_cuda else 0

        # Quality
        with torch.no_grad():
            out = model(input_ids).float()
        error = out - out_ref
        mse = error.pow(2).mean().item()
        cos_sim = F.cosine_similarity(
            out.reshape(-1).unsqueeze(0),
            out_ref.reshape(-1).unsqueeze(0),
        ).item()

        results.append({
            "name": name,
            "latency_ms": r.mean_ms,
            "model_mb": mem_mb,
            "gpu_mem_mb": gpu_mem,
            "mse": mse,
            "cos_sim": cos_sim,
        })

        if model is not model_ref:
            del model
        if has_cuda:
            torch.cuda.empty_cache()

    # Print results table
    fp32_ms = results[0]["latency_ms"]
    fp32_mb = results[0]["model_mb"]

    print(f"\n  {'Config':<15}  {'Latency':>10}  {'Model MB':>10}  {'GPU MB':>10}  "
          f"{'Speedup':>9}  {'Compress':>10}  {'MSE':>12}  {'Cos Sim':>10}")
    print(f"  {'-' * 15}  {'-' * 10}  {'-' * 10}  {'-' * 10}  "
          f"{'-' * 9}  {'-' * 10}  {'-' * 12}  {'-' * 10}")

    for r in results:
        speedup = fp32_ms / r["latency_ms"] if r["latency_ms"] > 0 else 0
        compress = fp32_mb / r["model_mb"] if r["model_mb"] > 0 else 0
        print(f"  {r['name']:<15}  {r['latency_ms']:>9.2f}ms  {r['model_mb']:>10.1f}  "
              f"{r['gpu_mem_mb']:>10.1f}  {speedup:>8.2f}x  {compress:>9.1f}x  "
              f"{r['mse']:>12.6f}  {r['cos_sim']:>10.6f}")

    del model_ref
    if has_cuda:
        torch.cuda.empty_cache()

    print(f"\n  Notes:")
    print(f"  - Naive INT8/INT4 use dequantize-on-the-fly (adds overhead)")
    print(f"  - Production INT4 kernels (GPTQ/AWQ) achieve real speedups")
    print(f"  - Quality: cosine similarity > 0.999 is typically acceptable")
    print(f"  - Memory compression is the primary benefit for serving")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chapter 7: Quantization Benchmark")
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_benchmark(
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device=args.device,
    )
