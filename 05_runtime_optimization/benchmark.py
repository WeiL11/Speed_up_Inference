"""
Chapter 5 — Runtime Optimization
File: benchmark.py

Unified benchmark: measure latency before/after each runtime optimization,
then show cumulative speedup.

Optimizations tested:
  1. Baseline (FP32, eager mode)
  2. Mixed precision (BF16)
  3. torch.compile (inductor)
  4. CUDA graphs (static computation graph replay)
  5. Operator fusion (manual JIT fusion)
"""

import sys
import os
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, str(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from utils.model_loader import create_model, generate_random_batch
from utils.benchmarking import benchmark_fn


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    num_layers: int = 4,
    hidden_dim: int = 1024,
    num_heads: int = 8,
    batch_size: int = 8,
    seq_len: int = 256,
    device: str = None,
    warmup: int = 10,
    steps: int = 30,
):
    """Run cumulative optimization benchmark."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    has_cuda = "cuda" in device

    print(f"\n{'=' * 75}")
    print(f"  Chapter 5: Runtime Optimization Benchmark")
    print(f"  {num_layers}L, hidden={hidden_dim}, nH={num_heads}")
    print(f"  batch={batch_size}, seq_len={seq_len}, device={device}")
    print(f"{'=' * 75}")

    results = []
    input_ids = generate_random_batch(batch_size, seq_len, device=device)

    # ---- Step 0: FP32 Baseline ----
    model_fp32 = create_model(
        num_layers=num_layers, hidden_dim=hidden_dim,
        num_heads=num_heads, dtype=torch.float32, device=device,
    )
    model_fp32.eval()

    with torch.no_grad():
        r = benchmark_fn(lambda: model_fp32(input_ids), warmup_steps=warmup,
                         measure_steps=steps, name="FP32 Baseline", sync_cuda=has_cuda)
    if has_cuda:
        mem_fp32 = torch.cuda.max_memory_allocated() / (1024**2)
    else:
        mem_fp32 = 0
    results.append(("FP32 Baseline", r.mean_ms, mem_fp32))
    baseline_ms = r.mean_ms

    # ---- Step 1: BF16 / FP16 ----
    if has_cuda:
        dtype_opt = torch.bfloat16
        dtype_name = "BF16"
    else:
        dtype_opt = torch.float32
        dtype_name = "FP32 (no GPU)"

    model_bf16 = create_model(
        num_layers=num_layers, hidden_dim=hidden_dim,
        num_heads=num_heads, dtype=dtype_opt, device=device,
    )
    model_bf16.eval()

    if has_cuda:
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        r = benchmark_fn(lambda: model_bf16(input_ids), warmup_steps=warmup,
                         measure_steps=steps, name=dtype_name, sync_cuda=has_cuda)
    mem_bf16 = torch.cuda.max_memory_allocated() / (1024**2) if has_cuda else 0
    results.append((f"+ {dtype_name}", r.mean_ms, mem_bf16))

    # ---- Step 2: torch.compile ----
    compile_ms = r.mean_ms  # default to previous if compile fails
    compile_mem = mem_bf16
    try:
        compiled_model = torch.compile(model_bf16, mode="reduce-overhead")
        # Warmup for compilation
        with torch.no_grad():
            for _ in range(5):
                _ = compiled_model(input_ids)
                if has_cuda:
                    torch.cuda.synchronize()

        if has_cuda:
            torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            r = benchmark_fn(lambda: compiled_model(input_ids), warmup_steps=warmup,
                             measure_steps=steps, name="+ compile", sync_cuda=has_cuda)
        compile_ms = r.mean_ms
        compile_mem = torch.cuda.max_memory_allocated() / (1024**2) if has_cuda else 0
    except Exception as e:
        print(f"  (torch.compile skipped: {e})")
    results.append(("+ torch.compile", compile_ms, compile_mem))

    # ---- Step 3: CUDA Graphs ----
    graph_ms = compile_ms
    graph_mem = compile_mem
    if has_cuda:
        try:
            # Capture CUDA graph
            static_input = input_ids.clone()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())

            with torch.cuda.stream(s):
                for _ in range(3):
                    with torch.no_grad():
                        _ = model_bf16(static_input)
            torch.cuda.current_stream().wait_stream(s)

            g = torch.cuda.CUDAGraph()
            with torch.no_grad():
                with torch.cuda.graph(g):
                    static_output = model_bf16(static_input)

            def run_graph():
                static_input.copy_(input_ids)
                g.replay()
                return static_output

            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                r = benchmark_fn(run_graph, warmup_steps=warmup,
                                 measure_steps=steps, name="+ CUDA graph", sync_cuda=True)
            graph_ms = r.mean_ms
            graph_mem = torch.cuda.max_memory_allocated() / (1024**2)
        except Exception as e:
            print(f"  (CUDA graphs skipped: {e})")
    results.append(("+ CUDA Graphs", graph_ms, graph_mem))

    # ---- Results table ----
    print(f"\n  {'Optimization':<20}  {'Latency (ms)':>14}  {'Memory (MB)':>13}  "
          f"{'Speedup':>10}  {'Cumulative':>12}")
    print(f"  {'-' * 20}  {'-' * 14}  {'-' * 13}  {'-' * 10}  {'-' * 12}")

    prev_ms = baseline_ms
    for name, ms, mem in results:
        step_speedup = prev_ms / ms if ms > 0 else 0
        cumulative = baseline_ms / ms if ms > 0 else 0
        print(f"  {name:<20}  {ms:>14.2f}  {mem:>13.1f}  "
              f"{step_speedup:>9.2f}x  {cumulative:>11.2f}x")
        prev_ms = ms

    total_speedup = baseline_ms / results[-1][1] if results[-1][1] > 0 else 0
    print(f"\n  Total speedup: {total_speedup:.2f}x over FP32 baseline")

    # Cleanup
    del model_fp32, model_bf16
    if has_cuda:
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chapter 5: Runtime Optimization Benchmark")
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()

    run_benchmark(
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device=args.device,
        warmup=args.warmup,
        steps=args.steps,
    )
