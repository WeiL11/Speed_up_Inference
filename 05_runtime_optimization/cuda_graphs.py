"""
CUDA Graph capture and replay for transformer inference.

Concept
-------
A CUDA Graph records a sequence of GPU operations as a static execution graph,
then replays them with a single CPU call.  This eliminates:
  - Python interpreter overhead per step
  - CUDA kernel launch latency (enqueue overhead)
  - Synchronization points between kernels

The graph replay dispatches all recorded operations in a single shot, letting the
GPU run continuously without waiting for the CPU.

Critical constraints
--------------------
1. Input tensor must be the SAME OBJECT across calls — you copy new data INTO the
   static buffer rather than passing a new tensor.  The graph records the memory
   address, not the value.

2. Graph capture must happen AFTER warmup — lazy initialization (cuBLAS handle
   creation, workspace allocation, etc.) would otherwise be captured and replayed
   unnecessarily.

3. Only works for STATIC shapes — every tensor operation in the graph must have
   the exact same shape on every replay.  This is why CUDA graphs are ideal for
   the DECODE phase (one token at a time, fixed shape) but NOT for the PREFILL
   phase (variable sequence lengths).

Prefill vs Decode
-----------------
  Prefill  : process the prompt — sequence length changes per request → dynamic
              shapes → CUDA graphs cannot be used directly (see CUDA graph pools
              or cudaStreamCapture with shape bucketing as an advanced workaround).

  Decode   : generate one token per step — shape is always (batch, 1) → static →
              CUDA graphs eliminate the Python overhead that otherwise dominates
              single-token latency.

torch.compile + CUDA graphs
---------------------------
torch.compile with mode="reduce-overhead" automatically inserts CUDA graph
capture/replay internally.  You can also combine them manually: compile first,
warm up the compiled model, then capture its graph.
"""

import sys
import time
from typing import Optional, Tuple

sys.path.insert(0, "/home/user/Speed_up_Inference")

import torch
import torch.nn as nn

from utils.benchmarking import BenchmarkResult, benchmark_fn, print_gpu_info
from utils.model_loader import SimpleTransformer, create_model, generate_random_batch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH_SIZE = 4
SEQ_LEN = 32       # Short — realistic for single decode step (static shape)
WARMUP_STEPS = 5
MEASURE_STEPS = 50


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_ms(fn) -> float:
    """Run fn once, return elapsed ms."""
    sync()
    t0 = time.perf_counter()
    fn()
    sync()
    return (time.perf_counter() - t0) * 1000.0


def measure_latency(fn, warmup: int = WARMUP_STEPS, steps: int = MEASURE_STEPS) -> float:
    """Return mean latency in ms over `steps` runs after `warmup` warm-up steps."""
    for _ in range(warmup):
        fn()
        sync()
    times = [timed_ms(fn) for _ in range(steps)]
    return sum(times) / len(times)


# ---------------------------------------------------------------------------
# Core CUDA graph API
# ---------------------------------------------------------------------------

def capture_cuda_graph(
    model: nn.Module,
    input_ids: torch.Tensor,
    warmup_steps: int = WARMUP_STEPS,
) -> Tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor]:
    """
    Capture the forward pass as a CUDA graph.

    Steps:
      1. Run `warmup_steps` forward passes OUTSIDE the capture to ensure all
         lazy CUDA operations (cuBLAS handle init, workspace alloc) are complete.
      2. Allocate a static input buffer (same object reused every replay).
      3. Use torch.cuda.graph() context manager to record all GPU ops.
      4. Return the graph, the static input buffer, and the static output tensor.

    The caller must use `run_with_cuda_graph` (below) for subsequent calls.

    Args:
        model      : Eval-mode model already on CUDA.
        input_ids  : Example input used to set shapes.  Content will be copied
                     into the static buffer.
        warmup_steps: Number of un-captured warm-up iterations.

    Returns:
        (graph, static_input, static_output)
    """
    assert torch.cuda.is_available(), "CUDA graphs require a CUDA device."
    model.eval()

    # --- Step 1: Warm-up (not captured) ---
    # This ensures cuBLAS/cuDNN handles and workspace tensors are already
    # allocated.  If we skip this, those allocation calls would be captured
    # in the graph and re-executed on every replay — wasting time.
    print(f"    Warming up ({warmup_steps} steps, not captured)...")
    with torch.no_grad():
        for _ in range(warmup_steps):
            _ = model(input_ids)
    sync()

    # --- Step 2: Allocate the static input buffer ---
    # CRITICAL: This is the tensor whose MEMORY ADDRESS is recorded in the graph.
    # On every replay we copy fresh data into this buffer — we never create a
    # new tensor or pass a different object.
    static_input = input_ids.clone()

    # --- Step 3: Capture ---
    print("    Capturing CUDA graph...")
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        with torch.no_grad():
            static_output = model(static_input)

    sync()
    print("    Capture complete.")
    return g, static_input, static_output


def run_with_cuda_graph(
    graph: torch.cuda.CUDAGraph,
    static_input: torch.Tensor,
    static_output: torch.Tensor,
    new_input: torch.Tensor,
) -> torch.Tensor:
    """
    Copy new data into the static buffer, replay the graph, return a copy of
    the output.

    Why clone the output?
    The static_output tensor is overwritten on every replay.  If the caller
    holds a reference from a previous call, they need their own copy.

    Args:
        graph         : Captured CUDAGraph.
        static_input  : The SAME tensor object used during capture.
        static_output : The SAME tensor object produced during capture.
        new_input     : New token IDs to process (same shape as static_input).

    Returns:
        Clone of static_output (safe to use after next replay).
    """
    # Copy new data into the static buffer without creating a new tensor
    static_input.copy_(new_input)
    graph.replay()
    return static_output.clone()


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------

def bench_baseline(model: nn.Module, input_ids: torch.Tensor) -> float:
    """Standard forward pass, no compile, no CUDA graph."""
    model.eval()
    fn = lambda: model(input_ids)
    with torch.no_grad():
        latency = measure_latency(fn)
    return latency


def bench_cuda_graph(model: nn.Module, input_ids: torch.Tensor) -> float:
    """CUDA graph forward pass."""
    g, static_in, static_out = capture_cuda_graph(model, input_ids)
    new_ids = generate_random_batch(
        input_ids.size(0), input_ids.size(1),
        model.config.vocab_size, input_ids.device
    )
    fn = lambda: run_with_cuda_graph(g, static_in, static_out, new_ids)
    latency = measure_latency(fn)
    return latency


def bench_compile_only(model: nn.Module, input_ids: torch.Tensor) -> float:
    """torch.compile only (inductor, default mode)."""
    compiled = torch.compile(model, mode="default")
    compiled.eval()
    # Trigger compilation (first call)
    with torch.no_grad():
        _ = compiled(input_ids)
    fn = lambda: compiled(input_ids)
    with torch.no_grad():
        latency = measure_latency(fn)
    return latency


def bench_compile_and_cuda_graph(model: nn.Module, input_ids: torch.Tensor) -> float:
    """
    torch.compile + CUDA graph.

    Order matters:
      1. Compile first (torch.compile wraps the model).
      2. Warm up the compiled model so Triton kernels are generated.
      3. Capture the graph of the compiled model.

    This gives you both JIT-compiled Triton kernels AND static graph replay.
    """
    compiled = torch.compile(model, mode="default")
    compiled.eval()

    # Extra warmup for compilation (Triton codegen)
    print("    Compiling (first forward pass)...")
    with torch.no_grad():
        for _ in range(3):
            _ = compiled(input_ids)
    sync()

    # Now capture the compiled graph
    g, static_in, static_out = capture_cuda_graph(compiled, input_ids, warmup_steps=3)
    new_ids = generate_random_batch(
        input_ids.size(0), input_ids.size(1),
        model.config.vocab_size, input_ids.device
    )
    fn = lambda: run_with_cuda_graph(g, static_in, static_out, new_ids)
    latency = measure_latency(fn)
    return latency


# ---------------------------------------------------------------------------
# Prefill limitation demo
# ---------------------------------------------------------------------------

def demo_prefill_limitation(model: nn.Module, device: str) -> None:
    """
    Show why CUDA graphs cannot be used for prefill (variable-length inputs).

    We capture a graph for shape (batch=2, seq=64) and then try to replay with
    shape (batch=2, seq=128).  This should fail or produce wrong results because
    the graph was compiled for a specific shape.
    """
    print()
    print("--- Prefill Limitation Demo ---")
    print("  CUDA graphs require STATIC shapes.")
    print("  Prefill processes prompts with variable lengths => cannot use static graphs.")
    print()

    seq_capture = 64
    seq_wrong   = 128

    ids_capture = generate_random_batch(2, seq_capture, model.config.vocab_size, device)
    ids_wrong   = generate_random_batch(2, seq_wrong,   model.config.vocab_size, device)

    g, static_in, static_out = capture_cuda_graph(model, ids_capture, warmup_steps=2)

    print(f"  Graph captured for shape {tuple(ids_capture.shape)}")
    print(f"  Attempting to replay with shape {tuple(ids_wrong.shape)}...")

    try:
        # Trying to copy a tensor of different shape into static_in will fail
        static_in.copy_(ids_wrong)
        g.replay()
        print("  Result: did not raise — but output is UNDEFINED (shape mismatch).")
    except Exception as e:
        print(f"  Result: RuntimeError as expected — {type(e).__name__}: {e}")

    print()
    print("  Solutions for prefill with varying shapes:")
    print("  1. Use separate graphs per shape bucket (e.g., 64, 128, 256, 512).")
    print("  2. Use torch.compile(mode='reduce-overhead') which handles this via")
    print("     internal graph caching with shape guards.")
    print("  3. Use CUDA stream capture with dynamic graph re-compilation.")
    print("  4. Accept non-graph execution for prefill (it's not the bottleneck).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("Chapter 5 — CUDA Graphs Demo")
    print("=" * 70)
    print()

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available.  CUDA graphs require a CUDA device.")
        print("Exiting — no CPU fallback for CUDA graphs.")
        return

    print_gpu_info()
    print()

    device = "cuda"
    model = create_model(
        num_layers=8,
        hidden_dim=1024,
        num_heads=8,
        num_kv_heads=4,
        device=device,
        dtype=torch.float32,
    )
    model.eval()
    print(f"Model: {model.param_count_str()} params")
    print()

    input_ids = generate_random_batch(BATCH_SIZE, SEQ_LEN, model.config.vocab_size, device)

    results: dict[str, float] = {}

    # -----------------------------------------------------------------------
    # 1. Baseline
    # -----------------------------------------------------------------------
    print("Benchmark 1: Baseline (no compile, no CUDA graph)")
    results["baseline"] = bench_baseline(model, input_ids)
    print(f"  Mean latency: {results['baseline']:.3f} ms")
    print()

    # -----------------------------------------------------------------------
    # 2. CUDA graph only
    # -----------------------------------------------------------------------
    print("Benchmark 2: CUDA graph (no compile)")
    results["cuda_graph"] = bench_cuda_graph(model, input_ids)
    print(f"  Mean latency: {results['cuda_graph']:.3f} ms")
    print()

    # -----------------------------------------------------------------------
    # 3. torch.compile only
    # -----------------------------------------------------------------------
    torch_version = tuple(int(x) for x in torch.__version__.split(".")[:2])
    if torch_version >= (2, 0):
        print("Benchmark 3: torch.compile only (mode='default')")
        results["compile"] = bench_compile_only(model, input_ids)
        print(f"  Mean latency: {results['compile']:.3f} ms")
        print()

        # -------------------------------------------------------------------
        # 4. torch.compile + CUDA graph
        # -------------------------------------------------------------------
        print("Benchmark 4: torch.compile + CUDA graph")
        results["compile_cuda_graph"] = bench_compile_and_cuda_graph(model, input_ids)
        print(f"  Mean latency: {results['compile_cuda_graph']:.3f} ms")
        print()
    else:
        print(f"torch.compile not available (PyTorch {torch.__version__} < 2.0) — skipping.")

    # -----------------------------------------------------------------------
    # 5. Summary
    # -----------------------------------------------------------------------
    baseline_ms = results["baseline"]
    print()
    print("=" * 65)
    print(f"{'Configuration':<30} {'Latency (ms)':>14} {'Speedup':>10}")
    print("-" * 65)
    for name, lat in results.items():
        speedup = baseline_ms / lat
        print(f"  {name:<28} {lat:>14.3f} {speedup:>9.2f}x")
    print("=" * 65)
    print()
    print("Analysis:")
    print("  - CUDA graphs eliminate per-step Python overhead and kernel launch latency.")
    print("  - For decode (single-token steps), the CPU overhead is often the bottleneck.")
    print("  - Combining compile + CUDA graph removes both Python and compute inefficiency.")
    print("  - CUDA graphs are NOT suitable for prefill (dynamic shapes).")
    print()

    # -----------------------------------------------------------------------
    # 6. Prefill limitation
    # -----------------------------------------------------------------------
    demo_prefill_limitation(model, device)


if __name__ == "__main__":
    main()
