"""
torch.compile modes and their effect on transformer inference speed.

torch.compile modes:
  default        - standard compilation with moderate optimization
  reduce-overhead - minimize Python/CUDA launch overhead (good for small batches)
  max-autotune   - tune kernel parameters exhaustively (slow compile, fast runtime)

Backends:
  inductor  - default, generates C++/Triton code
  eager     - essentially no-op, used for comparison
  aot_eager - Ahead-of-Time eager, useful for debugging

Key insight: The first call after torch.compile is slow because compilation happens
lazily (just-in-time). Subsequent calls are fast. Warm-up is essential before
measuring steady-state performance.

Additional options:
  fullgraph=True  - require the entire model to be one graph (no Python fallback);
                    raises an error if graph breaks are detected.
  dynamic=True    - compile for dynamic shapes (less specialization, more flexible).
"""

import sys
import time
import timeit
from typing import Optional

sys.path.insert(0, "/home/user/Speed_up_Inference")

import torch
import torch.nn as nn

from utils.benchmarking import BenchmarkResult, benchmark_fn, print_gpu_info
from utils.model_loader import SimpleTransformer, TransformerConfig, create_model, generate_random_batch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH_SIZE = 4
SEQ_LEN = 128
WARMUP_STEPS = 5
MEASURE_STEPS = 30

# Compile modes to evaluate
COMPILE_MODES = ["default", "reduce-overhead", "max-autotune"]


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_call(fn, label: str = "call") -> float:
    """Run fn once, return wall-clock time in ms (with CUDA sync)."""
    sync()
    t0 = time.perf_counter()
    fn()
    sync()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return elapsed_ms


def measure_first_and_steady(
    model: nn.Module,
    input_ids: torch.Tensor,
    label: str,
    steady_steps: int = MEASURE_STEPS,
) -> dict:
    """
    Measure:
      - first_call_ms  : time for the very first forward pass (includes compile if applicable)
      - steady_ms      : mean time for subsequent forward passes

    Returns a dict with keys: label, first_call_ms, steady_ms, speedup_vs_first.
    """
    with torch.no_grad():
        first_ms = timed_call(lambda: model(input_ids), label)

        times = []
        for _ in range(steady_steps):
            t = timed_call(lambda: model(input_ids), label)
            times.append(t)

    steady_ms = sum(times) / len(times)
    return {
        "label": label,
        "first_call_ms": first_ms,
        "steady_ms": steady_ms,
        "speedup_vs_first": first_ms / steady_ms if steady_ms > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Baseline (no compile)
# ---------------------------------------------------------------------------

def run_baseline(model: nn.Module, input_ids: torch.Tensor) -> dict:
    """Measure baseline forward pass with no compilation."""
    print("  Running baseline (no compile)...")
    result = measure_first_and_steady(model, input_ids, label="baseline (no compile)")
    return result


# ---------------------------------------------------------------------------
# torch.compile modes
# ---------------------------------------------------------------------------

def run_compile_mode(
    model: nn.Module,
    input_ids: torch.Tensor,
    mode: str,
    backend: str = "inductor",
    fullgraph: bool = False,
    dynamic: bool = False,
) -> dict:
    """
    Compile model with the given mode and benchmark it.

    Returns the same dict structure as run_baseline.
    """
    label_parts = [f"compile(mode='{mode}'"]
    if backend != "inductor":
        label_parts.append(f"backend='{backend}'")
    if fullgraph:
        label_parts.append("fullgraph=True")
    if dynamic:
        label_parts.append("dynamic=True")
    label = ", ".join(label_parts) + ")"

    print(f"  Compiling with {label}...")
    compiled = torch.compile(model, mode=mode, backend=backend,
                              fullgraph=fullgraph, dynamic=dynamic)

    # NOTE: torch.compile is lazy — compilation triggers on first call.
    # We intentionally do NOT warm up before the first timed call so we can
    # capture the true first-call (compilation) latency.
    result = measure_first_and_steady(compiled, input_ids, label=label)
    return result


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def print_compile_table(results: list[dict], baseline_steady_ms: float) -> None:
    """Print a formatted comparison table."""
    col_w = 55
    print()
    print("=" * 110)
    print(f"{'Configuration':<{col_w}} {'First call (ms)':>16} {'Steady (ms)':>12} {'vs First':>10} {'Speedup vs base':>16}")
    print("-" * 110)
    for r in results:
        speedup = baseline_steady_ms / r["steady_ms"] if r["steady_ms"] > 0 else float("nan")
        print(
            f"  {r['label']:<{col_w - 2}} "
            f"{r['first_call_ms']:>16.1f} "
            f"{r['steady_ms']:>12.2f} "
            f"{r['speedup_vs_first']:>9.1f}x "
            f"{speedup:>15.2f}x"
        )
    print("=" * 110)
    print()


# ---------------------------------------------------------------------------
# fullgraph=True demo
# ---------------------------------------------------------------------------

def demo_fullgraph(model: nn.Module, input_ids: torch.Tensor) -> None:
    """
    Demonstrate fullgraph=True.

    When fullgraph=True, torch.compile raises an error if it cannot trace the
    entire model as a single computation graph (i.e., if there are graph breaks
    due to data-dependent control flow or unsupported operations).

    Our SimpleTransformer has no graph breaks, so this should succeed.
    """
    print()
    print("--- fullgraph=True demo ---")
    print("  Compiling with fullgraph=True (raises error if graph breaks exist)...")
    try:
        compiled = torch.compile(model, mode="default", fullgraph=True)
        with torch.no_grad():
            _ = compiled(input_ids)   # trigger compile
        print("  fullgraph=True: succeeded — no graph breaks detected.")
    except Exception as e:
        print(f"  fullgraph=True: FAILED — graph break detected: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# dynamic=True demo
# ---------------------------------------------------------------------------

def demo_dynamic(model: nn.Module, device: str) -> None:
    """
    Demonstrate dynamic=True.

    With dynamic=True, torch.compile generates code that handles variable
    input shapes without recompilation. Useful when sequence lengths vary.
    With dynamic=False (default), a new kernel is compiled for each new shape.
    """
    print()
    print("--- dynamic=True demo ---")
    print("  Compiling with dynamic=True (handles variable shapes without recompile)...")
    compiled_dynamic = torch.compile(model, mode="default", dynamic=True)
    compiled_static = torch.compile(model, mode="default", dynamic=False)

    vocab_size = model.config.vocab_size
    shapes = [(BATCH_SIZE, 64), (BATCH_SIZE, 128), (BATCH_SIZE, 256)]

    print()
    print(f"  {'Shape':<20} {'Dynamic (ms)':>14} {'Static (ms)':>14}")
    print(f"  {'-'*20} {'-'*14} {'-'*14}")

    for bs, sl in shapes:
        ids = generate_random_batch(bs, sl, vocab_size, device)
        with torch.no_grad():
            t_dyn = timed_call(lambda: compiled_dynamic(ids), "dynamic")
            t_sta = timed_call(lambda: compiled_static(ids), "static")
        print(f"  bs={bs}, seq={sl:<12}  {t_dyn:>14.2f}   {t_sta:>14.2f}")

    print()
    print("  Note: dynamic=True avoids recompilation when shapes change.")
    print("  First call for each new static shape triggers recompilation (guard failure).")


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("Chapter 5 — torch.compile Modes Demo")
    print("=" * 70)
    print()

    # Check PyTorch version — torch.compile requires >= 2.0
    torch_version = tuple(int(x) for x in torch.__version__.split(".")[:2])
    if torch_version < (2, 0):
        print(f"WARNING: torch.compile requires PyTorch >= 2.0. "
              f"You have {torch.__version__}. Skipping compile demos.")
        compile_available = False
    else:
        compile_available = True
        print(f"PyTorch version: {torch.__version__} — torch.compile is available.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print_gpu_info()
    else:
        print("CUDA not available — running on CPU (compile still works, slower).")
    print()

    # Build a model large enough to show meaningful differences
    print("Building model...")
    model = create_model(
        num_layers=8,
        hidden_dim=1024,
        num_heads=8,
        num_kv_heads=4,   # GQA
        device=device,
        dtype=torch.float32,
    )
    model.eval()
    n_params = model.param_count_str()
    print(f"  Model: {n_params} params, device={device}")
    print()

    input_ids = generate_random_batch(BATCH_SIZE, SEQ_LEN, model.config.vocab_size, device)

    all_results: list[dict] = []

    # -----------------------------------------------------------------------
    # 1. Baseline
    # -----------------------------------------------------------------------
    print("Step 1: Baseline (no compile)")
    baseline = run_baseline(model, input_ids)
    all_results.append(baseline)
    print(f"    Steady-state latency: {baseline['steady_ms']:.2f} ms")
    print()

    if not compile_available:
        print("Skipping compile experiments (PyTorch < 2.0).")
        return

    # -----------------------------------------------------------------------
    # 2. Compile modes
    # -----------------------------------------------------------------------
    for mode in COMPILE_MODES:
        print(f"Step 2: torch.compile(mode='{mode}')")
        if mode == "max-autotune":
            print("    NOTE: max-autotune runs exhaustive kernel search on first call.")
            print("    Expect very long first-call time (minutes on first run).")
        result = run_compile_mode(model, input_ids, mode=mode)
        all_results.append(result)
        print(f"    First call: {result['first_call_ms']:.1f} ms  "
              f"(includes compilation)")
        print(f"    Steady:     {result['steady_ms']:.2f} ms  "
              f"({result['speedup_vs_first']:.1f}x faster than first call)")
        print()

    # -----------------------------------------------------------------------
    # 3. Non-default backends
    # -----------------------------------------------------------------------
    print("Step 3: Backend comparison (mode='default')")
    for backend in ["eager", "aot_eager", "inductor"]:
        print(f"  Backend: {backend}")
        try:
            r = run_compile_mode(model, input_ids, mode="default", backend=backend)
            all_results.append(r)
            print(f"    Steady: {r['steady_ms']:.2f} ms")
        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {e}")
    print()

    # -----------------------------------------------------------------------
    # 4. fullgraph and dynamic options
    # -----------------------------------------------------------------------
    demo_fullgraph(model, input_ids)
    demo_dynamic(model, device)

    # -----------------------------------------------------------------------
    # 5. Summary table
    # -----------------------------------------------------------------------
    print()
    print("SUMMARY TABLE")
    print_compile_table(all_results, baseline_steady_ms=baseline["steady_ms"])

    print("Key takeaways:")
    print("  1. First call after torch.compile is slow (compilation overhead).")
    print("  2. Subsequent calls benefit from compiled kernels — warm-up is essential.")
    print("  3. 'reduce-overhead' helps most for small batches (Python dispatch cost).")
    print("  4. 'max-autotune' has the highest first-call cost but best steady-state.")
    print("  5. fullgraph=True guarantees no Python fallback (good for production).")
    print("  6. dynamic=True avoids recompilation for varying sequence lengths.")
    print()


if __name__ == "__main__":
    main()
