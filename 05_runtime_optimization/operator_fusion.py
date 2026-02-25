"""
Chapter 5 — Runtime Optimization
File: operator_fusion.py

Manual operator fusion: combine multiple elementwise operations into a
single kernel to reduce memory traffic and kernel launch overhead.

Fused operations demonstrated:
  1. RMSNorm + Linear   → single kernel (norm + matmul in one pass)
  2. SiLU + Mul (gated MLP) → fuse activation with gate multiplication

Why fusion matters:
  - Each separate op reads/writes to HBM (slow global memory)
  - Fusing N ops into 1 reduces HBM round-trips from N to 1
  - Especially impactful for elementwise ops that are memory-bound

Contents:
  - FusedRMSNormLinear  : RMSNorm followed by Linear in a single module
  - FusedSiLUMul        : SiLU(gate) * up in one operation
  - FusedMLP            : complete fused gated MLP
  - benchmark           : measure speedup from fusion
"""

import sys
import os
import time
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.model_loader import TransformerConfig, RMSNorm, MLP
from utils.benchmarking import benchmark_fn


# ---------------------------------------------------------------------------
# Fused RMSNorm + Linear
# ---------------------------------------------------------------------------

class FusedRMSNormLinear(nn.Module):
    """
    Fused RMSNorm → Linear.

    Instead of:
      1. x_norm = rmsnorm(x)         # read x from HBM, write x_norm to HBM
      2. out = linear(x_norm)         # read x_norm from HBM, write out to HBM
    We do:
      1. out = linear(rmsnorm(x))     # single logical operation

    With torch.compile, this can be auto-fused. Here we show the manual pattern
    and measure the difference.
    """

    def __init__(self, dim: int, out_features: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.norm_weight = nn.Parameter(torch.ones(dim))
        self.linear = nn.Linear(dim, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute norm inline (avoids materializing normalized intermediate)
        norm = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        x_norm = (x.float() * norm).to(x.dtype) * self.norm_weight
        return self.linear(x_norm)


class UnfusedRMSNormLinear(nn.Module):
    """Separate RMSNorm then Linear (baseline for comparison)."""

    def __init__(self, dim: int, out_features: int, eps: float = 1e-6):
        super().__init__()
        self.norm = RMSNorm(dim, eps)
        self.linear = nn.Linear(dim, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(x))


# ---------------------------------------------------------------------------
# Fused SiLU + Mul (Gated MLP activation)
# ---------------------------------------------------------------------------

class FusedSiLUMul(nn.Module):
    """
    Fused SiLU(gate) * up for gated MLP.

    Standard (unfused):
      gate_out = silu(gate_proj(x))    # write to HBM
      up_out = up_proj(x)              # write to HBM
      out = gate_out * up_out          # read both from HBM, write result

    Fused: compute silu and multiply in a single pass over the data.
    """

    def __init__(self):
        super().__init__()

    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        # F.silu(gate) * up — PyTorch may fuse this with torch.compile
        return F.silu(gate) * up


@torch.jit.script
def fused_silu_mul_jit(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """JIT-compiled fused SiLU * mul for guaranteed fusion."""
    return F.silu(gate) * up


# ---------------------------------------------------------------------------
# Fused MLP
# ---------------------------------------------------------------------------

class FusedMLP(nn.Module):
    """
    Gated MLP with fused activation.

    Architecture (Gemma/LLaMA style):
      out = down_proj(silu(gate_proj(x)) * up_proj(x))

    The SiLU * mul part is fused into a single operation.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.up_proj   = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.down_proj = nn.Linear(config.intermediate_dim, config.hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(fused_silu_mul_jit(self.gate_proj(x), self.up_proj(x)))


class UnfusedMLP(nn.Module):
    """Standard unfused MLP for comparison."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.up_proj   = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.down_proj = nn.Linear(config.intermediate_dim, config.hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark_rmsnorm_linear(
    dim: int = 1024,
    out_features: int = 2048,
    batch: int = 8,
    seq_len: int = 512,
    device: str = None,
    warmup: int = 10,
    steps: int = 50,
):
    """Benchmark fused vs unfused RMSNorm + Linear."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    has_cuda = "cuda" in device

    fused = FusedRMSNormLinear(dim, out_features).to(device)
    unfused = UnfusedRMSNormLinear(dim, out_features).to(device)

    # Copy weights for fair comparison
    with torch.no_grad():
        fused.norm_weight.copy_(unfused.norm.weight)
        fused.linear.weight.copy_(unfused.linear.weight)

    x = torch.randn(batch, seq_len, dim, device=device)

    # Verify correctness
    with torch.no_grad():
        out_fused = fused(x)
        out_unfused = unfused(x)
        diff = (out_fused - out_unfused).abs().max().item()
        print(f"  RMSNorm+Linear: max diff = {diff:.2e}  ({'PASS' if diff < 1e-4 else 'FAIL'})")

    r_unfused = benchmark_fn(lambda: unfused(x), warmup_steps=warmup,
                             measure_steps=steps, name="unfused", sync_cuda=has_cuda)
    r_fused = benchmark_fn(lambda: fused(x), warmup_steps=warmup,
                           measure_steps=steps, name="fused", sync_cuda=has_cuda)

    speedup = r_unfused.mean_ms / r_fused.mean_ms if r_fused.mean_ms > 0 else 0

    print(f"\n  RMSNorm + Linear  (dim={dim}, out={out_features}, B={batch}, T={seq_len})")
    print(f"  Unfused: {r_unfused.mean_ms:.3f} ms")
    print(f"  Fused:   {r_fused.mean_ms:.3f} ms")
    print(f"  Speedup: {speedup:.2f}x")
    return speedup


def benchmark_gated_mlp(
    hidden_dim: int = 1024,
    batch: int = 8,
    seq_len: int = 512,
    device: str = None,
    warmup: int = 10,
    steps: int = 50,
):
    """Benchmark fused vs unfused gated MLP."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    has_cuda = "cuda" in device

    config = TransformerConfig(hidden_dim=hidden_dim)
    fused = FusedMLP(config).to(device)
    unfused = UnfusedMLP(config).to(device)

    # Copy weights
    with torch.no_grad():
        unfused.gate_proj.weight.copy_(fused.gate_proj.weight)
        unfused.up_proj.weight.copy_(fused.up_proj.weight)
        unfused.down_proj.weight.copy_(fused.down_proj.weight)

    x = torch.randn(batch, seq_len, hidden_dim, device=device)

    # Verify
    with torch.no_grad():
        out_fused = fused(x)
        out_unfused = unfused(x)
        diff = (out_fused - out_unfused).abs().max().item()
        print(f"  Gated MLP: max diff = {diff:.2e}  ({'PASS' if diff < 1e-4 else 'FAIL'})")

    r_unfused = benchmark_fn(lambda: unfused(x), warmup_steps=warmup,
                             measure_steps=steps, name="unfused", sync_cuda=has_cuda)
    r_fused = benchmark_fn(lambda: fused(x), warmup_steps=warmup,
                           measure_steps=steps, name="fused", sync_cuda=has_cuda)

    intermediate_dim = config.intermediate_dim
    speedup = r_unfused.mean_ms / r_fused.mean_ms if r_fused.mean_ms > 0 else 0

    print(f"\n  Gated MLP  (hidden={hidden_dim}, intermediate={intermediate_dim})")
    print(f"  Unfused: {r_unfused.mean_ms:.3f} ms")
    print(f"  Fused:   {r_fused.mean_ms:.3f} ms")
    print(f"  Speedup: {speedup:.2f}x")
    return speedup


def benchmark_torch_compile_fusion(
    hidden_dim: int = 1024,
    batch: int = 8,
    seq_len: int = 512,
    device: str = None,
    warmup: int = 10,
    steps: int = 50,
):
    """Show how torch.compile automatically fuses operations."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    has_cuda = "cuda" in device

    config = TransformerConfig(hidden_dim=hidden_dim)
    unfused = UnfusedMLP(config).to(device)
    x = torch.randn(batch, seq_len, hidden_dim, device=device)

    r_eager = benchmark_fn(lambda: unfused(x), warmup_steps=warmup,
                           measure_steps=steps, name="eager", sync_cuda=has_cuda)

    # torch.compile with inductor
    try:
        compiled = torch.compile(unfused, mode="reduce-overhead")
        # Extra warmup for compilation
        for _ in range(3):
            _ = compiled(x)
            if has_cuda:
                torch.cuda.synchronize()

        r_compiled = benchmark_fn(lambda: compiled(x), warmup_steps=warmup,
                                  measure_steps=steps, name="compiled", sync_cuda=has_cuda)

        speedup = r_eager.mean_ms / r_compiled.mean_ms if r_compiled.mean_ms > 0 else 0
        print(f"\n  torch.compile auto-fusion (MLP)")
        print(f"  Eager:    {r_eager.mean_ms:.3f} ms")
        print(f"  Compiled: {r_compiled.mean_ms:.3f} ms")
        print(f"  Speedup:  {speedup:.2f}x")
        print(f"  torch.compile's inductor backend auto-fuses elementwise ops,")
        print(f"  often matching or exceeding manual fusion.")
    except Exception as e:
        print(f"\n  torch.compile not available: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("  Chapter 5: Operator Fusion")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Running on: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")

    print("\n  [1] RMSNorm + Linear fusion")
    benchmark_rmsnorm_linear(device=device)

    print("\n  [2] Gated MLP (SiLU + mul) fusion")
    benchmark_gated_mlp(device=device)

    print("\n  [3] torch.compile automatic fusion")
    benchmark_torch_compile_fusion(device=device)

    print("\n  Key takeaways:")
    print("  - Fusing elementwise ops reduces HBM round-trips")
    print("  - JIT script provides guaranteed fusion for simple patterns")
    print("  - torch.compile (inductor) auto-fuses and often matches manual fusion")
    print("  - Biggest gains for memory-bound ops (small compute, large data movement)")
