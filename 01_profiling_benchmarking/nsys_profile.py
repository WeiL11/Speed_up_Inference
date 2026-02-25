#!/usr/bin/env python3
"""
Chapter 1 — Nsight Systems Profiling Script
=============================================

Wraps model execution with NVTX annotations so that Nsight Systems (nsys)
can produce detailed GPU kernel traces.

Features:
  - NVTX range annotations for: forward pass, backward pass, optimizer step
  - Per-layer annotations: self-attention, MLP, LayerNorm
  - Three profiling modes:
      * forward   — inference only
      * fwd_bwd   — forward + backward
      * train     — forward + backward + AdamW optimizer step
  - Generates nsys-compatible profiling session
  - Also runs a Python-level timing pass so you can compare against nsys totals

The script is designed to help answer these key profiling questions:

  Q1. Does the nsys total forward-pass time match the Python-level measurement?
  Q2. Which CUDA kernel takes the most cumulative GPU time? (forward vs fwd+bwd)
  Q3. What non-matmul kernels have non-trivial runtime in the forward pass?
  Q4. How does the matmul fraction change: inference vs full training step?
  Q5. Softmax runtime vs matmul runtime in self-attention — how do they compare?

Usage (two steps):

  Step 1: Run this script to generate the profiling data.
    nsys profile -t cuda,nvtx -o profile_forward --force-overwrite \\
        python nsys_profile.py --mode forward

    nsys profile -t cuda,nvtx -o profile_train --force-overwrite \\
        python nsys_profile.py --mode train

  Step 2: Open the .nsys-rep file in Nsight Systems GUI, or use:
    nsys stats profile_forward.nsys-rep

  Tip: In the Nsight Systems GUI, use "Stats System View" →
       "CUDA GPU Kernel Summary", and filter by NVTX ranges to find
       which layers produce which kernels.
"""

import argparse
import sys
import timeit
from pathlib import Path

import torch
import torch.nn as nn
import torch.cuda.nvtx as nvtx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.model_loader import (
    SimpleTransformer,
    TransformerConfig,
    TransformerBlock,
    SelfAttention,
    MLP,
    RMSNorm,
    create_model,
    generate_random_batch,
)
from utils.benchmarking import print_gpu_info


# ---------------------------------------------------------------------------
# NVTX-annotated model wrappers
# ---------------------------------------------------------------------------

class NVTXSelfAttention(SelfAttention):
    """SelfAttention with NVTX range annotations."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nvtx.range_push("SelfAttention")
        out = super().forward(x)
        nvtx.range_pop()
        return out


class NVTXMLP(MLP):
    """MLP with NVTX range annotations."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nvtx.range_push("MLP")
        out = super().forward(x)
        nvtx.range_pop()
        return out


class NVTXRMSNorm(RMSNorm):
    """RMSNorm with NVTX range annotations."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nvtx.range_push("RMSNorm")
        out = super().forward(x)
        nvtx.range_pop()
        return out


class NVTXTransformerBlock(TransformerBlock):
    """TransformerBlock with NVTX annotations on sub-components."""

    def __init__(self, config: TransformerConfig):
        # Call nn.Module.__init__ directly to avoid TransformerBlock
        # creating un-annotated submodules
        nn.Module.__init__(self)
        self.attn_norm = NVTXRMSNorm(config.hidden_dim)
        self.attn = NVTXSelfAttention(config)
        self.mlp_norm = NVTXRMSNorm(config.hidden_dim)
        self.mlp = NVTXMLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nvtx.range_push("TransformerBlock")
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        nvtx.range_pop()
        return x


class NVTXTransformer(SimpleTransformer):
    """SimpleTransformer with NVTX-annotated layers for nsys profiling."""

    def __init__(self, config: TransformerConfig):
        # Call nn.Module.__init__ to skip SimpleTransformer's layer creation
        nn.Module.__init__(self)
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.layers = nn.ModuleList(
            [NVTXTransformerBlock(config) for _ in range(config.num_layers)]
        )
        self.norm = NVTXRMSNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.vocab_size, config.hidden_dim, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        nvtx.range_push("Embedding")
        x = self.tok_emb(input_ids)
        nvtx.range_pop()

        for i, layer in enumerate(self.layers):
            nvtx.range_push(f"Layer_{i}")
            x = layer(x)
            nvtx.range_pop()

        nvtx.range_push("FinalNorm")
        x = self.norm(x)
        nvtx.range_pop()

        nvtx.range_push("LMHead")
        x = self.lm_head(x)
        nvtx.range_pop()

        return x


# ---------------------------------------------------------------------------
# Profiling runner
# ---------------------------------------------------------------------------

def run_profiling(args: argparse.Namespace) -> None:
    """Run the profiling session."""

    device = "cuda"
    dtype = torch.float32

    print("=" * 60)
    print("NSIGHT SYSTEMS PROFILING SESSION")
    print("=" * 60)
    print(f"  Mode:       {args.mode}")
    print(f"  Layers:     {args.num_layers}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Heads:      {args.num_heads}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Seq len:    {args.seq_len}")
    print(f"  Steps:      {args.warmup} warmup + {args.steps} profiled")
    print()
    print_gpu_info()
    print()

    # ---- Build NVTX-annotated model ----
    config = TransformerConfig(
        vocab_size=args.vocab_size,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_len=args.seq_len,
        dtype=dtype,
    )
    model = NVTXTransformer(config).to(dtype).to(device)
    print(f"Model parameters: {model.param_count_str()}")

    # ---- Data ----
    input_ids = generate_random_batch(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        device=device,
    )
    targets = generate_random_batch(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        device=device,
    )
    loss_fn = nn.CrossEntropyLoss()

    # ---- Optimizer (for train mode) ----
    optimizer = None
    if args.mode == "train":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # ---- Step functions with NVTX markers ----
    def forward_step():
        nvtx.range_push("ForwardPass")
        with torch.no_grad():
            logits = model(input_ids)
        nvtx.range_pop()
        return logits

    def fwd_bwd_step():
        model.zero_grad(set_to_none=True)

        nvtx.range_push("ForwardPass")
        logits = model(input_ids)
        nvtx.range_pop()

        nvtx.range_push("Loss")
        loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        nvtx.range_pop()

        nvtx.range_push("BackwardPass")
        loss.backward()
        nvtx.range_pop()

        return loss

    def train_step():
        model.zero_grad(set_to_none=True)

        nvtx.range_push("ForwardPass")
        logits = model(input_ids)
        nvtx.range_pop()

        nvtx.range_push("Loss")
        loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        nvtx.range_pop()

        nvtx.range_push("BackwardPass")
        loss.backward()
        nvtx.range_pop()

        nvtx.range_push("OptimizerStep")
        optimizer.step()
        nvtx.range_pop()

        return loss

    step_fn = {
        "forward": forward_step,
        "fwd_bwd": fwd_bwd_step,
        "train": train_step,
    }[args.mode]

    # ---- Warm-up (outside profiler capture) ----
    print(f"Running {args.warmup} warm-up steps...")
    for _ in range(args.warmup):
        step_fn()
        torch.cuda.synchronize()
    print("  Warm-up complete.\n")

    # ---- Python-level timing (for comparison with nsys) ----
    print(f"Python-level timing ({args.steps} steps)...")
    times_ms = []
    for _ in range(args.steps):
        torch.cuda.synchronize()
        start = timeit.default_timer()

        nvtx.range_push(f"Step_{args.mode}")
        step_fn()
        nvtx.range_pop()

        torch.cuda.synchronize()
        elapsed = (timeit.default_timer() - start) * 1000.0
        times_ms.append(elapsed)

    # ---- Report ----
    import numpy as np
    mean = np.mean(times_ms)
    std = np.std(times_ms)
    print(f"\n{'=' * 60}")
    print(f"PYTHON-LEVEL TIMING RESULTS")
    print(f"{'=' * 60}")
    print(f"  Mode:       {args.mode}")
    print(f"  Mean:       {mean:.2f} ms")
    print(f"  Std:        {std:.2f} ms")
    print(f"  Min:        {min(times_ms):.2f} ms")
    print(f"  Max:        {max(times_ms):.2f} ms")
    print(f"  Throughput: {1000.0 / mean:.1f} steps/s")
    print()
    print("Now open the .nsys-rep file in Nsight Systems to analyze GPU kernels.")
    print()
    print("Key things to look for:")
    print("  1. 'CUDA GPU Kernel Summary' → sort by 'Total Time'")
    print("  2. Filter by NVTX range 'ForwardPass' vs 'BackwardPass'")
    print("  3. Compare matmul kernel time vs softmax kernel time")
    print("  4. Check 'NVTX Push/Pop Range Statistics' for per-layer breakdown")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NVTX-annotated transformer profiling for Nsight Systems"
    )

    # Model
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--vocab_size", type=int, default=32000)

    # Data
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=512)

    # Profiling
    parser.add_argument("--warmup", type=int, default=3,
                        help="Warm-up steps before profiling (default: 3)")
    parser.add_argument("--steps", type=int, default=5,
                        help="Steps to profile (default: 5, keep small for nsys)")
    parser.add_argument("--mode", type=str, default="forward",
                        choices=["forward", "fwd_bwd", "train"],
                        help="Profiling mode (default: forward)")

    return parser.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is required for Nsight Systems profiling.")
        sys.exit(1)

    run_profiling(args)


if __name__ == "__main__":
    main()
