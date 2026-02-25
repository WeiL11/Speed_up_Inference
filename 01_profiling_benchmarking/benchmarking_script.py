#!/usr/bin/env python3
"""
Chapter 1 — End-to-End Benchmarking Script
===========================================

Profile any transformer model — works with our custom SimpleTransformer or
any HuggingFace model (e.g. google/gemma-3-1b) out of the box.

Hyperparameters
---------------
Model architecture:
  --num_layers        Number of transformer layers            (default: 6)
  --hidden_dim        Hidden / embedding dimension            (default: 1024)
  --num_heads         Number of attention (query) heads       (default: 8)
  --num_kv_heads      KV heads for GQA; == num_heads → MHA   (default: same as --num_heads)
  --intermediate_dim  MLP intermediate width (0 = auto)       (default: 0)
  --vocab_size        Vocabulary size                         (default: 32000)
  --max_seq_len       Maximum sequence length                  (default: 2048)
  --no_tie_weights    Disable weight tying (embed ↔ LM head)

Data / batching:
  --batch_size        Batch size                              (default: 8)
  --seq_len           Input sequence length                   (default: 512)

Precision:
  --dtype             float32 / float16 / bfloat16            (default: float32)

Timing:
  --warmup            Warm-up steps (untimed). Fills CUDA caches, triggers
                      lazy JIT compilation, eliminates first-run overhead.
                                                               (default: 5)
  --steps             Measured steps                           (default: 20)
  --mode              forward / backward / train
                      forward  = forward pass only (torch.no_grad)
                      backward = forward + backward
                      train    = forward + backward + optimizer step
                                                               (default: forward)

Optimizer (used only in train mode):
  --optimizer         adamw / sgd                             (default: adamw)
  --lr                Learning rate                           (default: 1e-4)

Device:
  --device            cuda / cpu                              (default: cuda)

Model source:
  --model_name        "custom" → use SimpleTransformer with above params.
                      HuggingFace ID → load pretrained model (e.g. google/gemma-3-1b).
                                                               (default: custom)

Examples
--------
  # Custom 6-layer model, forward only
  python benchmarking_script.py --mode forward --num_layers 6 --hidden_dim 1024

  # GQA model (4 query heads, 1 KV head — like Gemma-3 1B)
  python benchmarking_script.py --num_heads 4 --num_kv_heads 1 --hidden_dim 1152 --num_layers 26

  # Full training step with bfloat16
  python benchmarking_script.py --mode train --dtype bfloat16 --warmup 10 --steps 50

  # Load pretrained Gemma-3 1B from HuggingFace
  python benchmarking_script.py --model_name google/gemma-3-1b --dtype bfloat16 --mode forward
"""

import argparse
import sys
import timeit
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.model_loader import (
    create_model,
    load_hf_model,
    generate_random_batch,
    SimpleTransformer,
)
from utils.benchmarking import BenchmarkResult, print_gpu_info


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

DTYPE_MAP = {
    "float32":  torch.float32,
    "float16":  torch.float16,
    "bfloat16": torch.bfloat16,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark transformer forward/backward/train passes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Model architecture ---
    g = p.add_argument_group("Model architecture")
    g.add_argument("--num_layers",       type=int,   default=6)
    g.add_argument("--hidden_dim",       type=int,   default=1024)
    g.add_argument("--num_heads",        type=int,   default=8)
    g.add_argument("--num_kv_heads",     type=int,   default=None,
                   help="KV heads (GQA); defaults to --num_heads (MHA)")
    g.add_argument("--intermediate_dim", type=int,   default=0,
                   help="MLP intermediate dim; 0 = auto (8/3 × hidden_dim)")
    g.add_argument("--vocab_size",       type=int,   default=32_000)
    g.add_argument("--max_seq_len",      type=int,   default=2048)
    g.add_argument("--no_tie_weights",   action="store_true",
                   help="Disable token-embedding ↔ LM-head weight tying")

    # --- Data ---
    g = p.add_argument_group("Data / batching")
    g.add_argument("--batch_size", type=int, default=8)
    g.add_argument("--seq_len",    type=int, default=512)

    # --- Precision ---
    g = p.add_argument_group("Precision")
    g.add_argument("--dtype", type=str, default="float32",
                   choices=["float32", "float16", "bfloat16"])

    # --- Timing ---
    g = p.add_argument_group("Timing")
    g.add_argument("--warmup", type=int, default=5,
                   help="Warm-up steps before timing (fills CUDA caches, triggers JIT)")
    g.add_argument("--steps",  type=int, default=20,
                   help="Number of measured steps")
    g.add_argument("--mode",   type=str, default="forward",
                   choices=["forward", "backward", "train"],
                   help="forward / backward (fwd+bwd) / train (fwd+bwd+optimizer)")

    # --- Optimizer ---
    g = p.add_argument_group("Optimizer (only used in --mode train)")
    g.add_argument("--optimizer", type=str, default="adamw",
                   choices=["adamw", "sgd"])
    g.add_argument("--lr", type=float, default=1e-4)

    # --- Device / model source ---
    g = p.add_argument_group("Device & model source")
    g.add_argument("--device",     type=str, default="cuda",
                   choices=["cuda", "cpu"])
    g.add_argument("--model_name", type=str, default="custom",
                   help='"custom" or HuggingFace model ID (e.g. google/gemma-3-1b)')

    return p.parse_args()


# ---------------------------------------------------------------------------
# Step functions
# ---------------------------------------------------------------------------

def make_step_fn(model, input_ids, targets, loss_fn, optimizer, mode: str):
    """Return a zero-argument callable for the requested mode."""

    if mode == "forward":
        def step():
            with torch.no_grad():
                model(input_ids)

    elif mode == "backward":
        def step():
            model.zero_grad(set_to_none=True)
            logits = model(input_ids)
            loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()

    elif mode == "train":
        def step():
            model.zero_grad(set_to_none=True)
            logits = model(input_ids)
            loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()

    return step


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(args: argparse.Namespace) -> BenchmarkResult:
    device = args.device
    dtype  = DTYPE_MAP[args.dtype]

    # ── Print configuration ──────────────────────────────────────────────────
    print("=" * 64)
    print("BENCHMARK CONFIGURATION")
    print("=" * 64)
    print(f"  Model source:   {args.model_name}")
    if args.model_name == "custom":
        nkv = args.num_kv_heads or args.num_heads
        attn_type = "MHA" if nkv == args.num_heads else f"GQA ({nkv} KV heads)"
        print(f"  Architecture:   {args.num_layers} layers, "
              f"dim={args.hidden_dim}, heads={args.num_heads} [{attn_type}]")
        print(f"  Vocab size:     {args.vocab_size:,}")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  Seq length:     {args.seq_len}")
    print(f"  Mode:           {args.mode}")
    print(f"  Warmup steps:   {args.warmup}  (untimed — fills CUDA caches + JIT)")
    print(f"  Measured steps: {args.steps}")
    print(f"  dtype:          {args.dtype}")
    print(f"  Device:         {device}")
    if args.mode == "train":
        print(f"  Optimizer:      {args.optimizer}, lr={args.lr}")
    print()

    if device == "cuda" and torch.cuda.is_available():
        print_gpu_info()
        print()

    # ── Initialize model ─────────────────────────────────────────────────────
    hf_tokenizer = None
    if args.model_name == "custom":
        model = create_model(
            num_layers=args.num_layers,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            intermediate_dim=args.intermediate_dim,
            vocab_size=args.vocab_size,
            max_seq_len=args.max_seq_len,
            tie_weights=not args.no_tie_weights,
            dtype=dtype,
            device=device,
        )
        vocab_size = args.vocab_size
        print(f"Custom model:   {model.param_count_str()} params "
              f"({model.param_count():,})")
        print(f"Model size:     {model.model_size_mb():.1f} MB")
    else:
        model, hf_tokenizer = load_hf_model(args.model_name, dtype=dtype, device=device)
        vocab_size = model.config.vocab_size
        n_params   = sum(p.numel() for p in model.parameters())
        print(f"HuggingFace model: {n_params/1e9:.2f}B params")

    model.eval()

    if device == "cuda":
        model_mem = torch.cuda.memory_allocated() / (1024 ** 2)
        print(f"GPU memory after model load: {model_mem:.1f} MB")
    print()

    # ── Generate data ────────────────────────────────────────────────────────
    input_ids = generate_random_batch(args.batch_size, args.seq_len, vocab_size, device)
    targets   = generate_random_batch(args.batch_size, args.seq_len, vocab_size, device)
    loss_fn   = nn.CrossEntropyLoss()

    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer = None
    if args.mode == "train":
        if args.optimizer == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    step_fn = make_step_fn(model, input_ids, targets, loss_fn, optimizer, args.mode)

    # ── Warm-up ──────────────────────────────────────────────────────────────
    # Critical: warm-up steps fill CUDA kernel caches, trigger torch lazy init,
    # and let any JIT compilation settle — without this, the first measured
    # step would be unrepresentatively slow.
    print(f"Running {args.warmup} warm-up steps (untimed)...")
    for _ in range(args.warmup):
        step_fn()
        if device == "cuda":
            torch.cuda.synchronize()
    print("  Warm-up complete.\n")

    # ── Measured runs ────────────────────────────────────────────────────────
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    print(f"Timing {args.steps} steps with timeit.default_timer() ...")
    times_ms = []
    for _ in range(args.steps):
        # Synchronize before start so the timer captures only GPU work
        if device == "cuda":
            torch.cuda.synchronize()

        t0 = timeit.default_timer()
        step_fn()

        # Synchronize after so the timer waits for GPU to finish
        if device == "cuda":
            torch.cuda.synchronize()

        times_ms.append((timeit.default_timer() - t0) * 1000.0)

    # ── Collect and report ───────────────────────────────────────────────────
    result = BenchmarkResult(
        name=f"[{args.mode}] model={args.model_name}, "
             f"B={args.batch_size}, T={args.seq_len}, dtype={args.dtype}",
        times_ms=times_ms,
    )
    if device == "cuda":
        result.peak_memory_mb     = torch.cuda.max_memory_allocated() / (1024 ** 2)
        result.allocated_memory_mb = torch.cuda.memory_allocated() / (1024 ** 2)

    print()
    print(result.summary())
    print()

    tokens_per_step  = args.batch_size * args.seq_len
    total_tokens     = tokens_per_step * args.steps
    total_s          = sum(times_ms) / 1000.0
    print(f"  Tokens/step:    {tokens_per_step:,}")
    print(f"  Total tokens:   {total_tokens:,}")
    print(f"  Total time:     {total_s:.3f} s")
    print(f"  Tokens/sec:     {total_tokens / total_s:,.0f}")

    return result


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available — falling back to CPU.")
        args.device = "cpu"

    run_benchmark(args)


if __name__ == "__main__":
    main()
