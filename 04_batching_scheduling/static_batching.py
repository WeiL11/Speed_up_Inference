"""
Chapter 4 — Batching & Scheduling
FILE 1: static_batching.py

Naive static batching: pad ALL sequences in a batch to the same maximum
length. Simple to implement but wasteful — a single long sequence forces
every other sequence in the batch to be padded to that length, throwing
away compute on padding tokens.
"""

import sys
import os
import random
import time
from typing import Optional

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.model_loader import create_model


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class StaticBatcher:
    """
    Collects requests and pads them to the max sequence length in the batch.

    Simple but wasteful: if one sequence is 10 tokens and another is 500,
    ALL sequences are padded to 500, wasting 490 tokens of compute for the
    short sequence.

    Parameters
    ----------
    max_batch_size : int
        Maximum number of sequences per batch.
    pad_token_id : int
        Token ID used for padding (default 0).
    """

    def __init__(self, max_batch_size: int, pad_token_id: int = 0):
        self.max_batch_size = max_batch_size
        self.pad_token_id = pad_token_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collate(
        self,
        sequences: list[list[int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Pad a variable-length list of sequences to the maximum length.

        Parameters
        ----------
        sequences : list[list[int]]
            Each inner list is a sequence of token IDs (variable length).

        Returns
        -------
        padded_input_ids : torch.Tensor  shape (B, max_len)
            Sequences padded with `pad_token_id` on the RIGHT.
        attention_mask : torch.Tensor   shape (B, max_len)
            1 for real tokens, 0 for padding.
        """
        if not sequences:
            raise ValueError("sequences list must not be empty")

        batch_size = len(sequences)
        max_len = max(len(s) for s in sequences)

        padded = torch.full(
            (batch_size, max_len),
            fill_value=self.pad_token_id,
            dtype=torch.long,
        )
        mask = torch.zeros(batch_size, max_len, dtype=torch.long)

        for i, seq in enumerate(sequences):
            length = len(seq)
            padded[i, :length] = torch.tensor(seq, dtype=torch.long)
            mask[i, :length] = 1

        return padded, mask

    def compute_efficiency(self, sequences: list[list[int]]) -> float:
        """
        Compute the ratio of real tokens to total tokens (including padding).

        An efficiency of 1.0 means no padding waste; 0.5 means half the
        compute is wasted on padding.

        Parameters
        ----------
        sequences : list[list[int]]

        Returns
        -------
        float
            real_tokens / total_tokens, in [0, 1].
        """
        if not sequences:
            return 0.0

        real_tokens = sum(len(s) for s in sequences)
        max_len = max(len(s) for s in sequences)
        total_tokens = max_len * len(sequences)
        return real_tokens / total_tokens

    def batch_sequences(
        self,
        sequences: list[list[int]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Split a list of sequences into batches of size `max_batch_size`,
        padding each batch to its own maximum length.

        Returns a list of (padded_input_ids, attention_mask) tuples.
        """
        batches = []
        for start in range(0, len(sequences), self.max_batch_size):
            chunk = sequences[start : start + self.max_batch_size]
            batches.append(self.collate(chunk))
        return batches


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def generate_sequences(
    n: int,
    distribution: str = "uniform",
    min_len: int = 10,
    max_len: int = 500,
    mean_len: float = 200.0,
    seed: int = 42,
    vocab_size: int = 32_000,
) -> list[list[int]]:
    """
    Generate `n` random token sequences.

    Parameters
    ----------
    distribution : "uniform" | "exponential" | "bimodal"
    min_len, max_len : length bounds
    mean_len : mean length for exponential distribution
    seed : random seed for reproducibility
    vocab_size : vocabulary size for token sampling
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    sequences = []
    for _ in range(n):
        if distribution == "uniform":
            length = rng.randint(min_len, max_len)
        elif distribution == "exponential":
            length = int(np_rng.exponential(mean_len))
            length = max(min_len, min(length, max_len))
        elif distribution == "bimodal":
            # Mix of short (prompt-like) and long sequences
            if rng.random() < 0.5:
                length = rng.randint(min_len, 80)
            else:
                length = rng.randint(300, max_len)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        tokens = [rng.randint(1, vocab_size - 1) for _ in range(length)]
        sequences.append(tokens)

    return sequences


def analyze_batch_efficiency(
    sequences: list[list[int]],
    batch_size: int = 8,
    pad_token_id: int = 0,
) -> dict:
    """
    Compute padding waste statistics for static batching.

    Returns a dictionary with:
      - per_batch_efficiency : list of floats, one per batch
      - overall_efficiency   : weighted average across all batches
      - total_real_tokens    : total non-padding tokens
      - total_padded_tokens  : total tokens including padding
      - waste_pct            : padding waste as a percentage
    """
    batcher = StaticBatcher(max_batch_size=batch_size, pad_token_id=pad_token_id)
    batches = batcher.batch_sequences(sequences)

    per_batch = []
    total_real = 0
    total_padded = 0

    for padded_ids, mask in batches:
        real = int(mask.sum().item())
        padded = padded_ids.numel()
        total_real += real
        total_padded += padded
        per_batch.append(real / padded)

    overall = total_real / total_padded if total_padded > 0 else 0.0
    return {
        "per_batch_efficiency": per_batch,
        "overall_efficiency": overall,
        "total_real_tokens": total_real,
        "total_padded_tokens": total_padded,
        "waste_pct": (1 - overall) * 100,
        "num_batches": len(batches),
    }


# ---------------------------------------------------------------------------
# Timing simulation using the utility model
# ---------------------------------------------------------------------------

def time_static_batching(
    sequences: list[list[int]],
    batch_size: int = 8,
    device: str = "cpu",
    vocab_size: int = 32_000,
) -> dict:
    """
    Time a forward pass using static batching on the utility model.

    Returns timing stats (seconds) and efficiency metrics.
    """
    model = create_model(
        num_layers=4,
        hidden_dim=256,
        num_heads=4,
        vocab_size=vocab_size,
        device=device,
    )
    model.eval()

    batcher = StaticBatcher(max_batch_size=batch_size)
    batches = batcher.batch_sequences(sequences)

    latencies = []
    with torch.no_grad():
        for padded_ids, _mask in batches:
            padded_ids = padded_ids.to(device)
            t0 = time.perf_counter()
            _ = model(padded_ids)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms

    efficiency = analyze_batch_efficiency(sequences, batch_size)
    return {
        "mean_latency_ms": float(np.mean(latencies)),
        "total_latency_ms": float(np.sum(latencies)),
        "num_batches": len(batches),
        **efficiency,
    }


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def run_simulation():
    """
    Simulate 32 requests with uniform and varied length distributions,
    demonstrating padding waste for static batching.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running static batching simulation on device: {device}")
    print("=" * 65)

    configs = [
        ("Uniform lengths (10–500)",  "uniform",     10, 500, 200),
        ("Exponential (mean=200)",     "exponential", 10, 500, 200),
        ("Bimodal (short + long)",     "bimodal",     10, 500, 200),
    ]

    for label, dist, lo, hi, mean in configs:
        seqs = generate_sequences(
            n=32, distribution=dist, min_len=lo, max_len=hi, mean_len=mean
        )
        lengths = [len(s) for s in seqs]

        print(f"\n--- {label} ---")
        print(f"  Sequence lengths: min={min(lengths)}, max={max(lengths)}, "
              f"mean={np.mean(lengths):.1f}, std={np.std(lengths):.1f}")

        for bs in [4, 8, 16]:
            stats = analyze_batch_efficiency(seqs, batch_size=bs)
            print(
                f"  batch_size={bs:2d}: "
                f"efficiency={stats['overall_efficiency']:.3f}, "
                f"waste={stats['waste_pct']:.1f}%, "
                f"batches={stats['num_batches']}"
            )

    # ---- Detailed demonstration with a concrete example ----
    print("\n" + "=" * 65)
    print("Concrete example: 8 sequences, batch_size=8")
    print("(All in one batch — padded to the longest sequence)")

    demo_seqs = [
        [1] * 10,   # very short
        [1] * 25,
        [1] * 50,
        [1] * 100,
        [1] * 200,
        [1] * 300,
        [1] * 400,
        [1] * 500,  # longest — dictates padding for entire batch
    ]

    batcher = StaticBatcher(max_batch_size=8)
    padded, mask = batcher.collate(demo_seqs)
    eff = batcher.compute_efficiency(demo_seqs)

    print(f"\n  Padded shape: {list(padded.shape)}")
    print(f"  Real tokens:  {int(mask.sum())}")
    print(f"  Total tokens: {padded.numel()}")
    print(f"  Efficiency:   {eff:.3f}  ({(1-eff)*100:.1f}% wasted on padding)")

    print("\n  Per-sequence padding:")
    max_len = padded.shape[1]
    for i, seq in enumerate(demo_seqs):
        pad = max_len - len(seq)
        print(f"    seq[{i}]: len={len(seq):3d}, padded_to={max_len}, "
              f"wasted={pad:3d} tokens ({pad/max_len*100:.1f}%)")

    # ---- Timing comparison ----
    print("\n" + "=" * 65)
    print("Timing: uniform vs exponential distribution (32 requests, batch=8)")

    for dist, label in [("uniform", "Uniform"), ("exponential", "Exponential")]:
        seqs = generate_sequences(n=32, distribution=dist)
        stats = time_static_batching(seqs, batch_size=8, device=device)
        print(
            f"  {label:12s}: total={stats['total_latency_ms']:.1f}ms, "
            f"mean/batch={stats['mean_latency_ms']:.1f}ms, "
            f"waste={stats['waste_pct']:.1f}%"
        )

    print("\nKey insight: exponential length distribution has much higher padding")
    print("waste because one outlier sequence forces the whole batch to be long.")


if __name__ == "__main__":
    run_simulation()
