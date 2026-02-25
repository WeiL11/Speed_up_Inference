"""
Chapter 4 — Batching & Scheduling
FILE 2: dynamic_batching.py

Length-based bucket batching to dramatically reduce padding waste.

Instead of padding every sequence in a batch to the global maximum length,
we group sequences into "buckets" defined by fixed length boundaries. Each
sequence is padded only to its bucket ceiling, not to the worst-case length
in the entire request set.

Example with BUCKET_SIZES = [64, 128, 256, 512, 1024]:
  - A sequence of length 40  -> bucket 64   (24 tokens wasted)
  - A sequence of length 100 -> bucket 128  (28 tokens wasted)
  - A sequence of length 300 -> bucket 512  (212 tokens wasted)

Versus static batching (global max = 512):
  - The 40-token sequence wastes 472 tokens instead of 24.
"""

import sys
import os
import random
import time
from collections import defaultdict
from typing import Optional

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.model_loader import create_model
from static_batching import generate_sequences, StaticBatcher


# ---------------------------------------------------------------------------
# Bucket boundaries
# ---------------------------------------------------------------------------

BUCKET_SIZES: list[int] = [64, 128, 256, 512, 1024, 2048]


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class DynamicBatcher:
    """
    Groups sequences into buckets based on length to reduce padding waste.

    Sequences in the same bucket are padded to the bucket boundary, which
    is far less waste than padding every sequence to the global maximum
    (as static batching does).

    Parameters
    ----------
    bucket_sizes : list[int]
        Sorted list of bucket boundaries (e.g. [64, 128, 256, 512]).
        Sequences longer than the largest bucket are placed in an
        overflow bucket padded to the actual sequence maximum.
    max_batch_size : int
        Maximum number of sequences per batch within a bucket.
    pad_token_id : int
        Token ID used for padding (default 0).
    """

    def __init__(
        self,
        bucket_sizes: list[int],
        max_batch_size: int,
        pad_token_id: int = 0,
    ):
        self.bucket_sizes = sorted(bucket_sizes)
        self.max_batch_size = max_batch_size
        self.pad_token_id = pad_token_id

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def get_bucket(self, seq_len: int) -> int:
        """
        Return the smallest bucket size >= seq_len.

        If seq_len exceeds all defined buckets, return seq_len itself
        (overflow — no extra padding beyond what the sequence needs).

        Parameters
        ----------
        seq_len : int
            Length of the sequence.

        Returns
        -------
        int
            Bucket ceiling to pad to.
        """
        for b in self.bucket_sizes:
            if seq_len <= b:
                return b
        # Overflow: sequence longer than largest bucket
        return seq_len

    def _pad_to_length(
        self,
        sequences: list[list[int]],
        target_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad a list of sequences to `target_len` and return (ids, mask)."""
        B = len(sequences)
        padded = torch.full(
            (B, target_len),
            fill_value=self.pad_token_id,
            dtype=torch.long,
        )
        mask = torch.zeros(B, target_len, dtype=torch.long)
        for i, seq in enumerate(sequences):
            L = len(seq)
            padded[i, :L] = torch.tensor(seq, dtype=torch.long)
            mask[i, :L] = 1
        return padded, mask

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collate(
        self,
        sequences: list[list[int]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Group sequences by bucket, then split each bucket into batches
        of at most `max_batch_size`.

        Parameters
        ----------
        sequences : list[list[int]]
            Variable-length token sequences.

        Returns
        -------
        list of (padded_input_ids, attention_mask)
            One tuple per sub-batch (bucket group). Each tuple has shape
            (B_i, bucket_len) where B_i <= max_batch_size.
        """
        # Assign each sequence to a bucket
        bucket_map: dict[int, list[list[int]]] = defaultdict(list)
        for seq in sequences:
            bucket = self.get_bucket(len(seq))
            bucket_map[bucket].append(seq)

        batches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for bucket_len in sorted(bucket_map.keys()):
            bucket_seqs = bucket_map[bucket_len]
            # Split into sub-batches of max_batch_size
            for start in range(0, len(bucket_seqs), self.max_batch_size):
                chunk = bucket_seqs[start : start + self.max_batch_size]
                padded, mask = self._pad_to_length(chunk, bucket_len)
                batches.append((padded, mask))

        return batches

    def efficiency_vs_static(self, sequences: list[list[int]]) -> dict:
        """
        Compute and compare padding efficiency for dynamic vs static batching.

        Parameters
        ----------
        sequences : list[list[int]]

        Returns
        -------
        dict with keys:
          dynamic_efficiency  : float [0, 1]
          static_efficiency   : float [0, 1]
          dynamic_waste_pct   : float
          static_waste_pct    : float
          improvement_factor  : dynamic_waste_pct / static_waste_pct if > 0
          dynamic_batches     : int  (total sub-batches produced)
          per_bucket_stats    : dict  bucket_len -> {count, efficiency}
        """
        # ---- Dynamic efficiency ----
        dynamic_batches = self.collate(sequences)
        d_real = sum(int(m.sum()) for _, m in dynamic_batches)
        d_total = sum(p.numel() for p, _ in dynamic_batches)
        d_eff = d_real / d_total if d_total > 0 else 0.0

        # ---- Static efficiency (pad to global max) ----
        static_batcher = StaticBatcher(
            max_batch_size=self.max_batch_size,
            pad_token_id=self.pad_token_id,
        )
        static_batches = static_batcher.batch_sequences(sequences)
        s_real = sum(int(m.sum()) for _, m in static_batches)
        s_total = sum(p.numel() for p, _ in static_batches)
        s_eff = s_real / s_total if s_total > 0 else 0.0

        # ---- Per-bucket breakdown ----
        bucket_map: dict[int, list[list[int]]] = defaultdict(list)
        for seq in sequences:
            b = self.get_bucket(len(seq))
            bucket_map[b].append(seq)

        per_bucket = {}
        for blen, seqs in sorted(bucket_map.items()):
            real = sum(len(s) for s in seqs)
            total = blen * len(seqs)
            per_bucket[blen] = {
                "count": len(seqs),
                "efficiency": real / total,
                "waste_pct": (1 - real / total) * 100,
            }

        d_waste = (1 - d_eff) * 100
        s_waste = (1 - s_eff) * 100
        improvement = s_waste / d_waste if d_waste > 0 else float("inf")

        return {
            "dynamic_efficiency": d_eff,
            "static_efficiency": s_eff,
            "dynamic_waste_pct": d_waste,
            "static_waste_pct": s_waste,
            "improvement_factor": improvement,
            "dynamic_batches": len(dynamic_batches),
            "static_batches": len(static_batches),
            "per_bucket_stats": per_bucket,
        }


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def time_dynamic_batching(
    sequences: list[list[int]],
    bucket_sizes: list[int] = BUCKET_SIZES,
    batch_size: int = 8,
    device: str = "cpu",
    vocab_size: int = 32_000,
) -> dict:
    """
    Time forward passes using dynamic (bucket) batching.
    """
    model = create_model(
        num_layers=4,
        hidden_dim=256,
        num_heads=4,
        vocab_size=vocab_size,
        device=device,
    )
    model.eval()

    batcher = DynamicBatcher(
        bucket_sizes=bucket_sizes,
        max_batch_size=batch_size,
    )
    batches = batcher.collate(sequences)

    latencies = []
    with torch.no_grad():
        for padded_ids, _mask in batches:
            padded_ids = padded_ids.to(device)
            t0 = time.perf_counter()
            _ = model(padded_ids)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    stats = batcher.efficiency_vs_static(sequences)
    return {
        "mean_latency_ms": float(np.mean(latencies)),
        "total_latency_ms": float(np.sum(latencies)),
        "num_batches": len(batches),
        **stats,
    }


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def run_simulation():
    """
    Simulate 100 requests with exponential length distribution (mean=200).
    Compare static vs dynamic batching efficiency.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running dynamic batching simulation on device: {device}")
    print("=" * 70)

    # Generate 100 requests with exponential distribution (mean=200)
    seqs = generate_sequences(
        n=100,
        distribution="exponential",
        min_len=10,
        max_len=2000,
        mean_len=200,
        seed=42,
    )
    lengths = [len(s) for s in seqs]

    print(f"\nDataset: 100 sequences, exponential distribution (mean=200)")
    print(f"  Length stats: min={min(lengths)}, max={max(lengths)}, "
          f"mean={np.mean(lengths):.1f}, std={np.std(lengths):.1f}")

    # ---- Efficiency comparison for several batch sizes ----
    print("\n--- Static vs Dynamic Efficiency ---")
    print(f"{'Batch':>6}  {'Static Eff':>10}  {'Dynamic Eff':>11}  "
          f"{'Static Waste':>12}  {'Dynamic Waste':>13}  {'Speedup':>7}")
    print("-" * 70)

    for bs in [4, 8, 16, 32]:
        batcher = DynamicBatcher(bucket_sizes=BUCKET_SIZES, max_batch_size=bs)
        stats = batcher.efficiency_vs_static(seqs)
        print(
            f"{bs:>6}  "
            f"{stats['static_efficiency']:>10.3f}  "
            f"{stats['dynamic_efficiency']:>11.3f}  "
            f"{stats['static_waste_pct']:>11.1f}%  "
            f"{stats['dynamic_waste_pct']:>12.1f}%  "
            f"{stats['improvement_factor']:>6.1f}x"
        )

    # ---- Per-bucket breakdown ----
    print("\n--- Per-Bucket Breakdown (batch_size=16) ---")
    batcher = DynamicBatcher(bucket_sizes=BUCKET_SIZES, max_batch_size=16)
    stats = batcher.efficiency_vs_static(seqs)

    print(f"{'Bucket':>7}  {'Count':>5}  {'Efficiency':>10}  {'Waste':>7}")
    print("-" * 36)
    for blen, bstats in stats["per_bucket_stats"].items():
        print(
            f"{blen:>7}  "
            f"{bstats['count']:>5}  "
            f"{bstats['efficiency']:>10.3f}  "
            f"{bstats['waste_pct']:>6.1f}%"
        )
    print(f"\n  Total dynamic batches: {stats['dynamic_batches']}")
    print(f"  Total static  batches: {stats['static_batches']}")

    # ---- Different bucket configurations ----
    print("\n--- Effect of Bucket Granularity ---")
    bucket_configs = [
        ("Coarse [128, 512, 2048]",     [128, 512, 2048]),
        ("Default [64,128,256,512,1024,2048]", BUCKET_SIZES),
        ("Fine [32,64,128,192,256,320,512,1024,2048]",
         [32, 64, 128, 192, 256, 320, 512, 1024, 2048]),
    ]

    for label, buckets in bucket_configs:
        b = DynamicBatcher(bucket_sizes=buckets, max_batch_size=16)
        s = b.efficiency_vs_static(seqs)
        print(f"  {label}")
        print(f"    Dynamic efficiency: {s['dynamic_efficiency']:.3f} "
              f"(waste {s['dynamic_waste_pct']:.1f}%), "
              f"batches: {s['dynamic_batches']}")

    # ---- Timing comparison ----
    print("\n--- Timing Comparison (batch_size=8) ---")
    from static_batching import time_static_batching

    static_t = time_static_batching(seqs, batch_size=8, device=device)
    dynamic_t = time_dynamic_batching(seqs, batch_size=8, device=device)

    print(f"  Static  batching: total={static_t['total_latency_ms']:.1f}ms, "
          f"mean/batch={static_t['mean_latency_ms']:.1f}ms, "
          f"waste={static_t['waste_pct']:.1f}%")
    print(f"  Dynamic batching: total={dynamic_t['total_latency_ms']:.1f}ms, "
          f"mean/batch={dynamic_t['mean_latency_ms']:.1f}ms, "
          f"waste={dynamic_t['dynamic_waste_pct']:.1f}%")

    print("\nKey insight: dynamic batching trades a modest increase in the")
    print("number of kernel launches for a significant reduction in wasted")
    print("compute on padding tokens, especially with skewed length distributions.")


if __name__ == "__main__":
    run_simulation()
