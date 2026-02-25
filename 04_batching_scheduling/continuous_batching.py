"""
Chapter 4 — Batching & Scheduling
FILE 3: continuous_batching.py

Continuous (in-flight) batching — the key technique used by production
serving systems like vLLM, TensorRT-LLM, and Orca.

Unlike static batching (wait for all sequences to finish, then start next
batch), continuous batching:
  - Inserts new requests as soon as a slot opens (a sequence finishes)
  - Removes finished sequences immediately (no wasted compute on padding)
  - Processes prefill and decode tokens in the same forward pass

This maximizes GPU utilization by keeping the batch full at all times.

Contents:
  - Request           : single inference request with state tracking
  - ContinuousBatcher : manages active batch, inserts/evicts on the fly
  - simulate          : simulate a sequence of requests with varying lengths
"""

import sys
import os
import time
import random
from dataclasses import dataclass, field
from typing import Optional
from collections import deque

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.model_loader import create_model


# ---------------------------------------------------------------------------
# Request & Batch Slot
# ---------------------------------------------------------------------------

@dataclass
class Request:
    """A single generation request."""
    id: int
    prompt_tokens: list[int]           # input token IDs
    max_new_tokens: int = 128          # max tokens to generate
    generated_tokens: list[int] = field(default_factory=list)
    arrival_time: float = 0.0          # when the request arrived
    start_time: float = 0.0            # when processing started
    finish_time: float = 0.0           # when generation completed

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_tokens)

    @property
    def generated_len(self) -> int:
        return len(self.generated_tokens)

    @property
    def total_len(self) -> int:
        return self.prompt_len + self.generated_len

    @property
    def is_done(self) -> bool:
        return self.generated_len >= self.max_new_tokens

    @property
    def latency_ms(self) -> float:
        return (self.finish_time - self.arrival_time) * 1000

    @property
    def time_to_first_token_ms(self) -> float:
        return (self.start_time - self.arrival_time) * 1000


# ---------------------------------------------------------------------------
# Continuous Batcher
# ---------------------------------------------------------------------------

class ContinuousBatcher:
    """
    Continuous batching engine.

    Maintains a running batch of active sequences. At each step:
      1. Check for finished sequences → remove them
      2. Fill empty slots with waiting requests → run prefill
      3. Run one decode step for all active sequences

    Args:
        max_batch_size: maximum concurrent sequences
        max_seq_len: maximum total tokens per sequence (prompt + generated)
    """

    def __init__(self, max_batch_size: int = 8, max_seq_len: int = 2048):
        self.max_batch_size = max_batch_size
        self.max_seq_len    = max_seq_len

        self.waiting_queue: deque[Request] = deque()
        self.active_batch: list[Optional[Request]] = [None] * max_batch_size
        self.completed: list[Request] = []

        # Stats
        self.total_steps = 0
        self.total_tokens_generated = 0

    def add_request(self, request: Request):
        """Add a request to the waiting queue."""
        self.waiting_queue.append(request)

    @property
    def num_active(self) -> int:
        return sum(1 for r in self.active_batch if r is not None)

    @property
    def num_waiting(self) -> int:
        return len(self.waiting_queue)

    @property
    def num_free_slots(self) -> int:
        return sum(1 for r in self.active_batch if r is None)

    def _evict_finished(self) -> int:
        """Remove finished sequences from the active batch. Returns count evicted."""
        evicted = 0
        for i, req in enumerate(self.active_batch):
            if req is not None and req.is_done:
                req.finish_time = time.perf_counter()
                self.completed.append(req)
                self.active_batch[i] = None
                evicted += 1
        return evicted

    def _fill_slots(self) -> int:
        """Fill empty slots with waiting requests. Returns count added."""
        added = 0
        now = time.perf_counter()
        for i, req in enumerate(self.active_batch):
            if req is None and self.waiting_queue:
                new_req = self.waiting_queue.popleft()
                new_req.start_time = now
                self.active_batch[i] = new_req
                added += 1
        return added

    def step(self) -> dict:
        """
        Execute one iteration of the continuous batching loop.

        Returns dict with step statistics.
        """
        # 1. Evict finished sequences
        evicted = self._evict_finished()

        # 2. Fill empty slots
        added = self._fill_slots()

        # 3. Decode step: generate one token per active sequence
        decode_count = 0
        for req in self.active_batch:
            if req is not None and not req.is_done:
                # Simulate token generation (random token)
                req.generated_tokens.append(random.randint(1, 31999))
                decode_count += 1

        self.total_steps += 1
        self.total_tokens_generated += decode_count

        return {
            "step": self.total_steps,
            "evicted": evicted,
            "added": added,
            "active": self.num_active,
            "waiting": self.num_waiting,
            "tokens_generated": decode_count,
        }

    def run_to_completion(self, verbose: bool = True) -> dict:
        """Run until all requests are completed."""
        if verbose:
            print(f"\n  {'Step':>6}  {'Active':>7}  {'Wait':>6}  {'Evict':>6}  "
                  f"{'Add':>5}  {'Tokens':>7}  {'Done':>6}")
            print(f"  {'-' * 6}  {'-' * 7}  {'-' * 6}  {'-' * 6}  "
                  f"{'-' * 5}  {'-' * 7}  {'-' * 6}")

        while self.num_active > 0 or self.num_waiting > 0:
            stats = self.step()

            if verbose and (stats["step"] <= 5 or stats["step"] % 20 == 0
                            or stats["evicted"] > 0 or self.num_active == 0):
                print(f"  {stats['step']:>6}  {stats['active']:>7}  "
                      f"{stats['waiting']:>6}  {stats['evicted']:>6}  "
                      f"{stats['added']:>5}  {stats['tokens_generated']:>7}  "
                      f"{len(self.completed):>6}")

        return self._summary()

    def _summary(self) -> dict:
        latencies = [r.latency_ms for r in self.completed]
        ttfts = [r.time_to_first_token_ms for r in self.completed]
        return {
            "total_requests": len(self.completed),
            "total_steps": self.total_steps,
            "total_tokens": self.total_tokens_generated,
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
            "avg_ttft_ms": sum(ttfts) / len(ttfts) if ttfts else 0,
            "throughput_tok_per_step": (
                self.total_tokens_generated / self.total_steps
                if self.total_steps > 0 else 0
            ),
        }


# ---------------------------------------------------------------------------
# Comparison: static vs continuous batching
# ---------------------------------------------------------------------------

def compare_batching(
    num_requests: int = 16,
    max_batch_size: int = 4,
    min_gen_len: int = 20,
    max_gen_len: int = 200,
):
    """
    Compare static batching (wait for all to finish) vs continuous batching.
    """
    random.seed(42)

    # Generate requests with varying generation lengths
    requests = []
    for i in range(num_requests):
        gen_len = random.randint(min_gen_len, max_gen_len)
        prompt = [random.randint(1, 31999) for _ in range(random.randint(5, 50))]
        requests.append(Request(id=i, prompt_tokens=prompt, max_new_tokens=gen_len))

    gen_lengths = [r.max_new_tokens for r in requests]

    # ---- Static batching simulation ----
    # Process in fixed batches, wait for all to finish before next batch
    static_steps = 0
    static_tokens = 0
    for batch_start in range(0, num_requests, max_batch_size):
        batch = requests[batch_start:batch_start + max_batch_size]
        max_gen = max(r.max_new_tokens for r in batch)
        # All sequences run for max_gen steps (padded)
        static_steps += max_gen
        static_tokens += max_gen * len(batch)  # includes wasted padding tokens
    static_useful_tokens = sum(gen_lengths)

    # ---- Continuous batching simulation ----
    cont_requests = [
        Request(id=r.id, prompt_tokens=list(r.prompt_tokens),
                max_new_tokens=r.max_new_tokens)
        for r in requests
    ]

    batcher = ContinuousBatcher(max_batch_size=max_batch_size)
    now = time.perf_counter()
    for r in cont_requests:
        r.arrival_time = now
        batcher.add_request(r)

    print(f"\n  Continuous Batching Simulation")
    print(f"  {num_requests} requests, max_batch={max_batch_size}")
    print(f"  Generation lengths: {min_gen_len}-{max_gen_len} tokens")

    summary = batcher.run_to_completion(verbose=True)

    print(f"\n  === Comparison ===")
    print(f"  {'Metric':<30}  {'Static':>12}  {'Continuous':>12}")
    print(f"  {'-' * 30}  {'-' * 12}  {'-' * 12}")
    print(f"  {'Total steps':<30}  {static_steps:>12}  {summary['total_steps']:>12}")
    print(f"  {'Total token slots used':<30}  {static_tokens:>12}  {summary['total_tokens']:>12}")
    print(f"  {'Useful tokens':<30}  {static_useful_tokens:>12}  {summary['total_tokens']:>12}")
    eff_static = static_useful_tokens / static_tokens * 100 if static_tokens > 0 else 0
    eff_cont = 100.0  # continuous batching has no wasted slots
    print(f"  {'Compute efficiency':<30}  {eff_static:>11.1f}%  {eff_cont:>11.1f}%")
    print(f"  {'Avg tokens/step':<30}  "
          f"{static_useful_tokens / static_steps:>12.1f}  "
          f"{summary['throughput_tok_per_step']:>12.1f}")

    speedup = static_steps / summary['total_steps'] if summary['total_steps'] > 0 else 0
    print(f"\n  Continuous batching completes {speedup:.1f}x faster (fewer total steps)")
    print(f"  because finished sequences free up slots immediately for new ones.")


# ---------------------------------------------------------------------------
# Throughput scaling analysis
# ---------------------------------------------------------------------------

def throughput_scaling(max_batch_sizes: list = None):
    """Show how throughput scales with batch size in continuous batching."""
    if max_batch_sizes is None:
        max_batch_sizes = [1, 2, 4, 8, 16, 32]

    random.seed(42)
    num_requests = 64

    print(f"\n  Throughput Scaling: {num_requests} requests")
    print(f"\n  {'Batch Size':>12}  {'Steps':>8}  {'Tok/Step':>10}  {'Speedup':>10}")
    print(f"  {'-' * 12}  {'-' * 8}  {'-' * 10}  {'-' * 10}")

    base_steps = None
    for bs in max_batch_sizes:
        requests = []
        for i in range(num_requests):
            gen_len = random.randint(20, 200)
            prompt = [random.randint(1, 31999) for _ in range(10)]
            requests.append(Request(id=i, prompt_tokens=prompt, max_new_tokens=gen_len))

        batcher = ContinuousBatcher(max_batch_size=bs)
        now = time.perf_counter()
        for r in requests:
            r.arrival_time = now
            batcher.add_request(r)

        summary = batcher.run_to_completion(verbose=False)
        if base_steps is None:
            base_steps = summary["total_steps"]

        speedup = base_steps / summary["total_steps"] if summary["total_steps"] > 0 else 0
        print(f"  {bs:>12}  {summary['total_steps']:>8}  "
              f"{summary['throughput_tok_per_step']:>10.1f}  {speedup:>9.1f}x")

    print(f"\n  Larger batch → more requests processed concurrently → fewer steps → higher throughput.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("  Chapter 4: Continuous Batching (In-Flight Batching)")
    print("=" * 70)

    print("\n  [1] Static vs Continuous Batching Comparison")
    compare_batching()

    print("\n  [2] Throughput scaling with batch size")
    throughput_scaling()

    print("\n  Key takeaways:")
    print("  - Static batching wastes compute on padding (all wait for slowest)")
    print("  - Continuous batching fills slots immediately when sequences finish")
    print("  - Result: higher GPU utilization and throughput, lower latency")
    print("  - Used by vLLM, TensorRT-LLM, Orca, and other production systems")
