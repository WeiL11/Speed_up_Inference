"""
Chapter 4 — Batching & Scheduling
FILE 4: scheduler.py

Request scheduling strategies for LLM inference serving.

When more requests arrive than can fit in a single batch, a scheduler
decides which requests to process first. Different policies optimize
for different metrics:

  - FCFS (First-Come First-Served): fair ordering, simple implementation
  - SJF (Shortest Job First): minimizes average latency
  - Priority: lets callers specify urgency levels

Contents:
  - FCFSScheduler    : process requests in arrival order
  - SJFScheduler     : shortest expected generation first
  - PriorityScheduler: priority queue with preemption support
  - compare          : simulate all policies and compare metrics
"""

import sys
import os
import time
import random
import heapq
from dataclasses import dataclass, field
from typing import Optional
from collections import deque
from abc import ABC, abstractmethod

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

@dataclass
class SchedulerRequest:
    """Request with scheduling metadata."""
    id: int
    prompt_len: int
    max_new_tokens: int
    priority: int = 0               # higher = more urgent
    arrival_step: int = 0
    start_step: int = -1
    finish_step: int = -1
    tokens_generated: int = 0

    @property
    def is_done(self) -> bool:
        return self.tokens_generated >= self.max_new_tokens

    @property
    def wait_time(self) -> int:
        """Steps waiting before processing started."""
        return self.start_step - self.arrival_step if self.start_step >= 0 else -1

    @property
    def total_time(self) -> int:
        """Total steps from arrival to completion."""
        return self.finish_step - self.arrival_step if self.finish_step >= 0 else -1


# ---------------------------------------------------------------------------
# Base Scheduler
# ---------------------------------------------------------------------------

class BaseScheduler(ABC):
    """Abstract base class for request schedulers."""

    def __init__(self, max_batch_size: int = 8):
        self.max_batch_size = max_batch_size
        self.active: list[Optional[SchedulerRequest]] = [None] * max_batch_size
        self.completed: list[SchedulerRequest] = []
        self.current_step = 0

    @abstractmethod
    def add_request(self, request: SchedulerRequest):
        """Add a request to the scheduling queue."""
        pass

    @abstractmethod
    def _select_next(self) -> Optional[SchedulerRequest]:
        """Select the next request from the queue."""
        pass

    @property
    def num_active(self) -> int:
        return sum(1 for r in self.active if r is not None)

    @property
    def has_waiting(self) -> bool:
        return False  # subclasses override

    def step(self):
        """Run one scheduling + decode step."""
        # Evict finished
        for i, req in enumerate(self.active):
            if req is not None and req.is_done:
                req.finish_step = self.current_step
                self.completed.append(req)
                self.active[i] = None

        # Fill empty slots
        for i in range(self.max_batch_size):
            if self.active[i] is None and self.has_waiting:
                req = self._select_next()
                if req is not None:
                    req.start_step = self.current_step
                    self.active[i] = req

        # Decode: generate one token per active request
        for req in self.active:
            if req is not None and not req.is_done:
                req.tokens_generated += 1

        self.current_step += 1

    def run_to_completion(self, requests: list[SchedulerRequest]):
        """Process all requests until done."""
        for r in requests:
            r.arrival_step = self.current_step
            self.add_request(r)

        while self.num_active > 0 or self.has_waiting:
            self.step()

        return self._metrics()

    def _metrics(self) -> dict:
        wait_times = [r.wait_time for r in self.completed if r.wait_time >= 0]
        total_times = [r.total_time for r in self.completed if r.total_time >= 0]
        return {
            "total_requests": len(self.completed),
            "total_steps": self.current_step,
            "avg_wait": sum(wait_times) / len(wait_times) if wait_times else 0,
            "max_wait": max(wait_times) if wait_times else 0,
            "avg_total": sum(total_times) / len(total_times) if total_times else 0,
            "p99_total": sorted(total_times)[int(len(total_times) * 0.99)] if total_times else 0,
        }


# ---------------------------------------------------------------------------
# FCFS Scheduler
# ---------------------------------------------------------------------------

class FCFSScheduler(BaseScheduler):
    """First-Come, First-Served scheduler."""

    def __init__(self, max_batch_size: int = 8):
        super().__init__(max_batch_size)
        self._queue: deque[SchedulerRequest] = deque()

    def add_request(self, request: SchedulerRequest):
        self._queue.append(request)

    def _select_next(self) -> Optional[SchedulerRequest]:
        return self._queue.popleft() if self._queue else None

    @property
    def has_waiting(self) -> bool:
        return len(self._queue) > 0


# ---------------------------------------------------------------------------
# SJF Scheduler
# ---------------------------------------------------------------------------

class SJFScheduler(BaseScheduler):
    """
    Shortest Job First scheduler.

    Prioritizes requests with the smallest max_new_tokens, which minimizes
    average completion time (provably optimal for non-preemptive scheduling
    with known job sizes).
    """

    def __init__(self, max_batch_size: int = 8):
        super().__init__(max_batch_size)
        self._heap: list[tuple[int, int, SchedulerRequest]] = []
        self._counter = 0

    def add_request(self, request: SchedulerRequest):
        heapq.heappush(self._heap, (request.max_new_tokens, self._counter, request))
        self._counter += 1

    def _select_next(self) -> Optional[SchedulerRequest]:
        if self._heap:
            _, _, req = heapq.heappop(self._heap)
            return req
        return None

    @property
    def has_waiting(self) -> bool:
        return len(self._heap) > 0


# ---------------------------------------------------------------------------
# Priority Scheduler
# ---------------------------------------------------------------------------

class PriorityScheduler(BaseScheduler):
    """
    Priority-based scheduler with optional preemption.

    Higher priority value = scheduled first. Ties broken by arrival order.
    """

    def __init__(self, max_batch_size: int = 8):
        super().__init__(max_batch_size)
        self._heap: list[tuple[int, int, SchedulerRequest]] = []
        self._counter = 0

    def add_request(self, request: SchedulerRequest):
        # Negate priority for min-heap (higher priority = smaller key)
        heapq.heappush(self._heap, (-request.priority, self._counter, request))
        self._counter += 1

    def _select_next(self) -> Optional[SchedulerRequest]:
        if self._heap:
            _, _, req = heapq.heappop(self._heap)
            return req
        return None

    @property
    def has_waiting(self) -> bool:
        return len(self._heap) > 0


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_schedulers(
    num_requests: int = 32,
    max_batch_size: int = 4,
    min_gen: int = 10,
    max_gen: int = 300,
):
    """Compare FCFS, SJF, and Priority schedulers."""
    random.seed(42)

    # Generate request specifications
    specs = []
    for i in range(num_requests):
        gen_len = random.randint(min_gen, max_gen)
        prompt_len = random.randint(5, 50)
        priority = random.choice([0, 0, 0, 1, 1, 2])  # most are low priority
        specs.append((i, prompt_len, gen_len, priority))

    schedulers = {
        "FCFS":     lambda: FCFSScheduler(max_batch_size),
        "SJF":      lambda: SJFScheduler(max_batch_size),
        "Priority": lambda: PriorityScheduler(max_batch_size),
    }

    print(f"\n  Scheduler Comparison: {num_requests} requests, batch_size={max_batch_size}")
    print(f"  Generation lengths: {min_gen}-{max_gen} tokens")
    print(f"\n  {'Scheduler':<12}  {'Steps':>8}  {'Avg Wait':>10}  {'Max Wait':>10}  "
          f"{'Avg Total':>11}  {'P99 Total':>11}")
    print(f"  {'-' * 12}  {'-' * 8}  {'-' * 10}  {'-' * 10}  {'-' * 11}  {'-' * 11}")

    for name, make_scheduler in schedulers.items():
        scheduler = make_scheduler()
        requests = [
            SchedulerRequest(id=i, prompt_len=pl, max_new_tokens=gl, priority=pr)
            for i, pl, gl, pr in specs
        ]
        metrics = scheduler.run_to_completion(requests)
        print(f"  {name:<12}  {metrics['total_steps']:>8}  "
              f"{metrics['avg_wait']:>10.1f}  {metrics['max_wait']:>10}  "
              f"{metrics['avg_total']:>11.1f}  {metrics['p99_total']:>11}")


def priority_fairness_demo():
    """Show how priority scheduling affects wait time by priority level."""
    random.seed(42)
    num_requests = 48
    max_batch_size = 4

    scheduler = PriorityScheduler(max_batch_size)
    requests = []
    for i in range(num_requests):
        gen_len = random.randint(20, 100)
        priority = i % 3  # 0, 1, 2 cycling
        requests.append(SchedulerRequest(
            id=i, prompt_len=10, max_new_tokens=gen_len, priority=priority
        ))

    scheduler.run_to_completion(requests)

    print(f"\n  Priority Fairness Analysis ({num_requests} requests)")
    for p in [0, 1, 2]:
        reqs = [r for r in scheduler.completed if r.priority == p]
        if reqs:
            waits = [r.wait_time for r in reqs]
            totals = [r.total_time for r in reqs]
            print(f"  Priority {p}: count={len(reqs)}, "
                  f"avg_wait={sum(waits)/len(waits):.0f}, "
                  f"avg_total={sum(totals)/len(totals):.0f}")

    print(f"\n  Higher priority → lower wait time (scheduled sooner)")
    print(f"  Trade-off: low-priority requests starve when system is loaded")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("  Chapter 4: Request Scheduling")
    print("=" * 70)

    print("\n  [1] Scheduler comparison")
    compare_schedulers()

    print("\n  [2] Priority fairness analysis")
    priority_fairness_demo()

    print("\n  Key takeaways:")
    print("  - FCFS: simple and fair, but not optimal for average latency")
    print("  - SJF: minimizes average latency, but long requests may starve")
    print("  - Priority: flexible, but requires careful priority assignment")
    print("  - Real systems combine strategies (e.g., priority + aging to prevent starvation)")
