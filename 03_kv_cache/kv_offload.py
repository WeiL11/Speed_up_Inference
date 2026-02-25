"""
Chapter 3 — KV Cache Management
File: kv_offload.py

CPU ↔ GPU KV cache offloading.

For long sequences, the KV cache can exceed GPU memory. Offloading moves
older (less frequently accessed) KV blocks to CPU RAM and prefetches them
back to GPU when needed.

Strategy:
  - Keep the most recent `gpu_budget` tokens on GPU (hot cache)
  - Older tokens are offloaded to CPU (cold cache)
  - When attention needs the full context, prefetch from CPU with async copy
  - Use CUDA streams + pinned memory for overlapping transfer with compute

Contents:
  - KVOffloader       : manages GPU/CPU cache split with async prefetch
  - benchmark_offload : measure latency impact of offloading
"""

import sys
import time
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# KV Offloader
# ---------------------------------------------------------------------------

class KVOffloader:
    """
    Manages KV cache with GPU (hot) and CPU (cold) tiers.

    The most recent `gpu_budget` tokens stay on GPU. Older tokens are
    offloaded to CPU pinned memory for fast async transfer.

    Args:
        num_kv_heads: number of KV heads
        head_dim: dimension per head
        gpu_budget: max tokens to keep on GPU
        max_seq_len: total max tokens (GPU + CPU combined)
        device: GPU device
        dtype: tensor dtype
        use_pinned: use pinned (page-locked) CPU memory for faster transfers
    """

    def __init__(
        self,
        num_kv_heads: int,
        head_dim: int,
        gpu_budget: int = 512,
        max_seq_len: int = 4096,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        use_pinned: bool = True,
    ):
        self.num_kv_heads = num_kv_heads
        self.head_dim     = head_dim
        self.gpu_budget   = gpu_budget
        self.max_seq_len  = max_seq_len
        self.device       = device
        self.dtype        = dtype
        self.use_pinned   = use_pinned

        self.pos = 0

        # GPU buffer: holds most recent tokens
        self.k_gpu = torch.zeros(
            num_kv_heads, gpu_budget, head_dim,
            device=device, dtype=dtype,
        )
        self.v_gpu = torch.zeros(
            num_kv_heads, gpu_budget, head_dim,
            device=device, dtype=dtype,
        )

        # CPU buffer: holds offloaded older tokens
        cpu_budget = max_seq_len - gpu_budget
        if use_pinned and torch.cuda.is_available():
            self.k_cpu = torch.zeros(
                num_kv_heads, cpu_budget, head_dim, dtype=dtype,
            ).pin_memory()
            self.v_cpu = torch.zeros(
                num_kv_heads, cpu_budget, head_dim, dtype=dtype,
            ).pin_memory()
        else:
            self.k_cpu = torch.zeros(num_kv_heads, cpu_budget, head_dim, dtype=dtype)
            self.v_cpu = torch.zeros(num_kv_heads, cpu_budget, head_dim, dtype=dtype)

        self.cpu_pos = 0  # tokens currently on CPU

        # Async transfer stream
        if torch.cuda.is_available():
            self.transfer_stream = torch.cuda.Stream()
        else:
            self.transfer_stream = None

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor):
        """
        Append new tokens. If GPU buffer is full, offload oldest to CPU.

        Args:
            k_new: (nKV, new_tokens, hD) on GPU
            v_new: (nKV, new_tokens, hD) on GPU
        """
        new_tokens = k_new.shape[1]
        gpu_used = min(self.pos, self.gpu_budget)

        # Check if we need to offload
        if gpu_used + new_tokens > self.gpu_budget:
            # Number of tokens to offload
            offload_count = gpu_used + new_tokens - self.gpu_budget
            offload_count = min(offload_count, gpu_used)

            if offload_count > 0:
                # Copy oldest tokens from GPU to CPU
                k_offload = self.k_gpu[:, :offload_count, :].clone()
                v_offload = self.v_gpu[:, :offload_count, :].clone()

                cpu_end = self.cpu_pos + offload_count
                self.k_cpu[:, self.cpu_pos:cpu_end, :] = k_offload.cpu()
                self.v_cpu[:, self.cpu_pos:cpu_end, :] = v_offload.cpu()
                self.cpu_pos = cpu_end

                # Shift remaining GPU cache left
                remaining = gpu_used - offload_count
                if remaining > 0:
                    self.k_gpu[:, :remaining, :] = self.k_gpu[:, offload_count:gpu_used, :].clone()
                    self.v_gpu[:, :remaining, :] = self.v_gpu[:, offload_count:gpu_used, :].clone()
                gpu_used = remaining

        # Write new tokens to GPU buffer
        self.k_gpu[:, gpu_used:gpu_used + new_tokens, :] = k_new
        self.v_gpu[:, gpu_used:gpu_used + new_tokens, :] = v_new
        self.pos += new_tokens

    def get_full_kv(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Gather full KV context (CPU + GPU) onto GPU.

        Uses async copy from CPU pinned memory for overlap potential.

        Returns:
            (k, v): each (nKV, total_pos, hD) on GPU
        """
        gpu_used = min(self.pos, self.gpu_budget)

        if self.cpu_pos == 0:
            # Everything is on GPU
            return self.k_gpu[:, :gpu_used, :], self.v_gpu[:, :gpu_used, :]

        # Prefetch CPU portion to GPU
        if self.transfer_stream is not None:
            with torch.cuda.stream(self.transfer_stream):
                k_from_cpu = self.k_cpu[:, :self.cpu_pos, :].to(
                    self.device, non_blocking=True
                )
                v_from_cpu = self.v_cpu[:, :self.cpu_pos, :].to(
                    self.device, non_blocking=True
                )
            self.transfer_stream.synchronize()
        else:
            k_from_cpu = self.k_cpu[:, :self.cpu_pos, :].to(self.device)
            v_from_cpu = self.v_cpu[:, :self.cpu_pos, :].to(self.device)

        # Concatenate: CPU (older) + GPU (recent)
        k_full = torch.cat([k_from_cpu, self.k_gpu[:, :gpu_used, :]], dim=1)
        v_full = torch.cat([v_from_cpu, self.v_gpu[:, :gpu_used, :]], dim=1)
        return k_full, v_full

    def stats(self) -> dict:
        return {
            "total_tokens": self.pos,
            "gpu_tokens": min(self.pos, self.gpu_budget),
            "cpu_tokens": self.cpu_pos,
            "gpu_utilization": min(self.pos, self.gpu_budget) / self.gpu_budget,
        }

    def memory_mb(self) -> dict:
        """Memory breakdown."""
        bpe = self.k_gpu.element_size()
        gpu_mb = 2 * self.k_gpu.numel() * bpe / (1024 ** 2)
        cpu_mb = 2 * self.k_cpu.numel() * self.k_cpu.element_size() / (1024 ** 2)
        return {"gpu_mb": gpu_mb, "cpu_mb": cpu_mb, "total_mb": gpu_mb + cpu_mb}


# ---------------------------------------------------------------------------
# Demo: offloading in action
# ---------------------------------------------------------------------------

def demo_offloading():
    """Show how tokens are offloaded from GPU to CPU as sequence grows."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nKV, hD = 4, 64
    gpu_budget = 64
    max_seq = 256

    offloader = KVOffloader(
        nKV, hD, gpu_budget=gpu_budget, max_seq_len=max_seq,
        device=device, use_pinned=(device == "cuda"),
    )

    mem = offloader.memory_mb()
    print(f"\n  KV Offloader: gpu_budget={gpu_budget}, max_seq={max_seq}")
    print(f"  GPU memory: {mem['gpu_mb']:.1f} MB, CPU memory: {mem['cpu_mb']:.1f} MB")

    print(f"\n  {'Step':>6}  {'New':>4}  {'Total':>6}  {'GPU':>5}  {'CPU':>5}  {'GPU Util':>10}")
    print(f"  {'-' * 6}  {'-' * 4}  {'-' * 6}  {'-' * 5}  {'-' * 5}  {'-' * 10}")

    total_added = 0
    for step, n_tokens in enumerate([32, 16, 16, 32, 32, 32, 32, 32]):
        k = torch.randn(nKV, n_tokens, hD, device=device)
        v = torch.randn(nKV, n_tokens, hD, device=device)
        offloader.append(k, v)
        total_added += n_tokens
        s = offloader.stats()
        print(f"  {step:>6}  {n_tokens:>4}  {s['total_tokens']:>6}  "
              f"{s['gpu_tokens']:>5}  {s['cpu_tokens']:>5}  "
              f"{s['gpu_utilization']*100:>9.1f}%")

    # Retrieve full context
    print(f"\n  Retrieving full context...")
    t0 = time.perf_counter()
    k_full, v_full = offloader.get_full_kv()
    if torch.cuda.is_available() and device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    print(f"  Full KV shape: k={tuple(k_full.shape)}, v={tuple(v_full.shape)}")
    print(f"  Retrieval time: {(t1-t0)*1000:.2f} ms")


# ---------------------------------------------------------------------------
# Benchmark: latency with and without offloading
# ---------------------------------------------------------------------------

def benchmark_offload():
    """Measure the cost of CPU→GPU prefetch at various offload ratios."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("\n  Offload benchmark requires CUDA. Showing theoretical analysis.")
        print("  PCIe Gen4 x16: ~25 GB/s → transferring 100 MB takes ~4 ms")
        return

    nKV, hD = 8, 128
    total_tokens = 2048

    print(f"\n  Offload Latency Benchmark")
    print(f"  nKV={nKV}, hD={hD}, total_tokens={total_tokens}")
    print(f"\n  {'GPU Budget':>12}  {'CPU tokens':>12}  {'Prefetch ms':>13}  {'Overhead':>10}")
    print(f"  {'-' * 12}  {'-' * 12}  {'-' * 13}  {'-' * 10}")

    for gpu_budget_frac in [1.0, 0.75, 0.5, 0.25, 0.1]:
        gpu_budget = int(total_tokens * gpu_budget_frac)
        if gpu_budget < 1:
            gpu_budget = 1

        offloader = KVOffloader(
            nKV, hD, gpu_budget=gpu_budget,
            max_seq_len=total_tokens, device=device,
        )

        # Fill the cache
        batch_size = min(256, total_tokens)
        remaining = total_tokens
        while remaining > 0:
            n = min(batch_size, remaining)
            k = torch.randn(nKV, n, hD, device=device)
            v = torch.randn(nKV, n, hD, device=device)
            offloader.append(k, v)
            remaining -= n

        # Benchmark full retrieval
        torch.cuda.synchronize()
        times = []
        for _ in range(10):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            k_full, v_full = offloader.get_full_kv()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        import statistics
        mean_ms = statistics.mean(times)
        s = offloader.stats()
        print(f"  {gpu_budget:>12}  {s['cpu_tokens']:>12}  {mean_ms:>13.2f}  "
              f"{'none' if s['cpu_tokens'] == 0 else f'{mean_ms:.1f}ms':>10}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("  Chapter 3: KV Cache Offloading (CPU ↔ GPU)")
    print("=" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Running on: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")

    print("\n  [1] Offloading demo")
    demo_offloading()

    print("\n  [2] Offload latency benchmark")
    benchmark_offload()

    print("\n  Key takeaways:")
    print("  - Offloading extends effective context beyond GPU memory")
    print("  - Pinned memory + async streams minimize transfer overhead")
    print("  - Trade-off: longer prefetch latency vs fitting more context")
    print("  - Best suited for long sequences where GPU KV budget is tight")
