"""
Chapter 3 — KV Cache Management
File: paged_kv_cache.py

PagedAttention-style block-based KV cache allocation.

Inspired by vLLM's PagedAttention [Kwon et al. 2023]:
  - KV memory is divided into fixed-size blocks (like OS virtual memory pages)
  - A block table maps logical positions to physical block indices
  - Blocks are allocated on demand from a free pool
  - Enables: (1) zero internal fragmentation, (2) KV sharing across beams,
    (3) dynamic memory management across concurrent requests

Contents:
  - KVBlock        : single block holding block_size tokens of K/V
  - BlockManager   : allocates/frees blocks from a pool
  - PagedKVCache   : per-sequence paged cache with block table
  - demo           : show allocation, utilization, and sharing
"""

import sys
import time
import math
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch


# ---------------------------------------------------------------------------
# Block Manager (pool allocator)
# ---------------------------------------------------------------------------

class BlockManager:
    """
    Manages a pool of physical KV blocks.

    The total GPU memory budget is divided into num_blocks blocks,
    each holding block_size tokens of K and V for one layer+head group.

    Args:
        num_blocks: total physical blocks in the pool
        block_size: tokens per block
        num_kv_heads: number of KV heads
        head_dim: dimension per head
        device: torch device
        dtype: tensor dtype
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        self.num_blocks   = num_blocks
        self.block_size   = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim     = head_dim
        self.device       = device
        self.dtype        = dtype

        # Pre-allocate all physical blocks
        # Shape: (num_blocks, num_kv_heads, block_size, head_dim)
        self.k_pool = torch.zeros(
            num_blocks, num_kv_heads, block_size, head_dim,
            device=device, dtype=dtype,
        )
        self.v_pool = torch.zeros(
            num_blocks, num_kv_heads, block_size, head_dim,
            device=device, dtype=dtype,
        )

        # Free list: stack of available block indices
        self.free_blocks = list(range(num_blocks))

    def allocate(self) -> int:
        """Allocate one block from the free pool. Returns physical block index."""
        if not self.free_blocks:
            raise RuntimeError("Block pool exhausted — no free blocks available")
        return self.free_blocks.pop()

    def free(self, block_idx: int):
        """Return a block to the free pool."""
        assert 0 <= block_idx < self.num_blocks
        self.free_blocks.append(block_idx)

    @property
    def num_free(self) -> int:
        return len(self.free_blocks)

    @property
    def num_used(self) -> int:
        return self.num_blocks - self.num_free

    def utilization(self) -> float:
        return self.num_used / self.num_blocks if self.num_blocks > 0 else 0.0

    def memory_mb(self) -> float:
        """Total pool memory in MB."""
        bpe = self.k_pool.element_size()
        return 2 * self.k_pool.numel() * bpe / (1024 ** 2)

    def get_block_k(self, block_idx: int) -> torch.Tensor:
        """Get K data for a physical block. Returns (nKV, block_size, hD) view."""
        return self.k_pool[block_idx]

    def get_block_v(self, block_idx: int) -> torch.Tensor:
        return self.v_pool[block_idx]


# ---------------------------------------------------------------------------
# Paged KV Cache (per-sequence)
# ---------------------------------------------------------------------------

class PagedKVCache:
    """
    Paged KV cache for a single sequence.

    Maintains a block table (list of physical block indices) that maps
    logical positions to physical blocks in the BlockManager's pool.

    Args:
        block_manager: shared block pool
        batch_idx: which batch element this cache belongs to
    """

    def __init__(self, block_manager: BlockManager, batch_idx: int = 0):
        self.manager    = block_manager
        self.batch_idx  = batch_idx
        self.block_table: list[int] = []  # logical block idx → physical block idx
        self.pos = 0  # number of tokens stored

    def _ensure_blocks(self, needed_pos: int):
        """Allocate blocks to cover up to needed_pos tokens."""
        bs = self.manager.block_size
        needed_blocks = math.ceil(needed_pos / bs)
        while len(self.block_table) < needed_blocks:
            self.block_table.append(self.manager.allocate())

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor):
        """
        Append new K/V tokens to the paged cache.

        Args:
            k_new: (nKV, new_tokens, hD)
            v_new: (nKV, new_tokens, hD)
        """
        new_tokens = k_new.shape[1]
        end_pos = self.pos + new_tokens
        self._ensure_blocks(end_pos)

        bs = self.manager.block_size

        for t in range(new_tokens):
            logical_pos = self.pos + t
            block_idx = logical_pos // bs
            offset = logical_pos % bs
            phys_block = self.block_table[block_idx]

            self.manager.k_pool[phys_block, :, offset, :] = k_new[:, t, :]
            self.manager.v_pool[phys_block, :, offset, :] = v_new[:, t, :]

        self.pos = end_pos

    def get_kv(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Gather all cached K/V into contiguous tensors.

        Returns:
            (k, v): each of shape (nKV, pos, hD)
        """
        if self.pos == 0:
            nKV = self.manager.num_kv_heads
            hD = self.manager.head_dim
            dev = self.manager.device
            dt = self.manager.dtype
            return (torch.empty(nKV, 0, hD, device=dev, dtype=dt),
                    torch.empty(nKV, 0, hD, device=dev, dtype=dt))

        bs = self.manager.block_size
        k_parts = []
        v_parts = []

        for block_idx in range(len(self.block_table)):
            phys = self.block_table[block_idx]
            start = block_idx * bs
            end = min(start + bs, self.pos)
            length = end - start
            if length <= 0:
                break
            k_parts.append(self.manager.k_pool[phys, :, :length, :])
            v_parts.append(self.manager.v_pool[phys, :, :length, :])

        return torch.cat(k_parts, dim=1), torch.cat(v_parts, dim=1)

    def free_all(self):
        """Return all blocks to the pool."""
        for phys_block in self.block_table:
            self.manager.free(phys_block)
        self.block_table.clear()
        self.pos = 0

    @property
    def num_blocks_used(self) -> int:
        return len(self.block_table)

    def fork(self) -> "PagedKVCache":
        """
        Create a copy sharing the same physical blocks (copy-on-write style).

        Useful for beam search: multiple beams share prefix KV data.
        Note: This is a simplified version; real systems use reference counting.
        """
        new_cache = PagedKVCache(self.manager, self.batch_idx)
        new_cache.block_table = list(self.block_table)  # share same physical blocks
        new_cache.pos = self.pos
        return new_cache


# ---------------------------------------------------------------------------
# Demo: allocation, utilization, sharing
# ---------------------------------------------------------------------------

def demo_paged_allocation():
    """Show how blocks are allocated on demand."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nKV, hD, block_size = 4, 64, 16
    num_blocks = 32

    manager = BlockManager(num_blocks, block_size, nKV, hD, device)
    print(f"\n  Block Manager: {num_blocks} blocks × {block_size} tokens = {num_blocks * block_size} token capacity")
    print(f"  Pool memory: {manager.memory_mb():.1f} MB")

    cache = PagedKVCache(manager)

    print(f"\n  {'Action':<35}  {'Blocks':>7}  {'Free':>6}  {'Util%':>7}")
    print(f"  {'-' * 35}  {'-' * 7}  {'-' * 6}  {'-' * 7}")
    print(f"  {'Initial':35s}  {cache.num_blocks_used:>7}  {manager.num_free:>6}  {manager.utilization()*100:>6.1f}%")

    # Append tokens in varying batch sizes
    for n_tokens in [10, 6, 20, 12, 16]:
        k = torch.randn(nKV, n_tokens, hD, device=device)
        v = torch.randn(nKV, n_tokens, hD, device=device)
        cache.append(k, v)
        action = f"Append {n_tokens} tokens (pos={cache.pos})"
        print(f"  {action:35s}  {cache.num_blocks_used:>7}  {manager.num_free:>6}  {manager.utilization()*100:>6.1f}%")

    # Free
    cache.free_all()
    print(f"  {'Free all blocks':35s}  {cache.num_blocks_used:>7}  {manager.num_free:>6}  {manager.utilization()*100:>6.1f}%")


def demo_beam_sharing():
    """Show how beams can share prefix blocks."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nKV, hD, block_size = 4, 64, 16
    num_blocks = 64

    manager = BlockManager(num_blocks, block_size, nKV, hD, device)
    prefix_cache = PagedKVCache(manager)

    # Write a shared prefix
    prefix_tokens = 48
    k = torch.randn(nKV, prefix_tokens, hD, device=device)
    v = torch.randn(nKV, prefix_tokens, hD, device=device)
    prefix_cache.append(k, v)
    prefix_blocks = prefix_cache.num_blocks_used

    print(f"\n  Beam Search Sharing Demo")
    print(f"  Prefix: {prefix_tokens} tokens → {prefix_blocks} blocks")

    # Fork into 4 beams
    num_beams = 4
    beams = [prefix_cache.fork() for _ in range(num_beams)]

    print(f"  Forked into {num_beams} beams (sharing {prefix_blocks} physical blocks)")
    print(f"  Without sharing: {num_beams * prefix_blocks} blocks needed")
    print(f"  With sharing:    {prefix_blocks} blocks (shared) + beam-specific blocks")
    print(f"  Memory saving:   {(num_beams - 1) * prefix_blocks} blocks saved")

    # Each beam generates a few unique tokens
    for i, beam in enumerate(beams):
        unique_tokens = 8 + i * 4
        k = torch.randn(nKV, unique_tokens, hD, device=device)
        v = torch.randn(nKV, unique_tokens, hD, device=device)
        beam.append(k, v)
        print(f"  Beam {i}: +{unique_tokens} unique tokens, "
              f"total pos={beam.pos}, blocks={beam.num_blocks_used}")


def demo_fragmentation_comparison():
    """Compare memory fragmentation: static vs paged."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nKV, hD = 4, 64

    # Simulate 4 concurrent requests with different lengths
    request_lengths = [32, 128, 64, 256]
    max_len = max(request_lengths)

    # Static: allocate max_len for each
    static_total = sum(max_len for _ in request_lengths)
    static_used = sum(request_lengths)
    static_waste = static_total - static_used

    # Paged: block_size=16, allocate only what's needed
    block_size = 16
    paged_total = sum(math.ceil(l / block_size) * block_size for l in request_lengths)
    paged_used = sum(request_lengths)
    paged_waste = paged_total - paged_used

    print(f"\n  Fragmentation Comparison (4 concurrent requests)")
    print(f"  Request lengths: {request_lengths}")
    print(f"\n  {'Method':<12}  {'Allocated':>10}  {'Used':>8}  {'Wasted':>8}  {'Efficiency':>12}")
    print(f"  {'-' * 12}  {'-' * 10}  {'-' * 8}  {'-' * 8}  {'-' * 12}")
    print(f"  {'Static':12s}  {static_total:>10}  {static_used:>8}  {static_waste:>8}  {static_used/static_total*100:>11.1f}%")
    print(f"  {'Paged':12s}  {paged_total:>10}  {paged_used:>8}  {paged_waste:>8}  {paged_used/paged_total*100:>11.1f}%")
    print(f"\n  Paged allocation reduces waste by {(static_waste - paged_waste) / static_waste * 100:.0f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("  Chapter 3: Paged KV Cache (PagedAttention-style)")
    print("=" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Running on: {device}")

    print("\n  [1] Paged block allocation")
    demo_paged_allocation()

    print("\n  [2] Beam search KV sharing")
    demo_beam_sharing()

    print("\n  [3] Fragmentation comparison")
    demo_fragmentation_comparison()

    print("\n  Key takeaways:")
    print("  - Blocks allocated on demand → no wasted memory for short sequences")
    print("  - Block table indirection → KV can be shared across beams")
    print("  - Fixed block size → zero external fragmentation")
    print("  - Internal fragmentation limited to < 1 block per sequence")
