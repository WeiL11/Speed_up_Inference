# Chapter 6 — Hardware-Aware Design: Discussion

## Why LLM Inference is Memory-Bandwidth-Bound

### The Central Challenge

LLM inference at typical serving conditions (batch size 1-8) is overwhelmingly **memory-bandwidth-bound**, not compute-bound. This is the single most important insight for understanding why the optimizations in this project work.

### The Numbers

Consider an A100 GPU:
- **Peak FP16 compute**: 312 TFLOPS
- **HBM bandwidth**: 2.0 TB/s
- **Ridge point**: 312 / 2.0 = **156 FLOPs/byte**

For an operation to be compute-bound, it must perform at least **156 floating-point operations per byte loaded from memory**. Most LLM inference operations fall far below this threshold.

### Decode Phase: The Worst Case

During autoregressive decoding (generating one token at a time):

| Operation | Arithmetic Intensity | Bound |
|-----------|---------------------|-------|
| Linear (batch=1) | ~2 FLOPs/byte | MEMORY |
| Attention QK^T (batch=1) | ~4 FLOPs/byte | MEMORY |
| Softmax | 2.5 FLOPs/byte | MEMORY |
| RMSNorm | 2.0 FLOPs/byte | MEMORY |

At batch=1, **every single operation** is memory-bound. The GPU's 312 TFLOPS of compute is almost entirely wasted — we're limited by how fast we can read model weights from HBM.

### Prefill Phase: Better, But Still Mixed

During prefill (processing the entire prompt at once with long sequence lengths):
- Large matrix multiplications become compute-bound at reasonable batch sizes
- Attention score computation (O(N²)) becomes compute-bound for long sequences
- But elementwise ops (softmax, norm) remain memory-bound regardless

## How Each Chapter's Technique Addresses This

### Chapter 1: Profiling & Benchmarking
Before optimizing, we must **measure** to understand where time is spent. The profiling tools in Chapter 1 reveal that most wall-clock time is spent on memory-bound operations, guiding our optimization priorities.

### Chapter 2: Efficient Attention (FlashAttention)
**Problem**: Standard attention reads Q, K, V from HBM, computes the N×N attention matrix, writes it back to HBM, reads it again for the V multiplication, and writes the output back.

**Solution**: FlashAttention tiles the computation so Q/K/V blocks fit in L1/SRAM. The full N×N matrix is **never materialized in HBM**. This reduces HBM reads/writes from O(N²) to O(N), directly attacking the memory bandwidth bottleneck.

### Chapter 3: KV Cache Management
**Problem**: During decoding, KV cache must be read from HBM at every step. As sequences get longer, KV cache grows and consumes both memory capacity and bandwidth.

**Solution**: Paged allocation reduces memory waste (more sequences fit in GPU memory). Static allocation avoids O(N²) reallocation cost. Offloading to CPU extends effective capacity.

### Chapter 4: Batching & Scheduling
**Problem**: At batch=1, the GPU achieves <2% of peak FLOPS because every operation is memory-bound (loading weights dominates, regardless of how few tokens are processed).

**Solution**: Larger batches increase arithmetic intensity — the same weight load serves multiple sequences. Continuous batching keeps the batch full at all times, maximizing GPU utilization.

### Chapter 5: Runtime Optimization
**Operator fusion** reduces the number of HBM round-trips. Instead of: read → norm → write → read → linear → write, fusion gives us: read → norm+linear → write (one read, one write instead of three).

**CUDA graphs** eliminate Python and CUDA runtime overhead, reducing the constant-factor gap between kernel launches.

**Mixed precision** (FP16/BF16) halves the bytes per element, effectively doubling bandwidth utilization for memory-bound ops.

### Chapter 7: Quantization
**The most impactful optimization for memory-bound inference.** INT8 quantization reduces bytes per parameter by 4x (FP32→INT8) or 2x (FP16→INT8). For memory-bound operations, this translates almost directly to a proportional speedup:

- FP16 weight load: 2 bytes × parameters
- INT8 weight load: 1 byte × parameters → **~2x faster** for memory-bound ops
- INT4 weight load: 0.5 bytes × parameters → **~4x faster** for memory-bound ops

## The Compute-Bound Crossover

As batch size increases, operations transition from memory-bound to compute-bound:

```
Batch=1:   [====MEMORY BOUND====] (2 FLOPs/byte for linear)
Batch=8:   [====MEMORY====|==COMPUTE==] (~16 FLOPs/byte)
Batch=32:  [==MEM==|======COMPUTE========] (~64 FLOPs/byte)
Batch=128: [=M=|===========COMPUTE==============] (~256 FLOPs/byte)
```

**Implication**: Different optimizations matter at different operating points:
- **Small batch** (inference serving): quantization, fusion, FlashAttention
- **Large batch** (throughput mode): tensor core utilization, compute precision
- **Both**: continuous batching, efficient memory management

## Hardware Evolution

The memory-bandwidth gap is widening:
- A100: 312 TFLOPS / 2 TB/s = **156 FLOPs/byte** ridge
- H100: 990 TFLOPS / 3.35 TB/s = **296 FLOPs/byte** ridge

Newer GPUs have proportionally more compute than bandwidth, making the memory-bound problem **worse**. This is why techniques like quantization and FlashAttention become more important, not less, on newer hardware.

## Practical Takeaways

1. **Always profile first** — don't guess whether you're compute or memory bound
2. **Quantize aggressively** for serving — INT8 or INT4 gives near-linear speedup for memory-bound ops
3. **Maximize batch size** — larger batches improve GPU utilization
4. **Fuse operations** — fewer HBM round-trips = faster for memory-bound ops
5. **Use FlashAttention** — O(N) memory instead of O(N²) is always better
6. **Align tensor dimensions** — multiples of 128 for tensor core utilization
7. **Use BF16 minimum** — FP32 wastes both memory and bandwidth
