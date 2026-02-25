# Speed Up Inference — Project Plan

## Overview
An educational, hands-on project demonstrating how to speed up LLM inference **from scratch**, using **Gemma-3 1B** as the base model, with **NVIDIA GPU + PyTorch** as the target platform.

Each chapter has:
- Python scripts (the core implementations)
- A Jupyter notebook (`demo.ipynb`) for interactive walkthrough & visualization

---

## Project Structure

```
Speed_up_Inference/
├── README.md                          # Project overview, setup, table of contents
├── requirements.txt
├── pyproject.toml
│
├── 01_profiling_benchmarking/         # Chapter 1: Profiling & Benchmarking
│   ├── benchmarking_script.py        #   End-to-end forward/backward benchmarking
│   ├── nsys_profile.py               #   Nsight Systems profiling wrapper (NVTX)
│   ├── memory_profiling.py           #   GPU memory snapshots & peak tracking
│   └── demo.ipynb                    #   Interactive walkthrough & analysis
│
├── 02_efficient_attention/            # Chapter 2: Efficient Attention & Kernel Optimization
│   ├── standard_attention.py
│   ├── flash_attention.py
│   ├── triton_fused_attention.py
│   ├── benchmark.py
│   └── demo.ipynb
│
├── 03_kv_cache/                       # Chapter 3: KV Cache Management & Offloading
│   ├── naive_kv_cache.py
│   ├── static_kv_cache.py
│   ├── paged_kv_cache.py
│   ├── kv_offload.py
│   ├── benchmark.py
│   └── demo.ipynb
│
├── 04_batching_scheduling/            # Chapter 4: Batching & Scheduling
│   ├── static_batching.py
│   ├── dynamic_batching.py
│   ├── continuous_batching.py
│   ├── scheduler.py
│   ├── benchmark.py
│   └── demo.ipynb
│
├── 05_runtime_optimization/           # Chapter 5: Runtime Optimization
│   ├── torch_compile_demo.py
│   ├── cuda_graphs.py
│   ├── operator_fusion.py
│   ├── mixed_precision.py
│   ├── benchmark.py
│   └── demo.ipynb
│
├── 06_hardware_aware_design/          # Chapter 6: Hardware-Aware Design
│   ├── memory_hierarchy.py
│   ├── roofline_model.py
│   ├── tensor_core_utilization.py
│   ├── benchmark.py
│   └── demo.ipynb
│
├── 07_quantization/                   # Chapter 7: Model Compression via Quantization
│   ├── naive_quantization.py
│   ├── apply_quantization.py
│   ├── bitsandbytes_quant.py
│   ├── benchmark.py
│   └── demo.ipynb
│
├── utils/                             # Shared utilities
│   ├── __init__.py
│   ├── model_loader.py
│   ├── benchmarking.py
│   └── visualization.py
│
└── benchmarks/
    ├── end_to_end.py
    └── results/
```

---

## Chapter Breakdown

### Chapter 1 — Profiling & Benchmarking

**Goal**: Before optimizing anything, profile the model to understand where time and memory are spent.

1. **benchmarking_script.py** — End-to-end benchmarking
   - Initialize a transformer model given hyperparameters (num_layers, hidden_dim, etc.)
   - Generate a random batch of data
   - Run `w` warm-up steps (not timed), then time `n` steps using `timeit.default_timer()`
   - Support forward-only and forward+backward modes via CLI argument
   - Call `torch.cuda.synchronize()` after each step for accurate GPU timing
   - Report mean, std, min, max latency + throughput + GPU memory usage

2. **nsys_profile.py** — Nsight Systems profiling wrapper
   - Wrap model execution with NVTX range annotations (forward pass, backward pass, optimizer step)
   - Annotate individual layers: self-attention, MLP, LayerNorm, etc.
   - Support profiling modes: forward-only, forward+backward, full training step (with AdamW)
   - Generate `nsys`-compatible output for analysis in Nsight Systems GUI
   - Designed to answer these key questions:
     a. Does total forward pass time match Python-level measurements?
     b. Which CUDA kernel takes the most cumulative GPU time? Same for fwd+bwd?
     c. What non-matmul kernels account for non-trivial runtime?
     d. How does matmul fraction change between inference vs full training step?
     e. How does softmax runtime compare to matmul runtime in self-attention?

3. **memory_profiling.py** — GPU memory profiling
   - Record memory timelines via `torch.cuda.memory._record_memory_history()`
   - Export snapshots for pytorch.org/memory_viz (Active Memory Timeline)
   - Profile across context lengths (128, 256, 512) for forward / fwd+bwd / train
   - Support mixed-precision (FP16 via torch.amp)
   - Calculate theoretical activation tensor sizes
   - Designed to answer:
     a. Memory timeline shape: can you identify forward/backward/optimizer stages from peaks?
     b. Peak memory table by context length (forward vs. full training step)
     c. Mixed-precision impact on peak memory
     d. Theoretical size of a residual-stream activation tensor
     e. What allocations remain visible at reduced detail levels in memory_viz?

---

### Chapter 2 — Efficient Attention & Kernel Optimization

**Goal**: Show why attention is the bottleneck and how to fix it with better algorithms + custom kernels.

1. **Standard attention** — naive O(n²) implementation, measure memory
2. **Flash Attention** — explain the tiling algorithm, use `torch.nn.functional.scaled_dot_product_attention`
3. **Custom Triton kernel** — write a fused attention kernel in Triton from scratch
4. **Benchmark**: Sequence length sweep (512 → 4096), wall-clock & peak memory

---

### Chapter 3 — KV Cache Management & Offloading

**Goal**: Show how KV caching works in autoregressive generation and optimize memory usage.

1. **Naive cache** — append KV tensors each step (dynamic allocation)
2. **Static pre-allocated cache** — fixed buffer, pointer management
3. **Paged KV cache** — block-based allocation inspired by PagedAttention
4. **CPU offloading** — swap old KV blocks to CPU, prefetch on demand
5. **Benchmark**: Max batch size at fixed sequence length, generation throughput

---

### Chapter 4 — Batching & Scheduling

**Goal**: Maximize GPU utilization by processing multiple requests efficiently.

1. **Static batching** — pad all sequences to max length
2. **Dynamic batching** — group by similar length, bucket strategy
3. **Continuous batching** — add/remove sequences mid-generation
4. **Scheduler** — FCFS + priority-based request scheduling
5. **Benchmark**: Throughput (tokens/sec) at various concurrent request counts

---

### Chapter 5 — Runtime Optimization

**Goal**: Squeeze performance from the PyTorch runtime without changing model logic.

1. **torch.compile** — default, reduce-overhead, max-autotune modes
2. **CUDA Graphs** — capture and replay static computation graphs
3. **Operator fusion** — manual fusion examples (LayerNorm + Linear)
4. **Mixed precision** — FP16, BF16, TF32 comparison
5. **Benchmark**: Latency breakdown before/after each optimization

---

### Chapter 6 — Hardware-Aware Design

**Goal**: Understand *why* these optimizations work by analyzing hardware characteristics.

1. **Memory hierarchy** — L1/L2/HBM bandwidth, how data moves
2. **Roofline model** — plot ops vs memory bandwidth, identify if compute or memory bound
3. **Tensor cores** — which shapes trigger tensor core paths, alignment rules
4. **Profiling** — `torch.profiler` traces, reading Nsight Systems timelines
5. **Benchmark**: Before/after roofline position for each optimization from prior chapters

---

### Chapter 7 — Quantization (from scratch)

**Goal**: Understand what quantization does at the math level, then apply it to Gemma-3 1B.

1. **Naive implementation**
   - Absmax symmetric quantization (FP32 → INT8)
   - Zero-point asymmetric quantization (FP32 → INT8)
   - Per-tensor vs per-channel vs per-group granularity
   - Extend to INT4 (pack two values per byte)
   - Dequantize and measure error (MSE, max-abs-error)

2. **Apply to Gemma**
   - Replace `nn.Linear` weights with quantized versions
   - Run inference, measure latency + memory + perplexity on WikiText-2

3. **Library comparison**
   - `bitsandbytes` INT8 (LLM.int8()) and NF4
   - Brief note on GPTQ / AWQ (not from scratch, but show usage)

4. **Benchmark**: Table comparing FP32 / FP16 / INT8-naive / INT4-naive / bitsandbytes

---

## Implementation Order

1. `utils/` — model loader, benchmarking, visualization (shared foundation)
2. Chapter 1 — Profiling & Benchmarking (understand the baseline first)
3. Chapter 2 — Efficient Attention
4. Chapter 3 — KV Cache
5. Chapter 4 — Batching & Scheduling
6. Chapter 5 — Runtime Optimization
7. Chapter 6 — Hardware-Aware Design
8. Chapter 7 — Quantization
9. `benchmarks/end_to_end.py` — combine all optimizations

---

## Dependencies

```
torch >= 2.2
transformers >= 4.40
bitsandbytes
triton
matplotlib
numpy
datasets  (for WikiText-2 perplexity eval)
jupyter
tqdm
```

---

## Key Design Decisions (already made)

| Decision | Choice |
|----------|--------|
| Base model | Gemma-3 1B (small, fast iteration) |
| Format | Python scripts + Jupyter notebooks |
| Quantization | From-scratch first, then library comparison |
| Target hardware | NVIDIA GPU + PyTorch (CUDA, Triton, torch.compile) |
