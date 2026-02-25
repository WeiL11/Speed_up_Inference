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
├── 01_quantization/                   # Chapter 1: Model Compression via Quantization
│   ├── naive_quantization.py          #   Hand-rolled absmax & zero-point INT8/INT4
│   ├── apply_quantization.py          #   Apply naive quant to Gemma layers
│   ├── bitsandbytes_quant.py          #   Compare with bitsandbytes INT8/NF4
│   ├── benchmark.py                   #   Latency / memory / perplexity comparison
│   └── demo.ipynb                     #   Interactive walkthrough
│
├── 02_efficient_attention/            # Chapter 2: Efficient Attention & Kernel Optimization
│   ├── standard_attention.py          #   Naive scaled-dot-product attention
│   ├── flash_attention.py             #   Flash Attention 2 via torch SDPA
│   ├── triton_fused_attention.py      #   Custom Triton kernel for fused attention
│   ├── benchmark.py                   #   Wall-clock + memory comparison
│   └── demo.ipynb
│
├── 03_kv_cache/                       # Chapter 3: KV Cache Management & Offloading
│   ├── naive_kv_cache.py             #   Basic grow-on-every-token cache
│   ├── static_kv_cache.py            #   Pre-allocated fixed-size cache
│   ├── paged_kv_cache.py             #   PagedAttention-style block manager
│   ├── kv_offload.py                 #   CPU ↔ GPU offloading
│   ├── benchmark.py
│   └── demo.ipynb
│
├── 04_batching_scheduling/            # Chapter 4: Batching & Scheduling
│   ├── static_batching.py            #   Pad-to-max naive batching
│   ├── dynamic_batching.py           #   Group by similar length
│   ├── continuous_batching.py        #   In-flight batching (vLLM-style)
│   ├── scheduler.py                  #   Simple FCFS + priority scheduler
│   ├── benchmark.py
│   └── demo.ipynb
│
├── 05_runtime_optimization/           # Chapter 5: Runtime Optimization
│   ├── torch_compile_demo.py         #   torch.compile modes & backends
│   ├── cuda_graphs.py                #   CUDA graph capture & replay
│   ├── operator_fusion.py            #   Manual op fusion examples
│   ├── mixed_precision.py            #   FP16 / BF16 / TF32 settings
│   ├── benchmark.py
│   └── demo.ipynb
│
├── 06_hardware_aware_design/          # Chapter 6: Hardware-Aware Design
│   ├── memory_hierarchy.py           #   GPU memory bandwidth analysis
│   ├── roofline_model.py             #   Arithmetic intensity & roofline plot
│   ├── tensor_core_utilization.py    #   Shapes that hit tensor cores
│   ├── profiling_guide.py            #   torch.profiler + Nsight walkthrough
│   ├── benchmark.py
│   └── demo.ipynb
│
├── utils/                             # Shared utilities
│   ├── __init__.py
│   ├── model_loader.py               #   Load Gemma-3 1B, common config
│   ├── benchmarking.py               #   Timer, memory tracker, throughput calc
│   └── visualization.py              #   Plotting helpers (matplotlib)
│
└── benchmarks/
    ├── end_to_end.py                  #   Combine all optimizations, measure total speedup
    └── results/                       #   Saved benchmark results & plots
```

---

## Chapter Breakdown

### Chapter 1 — Quantization (from scratch)

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

### Chapter 2 — Efficient Attention & Kernel Optimization

**Goal**: Show why attention is the bottleneck and how to fix it with better algorithms + custom kernels.

1. **Standard attention** — naive O(n²) implementation, measure memory
2. **Flash Attention** — explain the tiling algorithm, use `torch.nn.functional.scaled_dot_product_attention` with Flash backend
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

## Implementation Order

1. `utils/` — model loader, benchmarking, visualization (shared foundation)
2. Chapter 1 — Quantization (most self-contained, good starting point)
3. Chapter 5 — Runtime Optimization (quick wins, useful baseline)
4. Chapter 2 — Efficient Attention
5. Chapter 3 — KV Cache
6. Chapter 4 — Batching & Scheduling
7. Chapter 6 — Hardware-Aware Design (ties everything together)
8. `benchmarks/end_to_end.py` — combine all optimizations

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
