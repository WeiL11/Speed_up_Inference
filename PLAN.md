# Speed Up Inference — Project Plan

## Overview

A **model-agnostic** educational project demonstrating how to speed up LLM inference **from scratch**. We use **Gemma-3 1B** as the running example, but every technique applies to any transformer-based model. Target platform: **NVIDIA GPU + PyTorch**.

Each chapter has Python scripts (core implementations) + `demo.ipynb` (interactive walkthrough).

---

## Project Structure

```
Speed_up_Inference/
├── README.md
├── requirements.txt
│
├── 01_profiling_benchmarking/         # Chapter 1: Profile before optimizing
│   ├── benchmarking_script.py        #   Forward/backward timing (timeit + cuda.synchronize)
│   ├── nsys_profile.py               #   Nsight Systems NVTX annotations
│   ├── memory_profiling.py           #   GPU memory snapshots & peak tracking
│   └── demo.ipynb
│
├── 02_efficient_attention/            # Chapter 2: Naive → JIT → Triton FlashAttention-2
│   ├── gemma_attention_breakdown.py  #   Gemma GQA + RoPE structure walkthrough
│   ├── naive_attention.py            #   Basic matmul attention (materializes N×N)
│   ├── jit_attention.py              #   torch.jit.script fused attention
│   ├── triton_flash_attention.py     #   FlashAttention-2 tiled Triton kernel
│   ├── benchmark.py
│   └── demo.ipynb
│
├── 03_kv_cache/                       # Chapter 3: RAM breakdown + paged allocation
│   ├── ram_breakdown.py              #   Where GPU memory goes (weights/activations/KV)
│   ├── naive_kv_cache.py             #   Grow-on-every-token (dynamic allocation)
│   ├── static_kv_cache.py            #   Pre-allocated fixed buffer
│   ├── paged_kv_cache.py             #   Block manager (PagedAttention-style)
│   ├── kv_offload.py                 #   CPU ↔ GPU offloading
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
├── 06_hardware_aware_design/          # Chapter 6: Hardware Design (discussion-heavy)
│   ├── memory_hierarchy.py
│   ├── roofline_model.py
│   ├── tensor_core_utilization.py
│   ├── discussion.md
│   └── demo.ipynb
│
├── 07_quantization/                   # Chapter 7: Quantization theory + from scratch
│   ├── naive_quantization.py         #   absmax / zero-point / per-group INT8 & INT4
│   ├── apply_quantization.py         #   Apply to model layers, measure quality loss
│   ├── bitsandbytes_quant.py         #   Survey: bitsandbytes, GPTQ, AWQ, GGUF
│   ├── benchmark.py
│   └── demo.ipynb
│
├── utils/                             # Shared utilities (model-agnostic)
│   ├── __init__.py
│   ├── model_loader.py               #   Custom transformer + HuggingFace Gemma loader
│   ├── benchmarking.py               #   Timer, memory tracker, throughput calc
│   └── visualization.py
│
└── benchmarks/
    ├── end_to_end.py                  #   All optimizations combined — total speedup
    └── results/
```

---

## Chapter Breakdown

### Chapter 1 — Profiling & Benchmarking (✅ done)

**Goal**: Profile first, optimize second.

Complete hyperparameter set for `benchmarking_script.py`:

| Category | Parameter | CLI flag | Default | Notes |
|----------|-----------|----------|---------|-------|
| Model | num_layers | `--num_layers` | 6 | Transformer layers |
| | hidden_dim | `--hidden_dim` | 1024 | Embedding dimension |
| | num_heads | `--num_heads` | 8 | Attention heads |
| | num_kv_heads | `--num_kv_heads` | 8 | KV heads (=num_heads → MHA; < → GQA) |
| | intermediate_dim | `--intermediate_dim` | auto | MLP width (default 8/3×hidden, rounded to 64) |
| | vocab_size | `--vocab_size` | 32000 | |
| | max_seq_len | `--max_seq_len` | 2048 | |
| Data | batch_size | `--batch_size` | 8 | |
| | seq_len | `--seq_len` | 512 | |
| Precision | dtype | `--dtype` | float32 | float32 / float16 / bfloat16 |
| Timing | warmup | `--warmup` | 5 | Untimed steps (fill CUDA caches, JIT) |
| | steps | `--steps` | 20 | Measured steps |
| | mode | `--mode` | forward | forward / backward / train |
| Optimizer | lr | `--lr` | 1e-4 | Used in train mode |
| | optimizer | `--optimizer` | adamw | adamw / sgd |
| Device | device | `--device` | cuda | cuda / cpu |
| Source | model_name | `--model_name` | custom | "custom" or HuggingFace ID |

---

### Chapter 2 — Efficient Attention & Kernel Optimization

**Goal**: Break down Gemma's attention, then rewrite it three ways — basic PyTorch → JIT → Triton FlashAttention-2.

1. **gemma_attention_breakdown.py** — Dissect Gemma's actual attention (GQA, RoPE, head dims, tensor shapes at each step)
2. **naive_attention.py** — Explicit `Q @ K.T → softmax → @ V`; materializes full N×N matrix; O(N²) memory
3. **jit_attention.py** — `torch.jit.script` version; show what the JIT fuses and measure speedup
4. **triton_flash_attention.py** — FlashAttention-2 [Dao 2023] in Triton from scratch:
   - Tile Q/K/V across sequence dimension (never materialize full N×N)
   - Online softmax with running max + running sum
   - Efficient HBM ↔ SRAM access patterns → O(N) memory, significant wall-clock speedup
5. **Benchmark**: Sequence length sweep 512→8192, wall-clock + peak memory for all three

---

### Chapter 3 — KV Cache Management & Paged Allocation

**Goal**: Break down where GPU RAM goes during Gemma inference, then optimize KV memory with paged allocation.

1. **ram_breakdown.py** — Decompose GPU memory: weights, activations, KV cache, optimizer state; show how KV grows with batch×seq_len
2. **naive_kv_cache.py** — Append tensors each step (dynamic, fragmented)
3. **static_kv_cache.py** — Pre-allocate max_seq_len buffer
4. **paged_kv_cache.py** — Block-based allocation (block table: logical positions → physical blocks); enables KV sharing across beams
5. **kv_offload.py** — Swap old blocks to CPU, prefetch on demand
6. **Benchmark**: Max batch size, generation throughput, memory utilization

---

### Chapter 4 — Batching & Scheduling

**Goal**: Maximize GPU utilization by processing multiple requests efficiently.

1. **static_batching.py** — Pad to max length (simple but wasteful)
2. **dynamic_batching.py** — Group by similar length (bucket strategy)
3. **continuous_batching.py** — Add/remove sequences mid-generation (in-flight batching)
4. **scheduler.py** — FCFS + priority-based request scheduling
5. **Benchmark**: Throughput (tokens/sec) at various concurrent request counts

---

### Chapter 5 — Runtime Optimization

**Goal**: Squeeze performance from the PyTorch runtime without changing model logic.

1. **torch_compile_demo.py** — `torch.compile` with default / reduce-overhead / max-autotune modes
2. **cuda_graphs.py** — Capture + replay static computation graphs; eliminate Python overhead
3. **operator_fusion.py** — Manual fusion: RMSNorm + Linear, SiLU + mul (gated MLP)
4. **mixed_precision.py** — FP16 / BF16 / TF32; AMP autocast comparison
5. **Benchmark**: Latency before/after each optimization, cumulative speedup

---

### Chapter 6 — Hardware-Aware Design

**Goal**: Understand *why* the optimizations work — connect GPU hardware to every earlier chapter.

1. **memory_hierarchy.py** — L1/L2/HBM bandwidth numbers; arithmetic intensity for each layer type
2. **roofline_model.py** — Plot ops vs bandwidth; show whether each layer is compute-bound or memory-bound
3. **tensor_core_utilization.py** — Which tensor shapes hit tensor cores; alignment rules; practical implications
4. **discussion.md** — Hardware challenge narrative: why LLM inference is memory-bandwidth-bound for small batch, compute-bound for large batch; how each chapter's technique moves the operating point

---

### Chapter 7 — Quantization

**Goal**: Explain how quantization works, survey current methods, implement from scratch.

1. **naive_quantization.py** — absmax INT8, zero-point INT8, per-group INT4 (pack 2 values/byte); measure MSE/max-abs-error
2. **apply_quantization.py** — Replace Gemma `nn.Linear` with quantized layers; measure latency + memory + perplexity
3. **bitsandbytes_quant.py** — Survey + comparison: bitsandbytes LLM.int8/NF4, GPTQ, AWQ, GGUF
4. **Benchmark**: FP32 / FP16 / INT8-naive / INT4-naive / bitsandbytes / GPTQ

---

## Key Design Decisions

| Decision | Choice |
|----------|--------|
| Base model | Gemma-3 1B (running example; project is model-agnostic) |
| Format | Python scripts + Jupyter notebooks |
| Quantization | Theory first, from-scratch implementation, then library survey |
| Target hardware | NVIDIA GPU + PyTorch (CUDA, Triton, torch.compile) |
| Philosophy | Every technique is model-agnostic; Gemma is the concrete example |
