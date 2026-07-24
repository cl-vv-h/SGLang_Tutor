[中文](./README.md) | [English](./README_EN.md)

# AI Infra Fundamentals

This directory supplements the foundational knowledge needed before and during SGLang source code reading. It doesn't replicate production-grade implementations directly, but instead breaks down the most common mechanisms in LLM serving through lecture notes and small Python demos: model architecture, inference flow, scheduling, KV Cache, attention kernels, execution graphs, Mamba/SSM, parallelism, KV transfer, speculative decoding, quantization, LoRA, and benchmark/profiling.

## Directory Structure

| Order | Directory | Content | Read First |
|---|---|---|---|
| 1 | [Model_Architecture](./Model_Architecture/) | Decoder-only Transformer, MHA/GQA/MLA, Sparse MoE, SSM Hybrid and mainstream model architecture families | [README.md](./Model_Architecture/README.md) |
| 2 | [Inference_Basics](./Inference_Basics/) | Transformer inference, Prefill/Decode, throughput and latency fundamentals | [README.md](./Inference_Basics/README.md) |
| 3 | [Schedule_Optimization](./Schedule_Optimization/) | Continuous batching, Chunked Prefill, scheduling trade-offs | [README.md](./Schedule_Optimization/README.md) |
| 4 | [KV_Cache_Memory](./KV_Cache_Memory/) | KV Cache layout, paging, prefix cache, memory estimation | [README.md](./KV_Cache_Memory/README.md) |
| 5 | [Attention_Kernel](./Attention_Kernel/) | Educational implementations of FlashAttention and FlashDecoding | [README.md](./Attention_Kernel/README.md) |
| 6 | [Execution_Graph](./Execution_Graph/) | From computation graph concepts to CUDA/NPU Graph, torch.compile, static shape reuse, and replay dataflow | [01-what-is-graph.md](./Execution_Graph/01-what-is-graph.md), [02-graph-execution-dataflow.md](./Execution_Graph/02-graph-execution-dataflow.md) |
| 7 | [Mamba_State_Space](./Mamba_State_Space/) | Mamba/SSM principles, Mamba state, MambaPool, scheduler strategy, and radix cache states | [01-mamba-and-sglang-state.md](./Mamba_State_Space/01-mamba-and-sglang-state.md) |
| 8 | [Parallel_Strategy](./Parallel_Strategy/) | DP, TP, PP, SP/CP, EP inference parallelism strategies | [README.md](./Parallel_Strategy/README.md) |
| 9 | [KV_Transfer](./KV_Transfer/) | PD disaggregation, KV sender/receiver, remote KV cache | [README.md](./KV_Transfer/README.md) |
| 10 | [Speculative_Decoding](./Speculative_Decoding/) | Speculative sampling math, target verify, KV commit, EAGLE/MTP/NGRAM/Medusa/REST algorithm landscape | [README.md](./Speculative_Decoding/README.md) |
| 11 | [Quantization](./Quantization/) | Weight-only, W8A8/FP8, KV quant, calibration and error | [README.md](./Quantization/README.md) |
| 12 | [LoRA](./LoRA/) | LoRA, QLoRA, DoRA, AdaLoRA and multi-LoRA serving | [README.md](./LoRA/README.md) |
| 13 | [Benchmark_Profiling](./Benchmark_Profiling/) | TTFT/ITL/TPS, load testing, profiling, bottleneck identification | [README.md](./Benchmark_Profiling/README.md) |

## Suggested Learning Path

1. Start with [Model_Architecture](./Model_Architecture/) and [Inference_Basics](./Inference_Basics/) to establish the basic model of architecture, tensor shapes, and prefill/decode.
2. Then read [Schedule_Optimization](./Schedule_Optimization/), [KV_Cache_Memory](./KV_Cache_Memory/), and [Attention_Kernel](./Attention_Kernel/) to understand batching, memory, KV access, and attention backend constraints.
3. Next, read [Execution_Graph](./Execution_Graph/), [Mamba_State_Space](./Mamba_State_Space/), and [Parallel_Strategy](./Parallel_Strategy/) to understand how production inference reduces CPU overhead, manages non-Transformer states, and scales to multiple GPUs.
4. Then read [KV_Transfer](./KV_Transfer/) and [Speculative_Decoding](./Speculative_Decoding/) to understand how advanced serving optimizations revolve around "getting tokens faster" and "better utilizing different resources."
5. Finally, read [Quantization](./Quantization/), [LoRA](./LoRA/), and [Benchmark_Profiling](./Benchmark_Profiling/) to close the loop on model compression, adapter serving, and performance validation.

## Relationship to SGLang Source Code Reading

- `Inference_Basics` maps to request lifecycle, forward mode, sampling, and token generation loop.
- `Schedule_Optimization` maps to Scheduler, waiting/running queue, continuous batching, chunked prefill.
- `KV_Cache_Memory` maps to KV cache manager, memory pool, RadixAttention, prefix cache, HiCache.
- `Attention_Kernel` maps to attention backend, prefill attention, decode attention, KV block read/write.
- `Execution_Graph` maps to CUDA graph, static batch, graph capture/replay, and shape padding.
- `Mamba_State_Space` maps to Mamba/SSM layers, Mamba state pool, mamba scheduler strategy, Mamba radix cache, and state transfer.
- `Parallel_Strategy` maps to TP/PP/DP/EP rank organization, communication patterns, and multi-process execution.
- `KV_Transfer` maps to PD disaggregation, bootstrap, prealloc, KV sender/receiver, and transfer backend.
- `Speculative_Decoding` maps to draft worker, target verify, `spec_info`, accept token, and grammar/sampling post-processing.
- `Quantization` maps to weight loader, quant method, kernel selection, FP8/W8A8/GPTQ/AWQ execution paths.
- `LoRA` maps to adapter registration, hot-loading, batch constraints, LoRA memory pool, and LoRA kernel.
- `Benchmark_Profiling` maps to benchmark scripts, metrics, trace, CUDA/NVTX profiling, and online parameter tuning.

## Running Demos

Existing runnable demos primarily depend on Python and PyTorch. It is recommended to run from the repository root for easier cross-referencing with source code reading notes:

```bash
# Example: run a demo from repo root
python learning/ai-infra-basic/Parallel_Strategy/tp_inference_demo.py
```
