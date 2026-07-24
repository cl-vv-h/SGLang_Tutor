[中文](./README.md) | [English](./README_EN.md)

# Benchmark & Profiling

Performance benchmarking and profiling for LLM serving: TTFT, ITL, TPS metrics, load testing, profiling tools, and bottleneck identification.

## Core Metrics

| Metric | Definition | What It Reveals |
|---|---|---|
| TTFT | Time To First Token | Prefill + scheduling + queuing latency |
| ITL | Inter-Token Latency | Decode step efficiency, KV cache access |
| TPS | Tokens Per Second | Overall throughput |
| QPS | Queries Per Second | System-level throughput |
| P95/P99 | 95th/99th percentile latency | Tail latency, worst-case performance |

## Profiling Approach

1. Establish baseline without profiler overhead
2. Use `torch_npu.profiler` (Ascend) or `torch.profiler` (CUDA) for trace collection
3. Analyze timeline: idle gaps, hot kernels, copy/format cast, HCCL/NCCL waits
4. Design profiling windows for prefill, decode, TP, and combined scenarios
5. Convert profiling conclusions into development tasks

## SGLang Integration

- `ProfilerManager._profile_batch_predicate()` triggers profiling for selected batches
- `forward_ct` counter used for profiling iteration tracking
- Benchmark scripts available for standard workloads
