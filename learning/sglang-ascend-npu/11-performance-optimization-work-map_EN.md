[中文](./11-performance-optimization-work-map.md) | [English](./11-performance-optimization-work-map_EN.md)

# 11. Performance Optimization Work Map

## Optimization Categories

| Direction | Main Repo | Typical Issues |
|---|---|---|
| Serving-layer scheduling & batching | `sglang` | NPU underutilization, P99 jitter, long prompt blocks decode |
| Model execution & NPU graph | `sglang` | Graph capture/replay misses, shape instability, high memory |
| Attention & KV cache | `sglang` + `sglang-kernel-npu` | Slow prefill/decode attention, KV layout mismatch |
| NPU kernel fusion | `sglang-kernel-npu` | Slow ops, excessive launches, insufficient shape coverage |
| Memory & data movement | `sglang` + `sglang-kernel-npu` | Frequent format casts/copies, abnormal HBM usage |
| TP/HCCL communication | `sglang` | Poor multi-card scaling, high communication wait |
| Feature combinations | Both | Performance degradation with PD+LoRA+MoE+Quant |
| Benchmark/Profiling | `sglang` | Unreproducible results, unclear bottleneck attribution |

## Decision Framework

```text
Performance problem observed
  ├─ Is the bottleneck in the scheduler or model?
  │   ├─ High scheduler CPU time → check batching, queue, policy
  │   └─ High GPU/NPU time → check kernels, graph, communication
  ├─ Is it a single-card or multi-card issue?
  │   ├─ Single-card slow → check attention, KV cache, graphs
  │   └─ Multi-card scaling poor → check HCCL, TP balance, EP load
  └─ Is it a specific feature combination?
      └─ Bisect: test each feature independently
```

## Performance Optimization PR Checklist

A qualified performance PR should deliver:
1. **Benchmark**: Before/after numbers in controlled conditions
2. **Profiling**: Trace showing the bottleneck improvement
3. **Correctness**: Pass accuracy tests, no regressions
4. **Scope**: Clear which scenarios benefit (prefill/decode/both, TP sizes, batch sizes)
