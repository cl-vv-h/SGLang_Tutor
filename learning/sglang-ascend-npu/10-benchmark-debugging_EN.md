[中文](./10-benchmark-debugging.md) | [English](./10-benchmark-debugging_EN.md)

# 10. Benchmark & Debugging

## 1. Verification Order

> Correctness first → Stability second → Performance last

## 2. Verification Matrix

| Scenario | Key Metrics | Common Issues |
|---|---|---|
| Single-card minimal request | Response success, TTFT | Missing dependencies, backend not `ascend` |
| Long prompt prefill | Prefill latency, HBM peak | `chunked_prefill_size` wrong, KV cache insufficient |
| Multi-batch decode | Tokens/s, latency jitter | Graph capture shape mismatch, batch size > graph config |
| TP multi-card | HCCL init, rank logs | Communication env, NIC, rank/device mapping |
| PD separation | KV transfer success rate | `memfabric_hybrid`, store URL, transfer protocol |
| LoRA | Adapter loading, output correctness | Ascend LoRA backend, segment info, rank limits |

## 3. Debugging Sequence

1. Verify `torch_npu` and NPU device visibility
2. Verify SGLang logs show NPU recognition
3. Verify `attention_backend`, `prefill_attention_backend`, `decode_attention_backend` are all `ascend`
4. Single-card working → enable TP, HiCache, LoRA, PD one at a time
5. Performance abnormal → check graph capture/replay, fallbacks, format casts, memory copies

## 4. Common Log Patterns

```text
# Good: NPU recognized
[INFO] device='npu', attention_backend='ascend'
[INFO] NPUGraph captured for bs=1,2,4,8

# Bad: Fallback to eager
[WARN] Graph capture failed for bs=16, falling back to eager

# Bad: Format conversion overhead
[DEBUG] Format cast: NCHW → FRACTAL_NZ (should be pre-converted)
```

## 5. Quick Health Check

```bash
# Service running?
curl http://localhost:8000/health

# NPU visible?
python -c "import torch_npu; print(torch_npu.npu.device_count())"

# Memory OK?
npu-smi info
```

## 6. Performance Profiling (Preview)

See `12-npu-profiling-guide.md` for detailed profiling workflow with `torch_npu.profiler`.
