[中文](./09-lora-moe-feature-branches.md) | [English](./09-lora-moe-feature-branches_EN.md)

# 09. LoRA, MoE & Feature Branches on Ascend

## 1. LoRA on Ascend NPU

### Backend

`python/sglang/srt/lora/backend/ascend_backend.py`

### Key Kernels

| Kernel | Operation | Purpose |
|---|---|---|
| `torch.ops.npu.sgmv_shrink` | `x @ A^T` (segmented) | LoRA down-projection per adapter |
| `torch.ops.npu.sgmv_expand` | `delta @ B^T` (segmented) | LoRA up-projection per adapter |

### LoRABatchInfo

```python
LoRABatchInfo:
    weight_indices: [B]        # Adapter ID per request
    seg_lens: [num_adapters]   # Tokens per adapter in batch
    scalings: [num_adapters]   # Scale factor per adapter
```

Batched LoRA forward:
```text
For each unique adapter in batch:
    affected = tokens using this adapter
    delta = sgmv_shrink(X[affected], A_adapter)
    delta = sgmv_expand(delta, B_adapter)
    Y[affected] += delta * scaling
```

## 2. MoE on Ascend NPU

### Stream Management

NPU utility manages separate streams for shared and routed experts:

```python
# In hardware_backend/npu/utils.py
process_shared_expert(...)   # Shared expert on dedicated stream
process_routed_expert(...)   # Routed experts on separate stream
```

### DeepEP Integration

`sgl-kernel-npu` provides DeepEP-based MoE communication:
- `deep_ep::Buffer` for dispatch/combine
- `low_latency_dispatch` / `low_latency_combine` for small-batch inference
- A2 layered path combining intra-node HCCS with inter-node RDMA

## 3. Feature Support Matrix

| Feature | Ascend Support | Implementation |
|---|---|---|
| LoRA | ✅ Full | Ascend LoRA backend via `torch.ops.npu` |
| Multi-LoRA batching | ✅ Full | SGMV kernels with segment grouping |
| MoE (EP) | ✅ Full | DeepEP + HCCL all-to-all |
| Speculative Decoding | ✅ Partial | May use fallback for some draft methods |
| HiCache | ✅ Full | `kernel_ascend` backend |
| Grammar | ✅ Full | Same as GPU path (CPU-side) |
| Quantization | ✅ Partial | FP8 supported, GPTQ/AWQ varies |

## 4. Fallback Paths

When Ascend-specific kernel is unavailable:

```python
# Example fallback pattern
if is_npu() and not has_npu_kernel(op):
    # Fall back to generic implementation
    return generic_implementation(...)
```

Fallbacks are correctness-safe but slower. Identify them via:
- Profiling traces showing unexpected CPU or generic kernel calls
- Log messages about "fallback" or "disable"
- Performance gaps vs GPU baseline
