[中文](./08-lora-serving.md) | [English](./08-lora-serving_EN.md)

# LoRA Serving in SGLang

## 1. LoRA Integration Architecture

```text
Request with lora_id → Scheduler → ScheduleBatch → ForwardBatch
  → ModelRunner: LoRA-aware forward
    → LoRA Backend: sgmv_shrink/sgmv_expand kernels
      → Adapted weights merged into base weights at runtime
```

## 2. LoRA Backend

| Backend | Source | Platform |
|---|---|---|
| CUDA LoRA | `lora/backend/cuda_backend.py` | NVIDIA GPUs |
| Ascend LoRA | `lora/backend/ascend_backend.py` | Ascend NPU |
| Triton LoRA | `lora/backend/triton_backend.py` | Cross-platform |

## 3. Key Operations

```text
Standard LoRA forward:
  y = W_base @ x + (B @ A) @ x * scaling

SGLang optimized (segment-based):
  For each unique LoRA adapter in batch:
    affected_tokens = find_tokens_with_lora(adapter_id)
    delta = sgmv_shrink(affected_x, A)  # x @ A^T, segmented
    delta = sgmv_expand(delta, B)        # delta @ B^T, segmented
    y[affected_tokens] += delta * scaling
```

## 4. LoRABatchInfo

```python
LoRABatchInfo:
    weight_indices: Tensor[B]     # Which LoRA adapter per request
    seg_lens: Tensor[num_loras]   # How many tokens per adapter  
    scalings: Tensor[num_loras]   # Scaling factor per adapter
    lora_ranks: List[int]         # Rank per adapter
```

## 5. Multi-LoRA Batching Constraints

- One batch can contain requests with different LoRA adapters
- `PrefillAdder` limits number of unique adapters per batch
- `LoRAMemoryPool` manages adapter weight storage on GPU
- Hot-load/unload supported at runtime
- `sgmv_shrink`/`sgmv_expand` are segment-grouped matrix-vector ops optimized for batched LoRA

## 6. Code References

- `python/sglang/srt/lora/` — LoRA core
- `python/sglang/srt/lora/backend/` — Backend-specific kernels
- `python/sglang/srt/managers/scheduler.py` — LoRA-aware scheduling
- `python/sglang/srt/model_executor/model_runner.py` — LoRA forward integration
