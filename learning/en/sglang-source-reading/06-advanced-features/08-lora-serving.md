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

Standard LoRA forward:

```math
y=W_{\mathrm{base}}\,x+B\,A\,x\cdot \mathrm{scaling}
```

```text
SGLang optimized (segment-based):
For each unique LoRA adapter in batch:
```

```math
\mathrm{affected\_tokens} = \mathrm{find\_tokens\_with\_lora}\left(\mathrm{adapter\_id}\right)
```

```math
\delta = \mathrm{sgmv\_shrink}\left(\mathrm{affected\_x}, A\right) \qquad\text{x @ A^T, segmented}
```

```math
\delta = \mathrm{sgmv\_expand}\left(\delta, B\right) \qquad\text{delta @ B^T, segmented}
```

```math
y\left[\mathrm{affected\_tokens}\right] += \delta \cdot  \mathrm{scaling}
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
