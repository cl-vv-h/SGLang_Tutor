[中文](./04-fla-chunk-gated-delta-rule-mixed-path.md) | [English](./04-fla-chunk-gated-delta-rule-mixed-path_EN.md)

# sgl-kernel-npu 04: FLA Chunk Gated Delta Rule — Dual Path

## FLA (Flash Linear Attention)

Combines chunked linear attention with gated delta rule for efficient long-context processing.

## Dual-Path Dispatch

```text
Same Python API: sgl_kernel_npu.fla_chunk_gated_delta_rule(...)

Path A: Segmented Triton kernel
  → For standard shapes, moderate batch sizes
  → Packed B=1, cu_seqlens for variable-length support

Path B: Mega custom op (Ascend C)
  → For large batch, high throughput
  → Custom state management, blockDim coordination
  → Manages workspace contracts for intermediate buffers
```

## Path Selection Logic

```python
def fla_chunk_gated_delta_rule(q, k, v, ...):
    if should_use_mega_kernel(batch_size, seq_len, dtype):
        return mega_kernel_path(q, k, v, ...)
    else:
        return triton_kernel_path(q, k, v, ...)
```
