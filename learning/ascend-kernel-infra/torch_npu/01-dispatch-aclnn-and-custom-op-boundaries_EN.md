[中文](./01-dispatch-aclnn-and-custom-op-boundaries.md) | [English](./01-dispatch-aclnn-and-custom-op-boundaries_EN.md)

# torch_npu 01: Dispatcher, ACLNN & Custom Op Boundaries

## Three Dispatch Paths

```text
torch.add(x, y)
  └─ PyTorch dispatcher
      ├─ CPU path: standard PyTorch CPU kernel
      ├─ CUDA path: CUDA kernel
      └─ PrivateUse1 (NPU) path: torch_npu
          ├─ ACLNN op: torch_npu calls CANN's aclnnAdd()
          ├─ ACLOP op: torch_npu calls CANN's aclopExecute()
          └─ Custom op: torch_npu calls custom .so kernel
```

## When Each Path is Used

| Path | When | Example |
|---|---|---|
| ACLNN | Standard neural network ops with NPU-optimized impl | MatMul, Conv2D, LayerNorm, Softmax |
| ACLOP | General operators not in ACLNN | Element-wise ops, custom reductions |
| Custom Op | SGLang-specific fused ops from sgl-kernel-npu | Fused QK Norm, Token Bitmask, MoE dispatch |

## Dispatch Decision in torch_npu

```python
# Simplified dispatch logic
def npu_add(a, b):
    if has_aclnn_impl("add"):
        return aclnn_add(a, b)       # Fast path: ACLNN
    elif has_acl_op("Add"):
        return acl_op_execute("Add", a, b)  # Fallback: ACLOP
    else:
        return custom_add_kernel(a, b)      # Last resort
```

## Registration Points

```python
# torch_npu registers itself as the PrivateUse1 backend
torch.utils.rename_privateuse1_backend("npu")

# Custom ops register under their own namespace
torch.ops.sgl_kernel.fused_attention(...)   # sgl-kernel-npu
torch.ops.npu.sgmv_shrink(...)              # torch_npu custom
```
