[中文](./code-reading-and-types.md) | [English](./code-reading-and-types_EN.md)

# Code Reading Manual: Types, Shapes & Addresses

## Type Distinctions

| Type System | Example | What It Is |
|---|---|---|
| Python object | `torch.Tensor` | Host-side tensor handle (metadata + data pointer) |
| Triton IR type | `tl.tensor` | Triton's internal representation (≠ torch.Tensor) |
| Pointer | `tl.tensor.to(pointer_type)` | Raw memory address in Triton |
| Value block | `tl.tensor` after load | Data loaded into registers/UB |

## Key Distinction: tl.tensor ≠ torch.Tensor

```python
# torch.Tensor: host-side, contains metadata + GPU data pointer
x = torch.randn(256, device="cuda")

# tl.tensor: device-side, inside Triton kernel
@triton.jit
def kernel(x_ptr, ...):
    # x_ptr is a pointer to GM
    data = tl.load(x_ptr + offsets)  # data is a tl.tensor (register/UB value)
```

## Ascend C Type Hierarchy

| Type | Memory Location | Usage |
|---|---|---|
| `GlobalTensor<T>` | GM (HBM) | Typed view of global memory |
| `LocalTensor<T>` | UB (on-chip) | Typed view of unified buffer |
| `T*` (raw pointer) | GM | Raw address (used with offset arithmetic) |

## Why Pointer + Offset Works

```cpp
GlobalTensor<float> A;  // Base address in GM
int offset = 128;
// A[offset] is valid because GlobalTensor overloads operator[]
// Equivalent to: *(reinterpret_cast<__gm__ float*>(A.GetPhyAddr()) + offset)
```
