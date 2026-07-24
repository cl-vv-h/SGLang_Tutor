[中文](./01-repository-and-op-lifecycle.md) | [English](./01-repository-and-op-lifecycle_EN.md)

# sgl-kernel-npu 01: Repository Structure & Operator Lifecycle

## Repository Layout

```text
sgl-kernel-npu/
├── python/           # Python wrappers & tests
│   └── sgl_kernel_npu/
├── csrc/             # C++/Ascend C source
│   ├── attention/
│   ├── moe/
│   ├── norm/
│   └── ...
├── cmake/            # Build configuration
└── setup.py          # Package build
```

## Operator Lifecycle

```text
1. import sgl_kernel_npu        # Python import triggers .so load
2. torch.ops.sgl_kernel.xxx()   # Python calls registered op
3. PyTorch dispatcher            # Routes to NPU backend
4. C++ host-side dispatch        # Tiling, workspace allocation
5. Launch stub                   # Sets blockDim, arguments
6. Device kernel                 # Ascend C / Triton kernel executes
```

## Registration Pattern

```python
# Python wrapper in sgl_kernel_npu
@torch.library.impl("sgl_kernel", "fused_attention", "PrivateUse1")
def fused_attention_npu(q, k, v, ...):
    # Validate inputs
    # Call C++ host-side function
    return _C.fused_attention(q, k, v, ...)
```

## Build System

```cmake
# cmake/ascend.cmake
ascendc_add_library(sgl_kernel_npu SHARED
    csrc/attention/ascend_attention.cpp
    csrc/moe/moe_dispatch.cpp
    ...
)
```
