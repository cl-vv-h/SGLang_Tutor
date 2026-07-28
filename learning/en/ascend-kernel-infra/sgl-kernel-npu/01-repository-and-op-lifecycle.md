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

The Host wrapper normally submits the launch asynchronously to PyTorch's **current NPU Stream**. `c10::Stream` is PyTorch's backend-neutral identity, `c10_npu::NPUStream` is torch_npu's NPU wrapper, and `aclrtStream` is the opaque native CANN handle passed to a launch. Using the current rather than blindly using the default Stream preserves the producer/custom-op/consumer order selected by the caller.

Custom raw launches must also keep asynchronously used tensor storage alive. `tensor.record_stream(stream)` informs the caching allocator about that use; it does not make one Stream wait for another. Read [torch_npu 02: Streams, Events, Asynchronous Lifetimes, and Graph Capture](../torch_npu/02-stream-events-and-graph-capture.md) for the full type path, Event dependencies, and source walkthrough.

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
