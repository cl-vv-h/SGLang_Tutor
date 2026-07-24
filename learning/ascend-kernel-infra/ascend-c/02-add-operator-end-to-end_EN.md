[中文](./02-add-operator-end-to-end.md) | [English](./02-add-operator-end-to-end_EN.md)

# Ascend C 02: Add Operator End-to-End

## Complete Add Operator Implementation

### 1. Host-Side Tiling & Launch

```cpp
// Host code: compute tiling, set blockDim, launch kernel
void launch_add_kernel(
    const float* a, const float* b, float* c, int n)
{
    constexpr int tile_size = 256;
    int num_tiles = (n + tile_size - 1) / tile_size;
    int block_dim = min(num_tiles, MAX_CORES);
    
    // Launch kernel with blockDim cores
    KernelAdd<<<block_dim>>>(a, b, c, n, tile_size);
}
```

### 2. Device Kernel

```cpp
__aicore__ void Process() {
    int core_id = get_core_id();
    int tile_start = core_id * tile_size;
    int tile_end = min(tile_start + tile_size, n);
    int tile_len = tile_end - tile_start;
    
    // CopyIn: load tile from GM to UB
    DataCopy(local_a, global_a[tile_start:tile_end]);
    DataCopy(local_b, global_b[tile_start:tile_end]);
    
    // Compute: vector add on UB
    VecAdd(local_c, local_a, local_b, tile_len);
    
    // CopyOut: store result from UB to GM
    DataCopy(global_c[tile_start:tile_end], local_c);
}
```

### 3. PyTorch Registration

```python
# Python: register custom op in PyTorch
import torch
from torch.library import Library

lib = Library("custom_ops", "DEF")
lib.define("add(Tensor a, Tensor b) -> Tensor")

@torch.library.impl(lib, "add", "PrivateUse1")  # NPU backend
def add_npu(a, b):
    c = torch.empty_like(a)
    launch_add_kernel(
        a.data_ptr(), b.data_ptr(), c.data_ptr(), a.numel()
    )
    return c
```

### 4. Build & Load

```bash
# Compile Ascend C kernel → .so
ascendc -c kernel_add.cpp -o kernel_add.o
ascendc -shared kernel_add.o -o libkernel_add.so

# Load in Python
torch.ops.load_library("libkernel_add.so")
result = torch.ops.custom_ops.add(a, b)
```

## End-to-End Checklist

- [ ] Host tiling: correct tile size and blockDim
- [ ] Device kernel: CopyIn/Compute/CopyOut pattern
- [ ] Shape handling: edge tiles may be smaller
- [ ] PyTorch registration: correct backend (`PrivateUse1` for NPU)
- [ ] Build system: correct compiler flags for Ascend C
- [ ] Testing: verify correctness vs CPU reference
