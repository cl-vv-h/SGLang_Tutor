# Foundation 01: From Formula to Parallel Kernel

## Core Concepts

### Operator vs Kernel

- **Operator**: The mathematical operation (e.g., matrix multiply, ReLU). Framework-level concept.
- **Kernel**: The actual device code that executes the operator. Hardware-level concept.

```math
\mathrm{Operator}: C = A \, B \left(\mathrm{what} \mathrm{to} \mathrm{compute}\right)
```

```text
Kernel:   __global__ void matmul_kernel(float* A, float* B, float* C)  (how to compute)
```

### Host vs Device

| Concept | Location | Role |
|---|---|---|
| Host | CPU | Manages data, launches kernels, controls flow |
| Device | NPU/GPU | Executes kernels, stores device memory |

### SPMD (Single Program, Multiple Data)

The fundamental parallel programming model: the same kernel code runs on many cores simultaneously, each processing different data.

```python
# Conceptual SPMD
def vector_add_kernel(a, b, c, my_core_id):
    my_start = my_core_id * chunk_size
    my_end = my_start + chunk_size
    for i in range(my_start, my_end):
        c[i] = a[i] + b[i]
```

### Program → Grid → Tile

```text
Program: The full computation (e.g., add two [N] vectors)
  Grid:    Split across M physical cores
    Tile:  Each core processes [N/M] elements per tile
```

### Shape vs Stride

- **Shape**: Logical dimensions $`\left[B,H\right]`$ — how we think about the tensor
- **Stride**: Memory layout $`\left[\mathrm{stride\_B},\mathrm{stride\_H}\right]`$ — how data is physically stored

A $`\left[3,4\right]`$ row-major tensor has stride $`\left[4,1\right]`$. A transposed view changes strides, not data.

## Why Parallel Kernels Matter for LLM Serving

Every operation in SGLang's forward pass becomes kernel launches:
- Attention: QKV projection kernels, attention compute kernel, output projection kernel
- MoE: Router kernel, dispatch kernel, expert GEMM kernels, combine kernel
- Norm: RMSNorm kernel
- Embedding: Lookup kernel

Understanding kernel principles helps you:
1. Read kernel source code in sgl-kernel-npu
2. Understand why certain shapes are more efficient
3. Diagnose performance issues from profiler traces
