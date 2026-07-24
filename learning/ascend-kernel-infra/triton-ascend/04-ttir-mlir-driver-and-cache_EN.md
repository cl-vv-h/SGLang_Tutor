[中文](./04-ttir-mlir-driver-and-cache.md) | [English](./04-ttir-mlir-driver-and-cache_EN.md)

# Triton-Ascend 04: TTIR, MLIR, Driver & Cache

## IR Pipeline

```text
Triton Python → TTIR → MLIR (Ascend) → NPU Binary

TTIR: Triton's internal IR
  - Device-agnostic
  - Captures tile structure, memory operations, compute

MLIR (Ascend dialect):
  - Device-specific (Ascend NPU)
  - Cube/Vector unit assignment
  - Memory hierarchy mapping (GM/L1/UB)
  - Tiling decomposition

NPU Binary:
  - Final executable for Ascend runtime
  - Cached for reuse
```

## Kernel Cache

```python
# Triton-Ascend caches compiled kernels
# Cache key: (kernel_hash, grid, signature)
# Cache location: ~/.triton/cache/ (default)

# Force recompile:
import triton
triton.runtime.cache.clear()
```

## Driver Communication

```text
Triton-Ascend → CANN Runtime → ACL → Ascend Driver → NPU Hardware

Key driver operations:
  - Memory allocation (HBM)
  - Kernel launch (to AI Cores)
  - Stream synchronization
  - Format conversion
```

## Intermediate Artifacts

```bash
# Dump TTIR
TRITON_DUMP_IR=1 python my_kernel.py

# Inspect MLIR
TRITON_ASCEND_DUMP_MLIR=1 python my_kernel.py

# Check cache
ls ~/.triton/cache/
```
