[中文](./03-compile-debug-optimize.md) | [English](./03-compile-debug-optimize_EN.md)

# Triton-Ascend 03: Compilation, Debugging & Optimization

## Compilation Pipeline

```text
Python Triton kernel
  → @triton.jit decorator
  → Triton-Ascend compiler (Ascend backend)
  → TTIR (Triton Intermediate Representation)
  → MLIR transformations (Ascend dialect)
  → NPU binary / kernel object
  → Runtime cache (auto-reuse for same signature)
```

## Debugging Workflow

```text
Error encountered:
  1. Is it a Triton language error? → Check Python syntax, tl.* usage
  2. Is it a compiler error? → Check TTIR/MLIR output (TRITON_DUMP_IR=1)
  3. Is it a runtime error? → Check device compatibility, memory access
  4. Is it a correctness error? → Compare against PyTorch reference
```

## Environment Variables

| Variable | Purpose |
|---|---|
| `TRITON_DUMP_IR=1` | Dump TTIR/MLIR intermediate representations |
| `TRITON_CACHE_DIR` | Override kernel cache location |
| `ASCEND_LAUNCH_BLOCKING=1` | Synchronous kernel launch for debugging |

## Optimization Loop

```text
1. Establish baseline (PyTorch reference or naive Triton)
2. Profile: identify bottleneck (compute vs memory)
3. Optimize:
   - Increase BLOCK_SIZE for better occupancy
   - Use tl.dot for Cube Unit acceleration
   - Fuse operations to reduce memory traffic
   - Prefer fp32 accumulation, fp16 storage
4. Benchmark: measure improvement
5. Autotune: let Triton search config space
```

## Autotune

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(...):
    ...
```
