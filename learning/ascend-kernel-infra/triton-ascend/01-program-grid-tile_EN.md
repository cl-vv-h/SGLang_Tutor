[中文](./01-program-grid-tile.md) | [English](./01-program-grid-tile_EN.md)

# Triton-Ascend 01: Program, Grid, Tile & First Kernel

## Vector Add: Triton vs PyTorch

```python
# PyTorch (eager):
c = a + b  # Single operation, CPU dispatches to GPU

# Triton (kernel):
@triton.jit
def add_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a + b
    tl.store(c_ptr + offsets, c, mask=mask)

# Launch with grid
grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
add_kernel[grid](a, b, c, n, BLOCK_SIZE=256)
```

## Key Concepts

| Concept | Meaning | In Code |
|---|---|---|
| Program | The kernel function | `@triton.jit` decorated function |
| Grid | How many program instances (blocks) | `grid` function |
| Block | One instance of the program on one core | `pid = tl.program_id(0)` |
| Tile | Chunk of data processed by one block | `BLOCK_SIZE` elements |

## Triton → Ascend NPU Pipeline

```text
Python Triton kernel (.py)
  → Triton-Ascend compiler
    → TTIR (Triton IR)
    → MLIR (Ascend dialect)
    → NPU binary (.o)
  → Ascend runtime loads and executes
```

## Ascend Grid Strategy

On Ascend NPU, `grid` maps differently than CUDA:
- CUDA: grid → thread blocks on SMs
- Ascend: grid → cores on AI Core array

The Triton-Ascend compiler handles the mapping automatically.
