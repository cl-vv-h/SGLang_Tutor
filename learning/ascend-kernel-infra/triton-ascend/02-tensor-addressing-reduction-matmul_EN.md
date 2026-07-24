[中文](./02-tensor-addressing-reduction-matmul.md) | [English](./02-tensor-addressing-reduction-matmul_EN.md)

# Triton-Ascend 02: Tensor Addressing, Reduction & MatMul

## 2D Addressing

```python
@triton.jit
def copy_2d_kernel(src, dst, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # 2D tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # 2D load with masking
    mask_m = offs_m[:, None] < M
    mask_n = offs_n[None, :] < N
    
    data = tl.load(src + offs_m[:, None] * N + offs_n[None, :], 
                   mask=mask_m & mask_n)
    tl.store(dst + offs_m[:, None] * N + offs_n[None, :], 
             data, mask=mask_m & mask_n)
```

## RMSNorm in Triton

```python
@triton.jit
def rmsnorm_kernel(x, y, gamma, N, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    
    x_val = tl.load(x + offs, mask=mask, other=0.0)
    
    # RMS reduction
    x_sq = x_val * x_val
    rms = tl.sqrt(tl.sum(x_sq) / N + 1e-6)
    
    # Normalize + scale
    y_val = gamma * x_val / rms
    tl.store(y + offs, y_val, mask=mask)
```

## Tiled MatMul

```python
@triton.jit
def matmul_kernel(A, B, C, M, N, K, 
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Accumulator in registers
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        a_tile = tl.load(A + ..., mask=...)
        b_tile = tl.load(B + ..., mask=...)
        acc += tl.dot(a_tile, b_tile)
    
    tl.store(C + ..., acc.to(dtype))
```
