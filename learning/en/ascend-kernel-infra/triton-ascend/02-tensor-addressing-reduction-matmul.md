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

## Does Triton Need Ascend C `TPosition`?

Usually, a Triton kernel does not write `TPosition::A1`, `VECIN`, or `CO1`. The programmer describes tile-level data flow; Triton-Ascend lowering, bufferization, and backend code generation choose UB, L1, L0A/L0B/L0C, transfers, and synchronization.

| Concern | Ascend C | Triton-Ascend |
|---|---|---|
| GM input | `GlobalTensor<T>` | Pointer argument and `tl.load` |
| Local value | `LocalTensor<T>` plus `TPosition`/queue | `tl.tensor` SSA block value |
| Cube path | Explicit A1/A2/B1/B2 or high-level Matmul API | `tl.dot` plus tile/dtype/layout |
| Vector path | VECIN/VECCALC/VECOUT | Block operations lowered mainly to UB/Vector work |
| Movement | Explicit `DataCopy` and queue protocol | Mostly generated from `tl.load/store` and compiler passes |

A `tl.tensor` is not automatically a permanently allocated UB buffer. It is first an IR SSA value: the compiler may materialize it in local memory, fuse it away, split it into internal loops, or rematerialize it.

The backend still models real hardware address spaces. [`ascend_ir.cc`](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/third_party/ascend/ascend_ir.cc#L461-L466) exposes `L1`, `UB`, `L0A`, `L0B`, and `L0C`. The distinction is:

```text
Ascend C: programmer explicitly chooses logical locations and movement
Triton:   programmer describes load/tile compute/store;
          compiler chooses and verifies physical placement
```

For Vector Add, the conceptual path is `GM -> UB -> Vector -> UB -> GM`. `BLOCK_SIZE`, dtype, masks, live temporaries, and multibuffering determine UB pressure even though the source never writes `VECIN`.

For `tl.dot`, the conceptual path is `GM -> L1 -> L0A/L0B -> Cube -> L0C -> output`. Tile shapes and layouts drive the backend's Cube placement. `C1/C2` are bias-input roles and only matter when the generated matrix path has a bias; `CO1/CO2` are output roles. See [Foundation 02](../foundations/02-ascend-hardware.md#a1b1c1-a2b2c2-and-co1co2).

Triton developers must still manage hardware indirectly:

1. keep `BLOCK_M/N/K`, accumulators, masks, and live temporaries within UB/L1/L0 budgets;
2. choose layouts and alignments that permit efficient movement and Cube instructions;
3. understand whether an operation maps to Vector, Cube, or CV fusion;
4. account for `num_stages`, multibuffering, and other compiler options;
5. inspect IR dumps, memory reports, compiler errors, and profiler data.

Some Ascend extensions expose more explicit UB-oriented operations or compile hints, but they still do not turn ordinary Triton code into Ascend C `TQue<TPosition::...>` code.
