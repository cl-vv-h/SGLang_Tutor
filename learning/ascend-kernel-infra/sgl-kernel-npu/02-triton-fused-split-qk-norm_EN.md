[中文](./02-triton-fused-split-qk-norm.md) | [English](./02-triton-fused-split-qk-norm_EN.md)

# sgl-kernel-npu 02: Triton Fused Split Q/K Norm

## Operation

```text
Input:  qkv_tensor [T, Qdim + 2*KVdim]
Output: q_norm [T, Nq, D], k_norm [T, Nkv, D]

Operation:
  1. Split qkv → q, k, v
  2. Apply RMSNorm per head to q and k
  3. Return normalized q, k
```

## Triton Kernel: 3-Segment Tile

The kernel processes tokens in three tile segments:
- Segment 1: Q tokens (Nq heads, D dims each)
- Segment 2: K tokens (Nkv heads, D dims each)
- Segment 3: V tokens (pass-through, no norm)

Grid: `(B,)` — one program per batch element

## Key Optimizations

- **Fused**: Split + Norm in one kernel, avoiding intermediate buffers
- **FP32 reduction**: RMS computation in FP32 for numerical stability
- **Constexpr bias**: Optional bias fusion at compile time
- **Tile alignment**: Tiles aligned to NPU memory access granularity
