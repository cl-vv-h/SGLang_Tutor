[中文](./03-ascend-c-apply-token-bitmask.md) | [English](./03-ascend-c-apply-token-bitmask_EN.md)

# sgl-kernel-npu 03: Ascend C Apply Token Bitmask

## Purpose

Apply a packed bitmask to token selection, used for structured output (grammar-constrained generation). The bitmask restricts which tokens are valid at each generation step.

## Ascend C Implementation

```text
Key design:
  - Host-side UB tiling: tiles computed on CPU, passed to kernel
  - Row-wise core assignment: each core processes a range of rows
  - Three TQue: input_queue (data), mask_queue (bitmask), output_queue (result)
  - Packed bitmask: 1 bit per token, compressed for bandwidth efficiency
```

## Data Flow

```text
GM:  logits [B, V]     # Unconstrained logits
     bitmask [B, V//8]  # Packed bitmask (1 bit/token)

UB:  tile_logits [TILE_B, V]     # Per-core tile
     tile_bitmask [TILE_B, V//8]  # Corresponding bitmask

Compute:
  For each (b, v) in tile:
    if bitmask[b][v] == 0:
        logits[b][v] = -inf     # Mask out invalid token

GM:  masked_logits [B, V]  # Constrained logits
```

## Async Lifecycle

- Bitmask loaded asynchronously from host
- Can be updated mid-generation (new grammar constraints)
- Three-TQue pipeline ensures correct ordering
