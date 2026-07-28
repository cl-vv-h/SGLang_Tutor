# sgl-kernel-npu 03: Ascend C Apply Token Bitmask

## Purpose

Apply a packed bitmask to token selection, used for structured output (grammar-constrained generation). The bitmask restricts which tokens are valid at each generation step.

## Host Type and Namespace Boundary

The real Host signature contains:

```cpp
HOST_API at::Tensor apply_token_bitmask(
    at::Tensor logits,
    at::Tensor bitmask,
    c10::optional<at::Tensor> indices);
```

Read `c10::optional<at::Tensor>` from the outside in. `at::Tensor` is a PyTorch/ATen Host tensor handle. `c10::optional<T>` means that a value of type `T` may be absent. The complete type therefore represents Python `indices=None` or a real indices tensor; it is not an Ascend C local tensor.

`c10` is the PyTorch core namespace and `at` is the ATen namespace. C++ templates can freely compose types from different namespaces. A present `indices` value still needs checks for `defined()`, `numel()`, dtype, device, and dimensions. See the [detailed optional-Tensor explanation](../reference/code-reading-and-types.md#5-read-c10optionalattensor-from-the-outside-in).

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
