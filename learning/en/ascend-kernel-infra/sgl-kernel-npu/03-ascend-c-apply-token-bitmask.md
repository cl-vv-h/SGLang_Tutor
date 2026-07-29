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

The real Host wrapper obtains the current Stream and records the two working tensors before launch:

```cpp
auto npuStream = c10_npu::getCurrentNPUStream();
workingLogits.record_stream(npuStream);
workingBitmask.record_stream(npuStream);
```

`npuStream` is a Host-side `c10_npu::NPUStream`. The calls insert current Stream into the two storage blocks' allocator `stream_uses`; they do not prove that this operator crosses Streams.

Branch-by-branch analysis shows that index/zeros/contiguous preparation, `EXEC_KERNEL_CMD`, and later copy/scatter all use current Stream. The wrapper creates no side Stream. Both records are therefore correctness-redundant for normal same-Stream calls, not requirements of bitmask mathematics or raw-pointer asynchronous launch. `workingLogits` either aliases returned `logits` or is a current-Stream temporary, and `workingBitmask` is normally allocated/used on current Stream.

They add limited protection only when working storage aliases an input allocated on another Stream and the caller may release its final reference early. They do not establish data readiness and do not cover every possible alias, so they are not complete support for arbitrary cross-Stream inputs.

Other repository wrappers have no hidden generic record: `EXEC_KERNEL_CMD` only gets current Stream and converts arguments. Local tiling/workspace in `causal_conv1d` and `catlass_matmul_basic` is also submitted asynchronously but relies on same allocation/current-Stream ordering. A real side/communication-Stream handoff needs `record_stream(consumer)` or an Event-plus-keepalive/reverse-wait scheme. See the [allocator source, branch audit, source comparisons, and experiments](../torch_npu/02-stream-events-and-graph-capture.md#74-why-is-apply_token_bitmask-the-only-explicit-call-site).

`record_stream` does not synchronize execution or build a computation graph. Cross-Stream readiness still requires an Event or `wait_stream`.

The three Ascend C `TQue` objects solve a different problem: CopyIn/Compute/CopyOut ordering inside this one Device kernel. They are not runtime Streams. See [torch_npu 02: Streams, Events, Asynchronous Lifetimes, and Graph Capture](../torch_npu/02-stream-events-and-graph-capture.md) for the complete distinction.
