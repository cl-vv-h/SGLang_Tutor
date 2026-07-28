# Foundation 03: Data Movement, Synchronization & Pipeline

## CopyIn-Compute-CopyOut Pattern

```text
Every NPU kernel follows this pattern:

1. CopyIn:  Move input data from GM → UB (DMA)
2. Compute: Execute kernel on data in UB
3. CopyOut: Move results from UB → GM (DMA)
```

## Pipeline & Double Buffering

```text
Without pipeline:
  [CopyIn_0][Compute_0][CopyOut_0][CopyIn_1][Compute_1][CopyOut_1]...
  Total time = N × (T_copyin + T_compute + T_copyout)

With double-buffer pipeline:
  [CopyIn_0][CopyIn_1 | Compute_0][CopyIn_2 | Compute_1 | CopyOut_0][Compute_2 | CopyOut_1]...
  Total time ≈ max(T_copyin, T_compute, T_copyout) × N (ideal)
```

## Ascend C Queue (TQue)

Queue-based synchronization between pipeline stages:

```cpp
// Producer stage
TQue<LocalTensor<float>, 2> data_queue;  // 2-deep queue
LocalTensor<float> produced = ...;
data_queue.EnQueue(produced);

// Consumer stage
LocalTensor<float> consumed = data_queue.DeQueue();
// Use consumed data
data_queue.Free(consumed);  // Return to producer
```

## Synchronization Primitives

| Primitive | Scope | Purpose |
|---|---|---|
| `TQue::EnQueue/DeQueue` | Between pipeline stages | Data passing with backpressure |
| `SetFlag/WaitFlag` | Between cores | Event signaling |
| `Barrier()` | All cores in block | Synchronization point |
| Stream order / Event | Between kernel launches | Cross-kernel dependency without necessarily blocking Host |
| `stream.synchronize()` | Host/Device | Host waits for work previously submitted to one Stream |

A runtime **Stream** is an ordered Host-to-Device submission sequence spanning kernel launches, asynchronous copies, and Events. It is not an Ascend C `TQue`: a `TQue` coordinates tile stages inside one Device kernel. See [torch_npu 02: Streams, Events, Asynchronous Lifetimes, and Graph Capture](../torch_npu/02-stream-events-and-graph-capture.md) for current/default Streams, `record_stream`, Events, and the distinction from computation graphs.

## Arithmetic Intensity

```text
Arithmetic Intensity = FLOPs / Bytes_Moved

High AI (compute-bound): Matrix multiply, convolution
  → Optimize for Cube Unit utilization

Low AI (memory-bound): Element-wise ops, reductions, norms
  → Optimize for memory access patterns, fuse with neighbors

Example:
  MatMul [M,K]×[K,N]: 2MNK FLOPs / 2(MK+KN+MN) bytes → ~O(min(M,N,K))
  ReLU [N]: N FLOPs / 2N bytes → ~0.5 (very memory-bound)
```
