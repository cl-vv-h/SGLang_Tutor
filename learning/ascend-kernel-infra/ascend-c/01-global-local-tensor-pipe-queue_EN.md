[中文](./01-global-local-tensor-pipe-queue.md) | [English](./01-global-local-tensor-pipe-queue_EN.md)

# Ascend C 01: GlobalTensor, LocalTensor, TPipe & TQue

## GlobalTensor: View of GM Data

```cpp
// GlobalTensor: data resides in Global Memory (HBM)
GlobalTensor<float> input;   // Read-only view of GM
GlobalTensor<float> output;  // Write-only view of GM

// Initialize from kernel launch parameters
input.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(input_gm));
output.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output_gm));
```

## LocalTensor: View of UB Data

```cpp
// LocalTensor: data resides in Unified Buffer (on-chip)
LocalTensor<float> local_input;   // Local copy of input tile
LocalTensor<float> local_output;  // Local copy of output tile

// Allocate UB space
PipeBarrier<LocalTensor<float>> local_input_pipe;
local_input = local_input_pipe.AllocTensor<float>();
```

## TPipe: Pipeline Framework

```cpp
class KernelAdd {
public:
    __aicore__ void Process() {
        // Define pipeline stages
        // Stage 1: CopyIn (DMA from GM→UB)
        // Stage 2: Compute (operate on UB data)
        // Stage 3: CopyOut (DMA from UB→GM)
    }
};
```

## TQue: Inter-Stage Communication

```cpp
// Between CopyIn and Compute:
TQue<LocalTensor<float>, 2> compute_queue;  // Depth 2

// CopyIn stage enqueues:
LocalTensor<float> tile = ...;
compute_queue.EnQue(tile);

// Compute stage dequeues:
LocalTensor<float> data = compute_queue.DeQue();
// ... compute on data ...
compute_queue.Free(data);
```

## init() / process() Pattern

```cpp
__aicore__ void Init() {
    // One-time setup: allocate UB buffers, initialize queues
}

__aicore__ void Process() {
    // Main kernel loop: CopyIn → Compute → CopyOut
    for (int i = 0; i < num_tiles; i++) {
        CopyIn(i);    // DMA tile from GM to UB
        Compute(i);   // Process tile in UB
        CopyOut(i);   // DMA result from UB to GM
    }
}
```
