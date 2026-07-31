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

## `TPosition`: Logical Placement

`TPosition` describes a tensor's role in the Device data path. It hides part of the product-specific physical-memory mapping:

| Logical location | Common physical role | Meaning |
|---|---|---|
| `VECIN` / `VECCALC` / `VECOUT` | UB | Vector input, temporary, and output |
| `A1` / `B1` | L1 | Larger left/right Cube operand tiles |
| `C1` | L1 or UB | Bias input, product-dependent |
| `A2` / `B2` | L0A/L0B | Near-Cube left/right blocks |
| `C2` | BT or L0C | Near-compute bias block, product-dependent |
| `CO1` | L0C | Block-wise Cube result/accumulator |
| `CO2` | GM or UB | Final Cube output stage, product-dependent |

The `C` in `C1/C2` denotes the bias-input role, while `CO` denotes Cube output. The suffix is a logical pipeline stage, not a hardware generation or an ordinary CPU L1/L2 cache number. See [Foundation 02](../foundations/02-ascend-hardware.md#a1b1c1-a2b2c2-and-co1co2) for the complete path.

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
