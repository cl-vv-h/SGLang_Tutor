[中文](./02-ascend-hardware.md) | [English](./02-ascend-hardware_EN.md)

# Foundation 02: Ascend NPU Hardware & Memory Hierarchy

## Da Vinci AI Core Architecture

```text
┌──────────────────────────────────────┐
│           AI Core (Da Vinci)          │
│  ┌─────────┐ ┌──────┐ ┌──────────┐  │
│  │  Scalar  │ │Vector│ │   Cube   │  │
│  │  Unit    │ │ Unit │ │   Unit   │  │
│  │(control) │ │(SIMD)│ │ (MatMul) │  │
│  └─────────┘ └──────┘ └──────────┘  │
│  ┌────────────────────────────────┐  │
│  │         Unified Buffer (UB)     │  │
│  │         On-chip scratchpad      │  │
│  └────────────────────────────────┘  │
│  ┌────────────────────────────────┐  │
│  │     Memory Transfer Engine     │  │
│  │         (MTE / DMA)            │  │
│  └────────────────────────────────┘  │
└──────────────────────────────────────┘
```

## Memory Hierarchy

| Level | Name | Size (typical) | Bandwidth | Latency |
|---|---|---|---|---|
| HBM | Global Memory (GM) | 32-64 GB | ~1-2 TB/s | ~hundreds ns |
| L2 | On-chip Cache | ~32-64 MB | ~4-8 TB/s | ~tens ns |
| L1 | Per-core Buffer | ~1 MB | ~8-16 TB/s | ~few ns |
| L0/UB | Unified Buffer | ~192 KB | ~16-32 TB/s | ~single cycle |

## Compute Units

| Unit | Purpose | Data Types | Peak Throughput |
|---|---|---|---|
| Cube Unit | Matrix multiply-accumulate | FP16, BF16, INT8, FP8 | ~256 TFLOPS (FP16) |
| Vector Unit | Element-wise ops, reductions, activations | FP32, FP16, BF16 | ~32 TFLOPS (FP32) |
| Scalar Unit | Control flow, address calculation | INT32, FP32 | Low, for control only |

## Key Hardware Constraints

1. **Cube Unit efficiency**: Works best with matrices aligned to 16×16 blocks
2. **UB capacity**: Only ~192KB — kernels must tile data to fit within UB
3. **Data movement cost**: GM→UB transfer is ~10-100× slower than UB access
4. **Double buffering**: Must overlap compute with data movement to hide latency
5. **Bank conflicts**: UB memory banks can cause stalls if accessed poorly
