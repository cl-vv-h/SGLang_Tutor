[中文](./03-tiling-pipeline-sync-optimization.md) | [English](./03-tiling-pipeline-sync-optimization_EN.md)

# Ascend C 03: Tiling, Pipeline, Sync & Optimization

## Tiling Strategy

| Strategy | Description | Best For |
|---|---|---|
| 1D Tiling | Split along one dimension | Vector ops, element-wise |
| 2D Tiling | Split along two dimensions | MatMul, 2D convolutions |
| Multi-core | Split across AI Cores | Large problems |
| Intra-core | Split within one core's UB | Fitting data in UB |

## Double Buffer Pipeline

```cpp
// Double-buffered CopyIn + Compute
TQue<LocalTensor<float>, 2> input_queue;
TQue<LocalTensor<float>, 2> output_queue;

// Pipeline:
// While CopyIn(tile_i+1):
//   Compute(tile_i) → CopyOut(tile_i)
// Benefit: hides DMA latency behind computation
```

## Cube + Vector Pipeline

```text
Typical pattern for MatMul:
  CopyIn A_tile, B_tile → Cube(MatMul) → Vector(Add Bias + Activate) → CopyOut

Pipeline:
  [CopyIn_A0,B0] → [Cube_0 | CopyIn_A1,B1] → [Vec_0 | Cube_1 | CopyIn_A2,B2] → ...
```

## Performance Optimization Checklist

1. **Align to 16**: Cube Unit works on 16×16 blocks — pad dimensions to multiples of 16
2. **Double buffer**: Overlap DMA with compute
3. **Avoid bank conflicts**: Access UB with correct stride
4. **Minimize sync**: Use queues instead of barriers where possible
5. **Maximize occupancy**: Ensure enough work per core
6. **Vectorize**: Use Vector Unit for element-wise ops (32 elements at a time)
