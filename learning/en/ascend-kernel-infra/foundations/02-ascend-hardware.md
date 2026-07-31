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

Do not merge L0 and UB into one memory level. They serve different compute paths:

| Storage | Main role |
|---|---|
| GM/HBM | Full tensors, weights, KV cache, and kernel inputs/outputs |
| L2 | Shared cache on the path to GM; normally not allocated as an Ascend C tensor |
| L1 Buffer | On-chip staging and reuse of larger Cube tiles |
| L0A/L0B | Near-Cube buffers for the left and right matrix operands |
| L0C | Cube accumulation/result buffer |
| UB | Main on-chip workspace for Vector inputs, outputs, and temporaries |

The two common paths are:

```text
Vector: GM -> UB -> Vector -> UB -> GM
Cube:   GM -> L1 -> L0A/L0B -> Cube -> L0C -> output path -> GM
```

Capacities, legal transfers, alignment, and some logical-to-physical mappings vary by Ascend product. Query or consult the target product instead of treating a capacity from one model as universal.

## `A1/B1/C1`, `A2/B2/C2`, and `CO1/CO2`

These names are Ascend C `TPosition` logical locations for Cube computation, not chip generations or physical-core names. Start from:

```text
Output = A @ B + Bias
```

The letters describe operand roles:

- `A`: left matrix;
- `B`: right matrix;
- `C`: bias input;
- `CO`: Cube output.

The suffix describes a logical stage:

| TPosition | Data role | Common physical mapping |
|---|---|---|
| `A1` | Larger left-matrix tile | L1 Buffer |
| `B1` | Larger right-matrix tile | L1 Buffer |
| `C1` | Bias before near-compute blocking | L1 or UB, product-dependent |
| `A2` | Near-Cube left block | L0A |
| `B2` | Near-Cube right block | L0B |
| `C2` | Near-compute bias block | BiasTable/BT or L0C, product-dependent |
| `CO1` | Block-wise Cube accumulation/result | L0C |
| `CO2` | Final matrix result/output stage | GM or UB, product-dependent |

Conceptually:

```text
A: GM -> A1(L1) -> A2(L0A) --+
B: GM -> B1(L1) -> B2(L0B) --+-> Cube/Mmad -> CO1(L0C) -> CO2/output -> GM
Bias: GM -> C1(L1 or UB) -> C2(BT or L0C) --+
```

`C1/C2` are not abbreviations for `CO1/CO2`: the former are bias-input roles; the latter are output roles. A no-bias matmul does not need C1/C2, and a high-level Ascend C Matmul API may manage several stages internally.

See the official [Ascend C glossary](https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_00013.html) and [logical-to-physical TPosition mapping](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0004.html).

## Compute Units

| Unit | Purpose |
|---|---|
| Cube | Matrix multiply-accumulate |
| Vector | Element-wise operations, reductions, activations, and layout work |
| Scalar | Control flow, address calculation, and instruction issue |
| MTE | Moving data among GM and on-chip storage |

## Key Hardware Constraints

1. Cube instructions impose dtype-, layout-, and tile-alignment constraints.
2. UB/L1/L0 capacity is limited, so kernels must tile.
3. Data movement is part of the performance model, not free bookkeeping.
4. Double or multiple buffering can overlap movement and computation.
5. Every exact core count, capacity, and supported path must be checked for the target product.
