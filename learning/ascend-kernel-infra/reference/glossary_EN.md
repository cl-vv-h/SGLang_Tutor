[中文](./glossary.md) | [English](./glossary_EN.md)

# Glossary / 术语表

## Ascend NPU Hardware

| Term | Definition |
|---|---|
| AI Core | Ascend NPU's compute unit (Da Vinci architecture) |
| AIC | AI Core — general compute core |
| AIV | AI Vector Core — vector compute unit |
| Cube Unit | Matrix multiply-accumulate unit within AI Core |
| Vector Unit | Element-wise/SIMD unit within AI Core |
| Scalar Unit | Scalar/control flow unit within AI Core |
| MTE | Memory Transfer Engine — DMA unit |
| HBM | High Bandwidth Memory — main device memory (Global Memory / GM) |
| L2 | On-chip L2 cache between HBM and AI Cores |
| L1 | Per-core L1 buffer |
| UB | Unified Buffer — per-core scratchpad (L0 equivalent) |
| HCCS | Huawei Cache Coherence System — inter-chip interconnect |
| RoCE | RDMA over Converged Ethernet — cross-node networking |

## Programming Models

| Term | Definition |
|---|---|
| SPMD | Single Program Multiple Data — same code, different data per core |
| Program | The full computation specification |
| Grid | Distribution of work across physical cores |
| Tile | A chunk of data processed by one core in one iteration |
| BlockDim | Number of AI Cores allocated for a kernel launch |
| Tiling | Partitioning computation into tiles for parallel execution |
| Pipeline | Overlapping data movement with computation |
| Double Buffer | Two buffers alternating: compute on one while loading the next |
| CopyIn | Move data from GM → UB |
| Compute | Execute kernel on data in UB |
| CopyOut | Move results from UB → GM |
| Queue | Ascend C's synchronization primitive for data flow between stages |

## Data Structures

| Term | Definition |
|---|---|
| GlobalTensor | Ascend C: view of data in Global Memory (HBM) |
| LocalTensor | Ascend C: view of data in UB (on-chip buffer) |
| TPipe | Ascend C: pipeline abstraction connecting stages |
| TQue | Ascend C: queue connecting producer to consumer in a pipeline |
| Format FRACTAL_NZ | Ascend's specialized matrix layout for Cube Unit efficiency |

## SGLang-Specific

| Term | Definition |
|---|---|
| is_npu() | Runtime check: is current device Ascend NPU? |
| init_npu_backend() | Initialize NPU-specific runtime settings |
| NPUGraph | Ascend equivalent of CUDA Graph — capture & replay |
| HCCL | Ascend's collective communication library (like NCCL) |
| AscendTransferEngine | KV transfer engine for PD disaggregation on Ascend |
| kernel_ascend | HiCache backend that uses Ascend-specific kernels |
| ZBAL | Zero-Balance — memory optimization for Ascend |
| ACL | Ascend Compute Language — CANN runtime API |
| ACLNN | ACL Neural Network — operator library |
