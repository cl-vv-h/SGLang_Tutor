# Glossary / 术语表

## Framework, Runtime, and C++ Namespaces

| Term | Definition |
|---|---|
| C++ Namespace | A scope used to organize C++ names and avoid collisions. `A::B` looks up `B` in scope `A`; the namespace itself does not allocate memory or prove Host/Device placement |
| `at::` / ATen | PyTorch's foundational C++ tensor/operator namespace. `at::Tensor` is a metadata and storage-lifetime handle whose element storage may reside on an NPU |
| `c10::` / C10 | PyTorch core namespace for backend-neutral types and utilities such as `optional`, `Device`, `DeviceType`, `Scalar`, and `Stream` |
| `torch::` | PyTorch C++ frontend and extension-registration namespace, including `torch::Library` and `torch::nn`; normal frontend headers expose ATen Tensor names into this namespace |
| `c10_npu::` | torch_npu NPU core-adaptation namespace for current streams, devices, and guards; it is a top-level name, not `c10::npu` |
| `at_npu::native::` | torch_npu native/execution-framework namespace, including `OpCommand`; it belongs to Host backend implementation rather than the Ascend C Device API |
| `platform_ascendc::` | CANN Host platform/tiling namespace for querying AIC/AIV counts and UB/L1/L0 resources |
| `AscendC::` | CANN Ascend C programming namespace, primarily used in Device kernels for tensors, pipes, queues, movement, computation, and synchronization |
| Dispatcher Namespace | A logical PyTorch operator-registry namespace. `TORCH_LIBRARY_FRAGMENT(npu, m)` corresponds to Python `torch.ops.npu`, but is not C++ `namespace npu` and does not prove repository ownership |
| Anonymous Namespace | `namespace { ... }` creates translation-unit-local C++ scope, often used to hide registration helper symbols |
| ACL/ACLRT Name Prefix | C-style APIs use prefixes such as `aclrtStream` and `aclrtMemcpy`; `aclrtStream` is a complete type name, not `aclrt::Stream` |
| `ge::` | CANN Graph Engine types such as `DataType`, `Format`, and `graphStatus`, mainly used for Host operator description and validation |
| `gert::` | CANN Graph Engine Runtime descriptors and contexts; `gert::Tensor` is not `at::Tensor` |
| `optiling::` | Common namespace for Ascend operator tiling classes, validation, and tiling helpers |
| `matmul_tiling::` | CANN Host MatMul tiling helpers that derive matrix execution parameters from shape, layout, dtype, and hardware |
| `AscendC::MicroAPI::` | Lower-level Ascend C Device API for Vector registers and micro-instruction control; `RegTensor`/`MaskReg` are distinct from UB `LocalTensor` |
| Namespace Alias | `namespace py = pybind11;` creates a short alias; `py::arg` and `pybind11::arg` name the same declaration without conversion or copying |

## Streams, Events, and Asynchronous Execution

| Term | Definition |
|---|---|
| Stream | An ordered Host/runtime submission sequence for kernel launches, asynchronous copies, Events, and other device tasks; one Stream has implicit order but is not bound one-to-one to a CPU thread or AI Core |
| Default Stream | The fixed default Stream made available for a device/context; “default” is an identity and does not mean it is always the current Stream |
| Current Stream | The submission target selected by the framework for the current device and Host execution context; PyTorch NPU operators and custom wrappers normally inherit it |
| Allocation / Creation Stream | The Stream associated with a storage block when the caching allocator allocates it; the allocator knows this Stream but not every non-creation-Stream use |
| `c10::Stream` | PyTorch C10's backend-neutral Stream identity value; it carries device/Stream identity without being a native CANN queue handle |
| `c10_npu::NPUStream` | torch_npu's Host C++ wrapper for an NPU Stream; `.stream(false)` exposes the underlying `aclrtStream` |
| `aclrtStream` | CANN Runtime's native opaque Stream handle passed to asynchronous copies, kernels, and ACLNN launch interfaces |
| Event | A runtime progress marker; an Event recorded in a Stream completes after preceding work, and another Stream can wait on it to establish cross-Stream order |
| `wait_event` / `wait_stream` | Enqueues a dependency that makes future work in one Stream wait for another Stream's progress, normally without blocking Host |
| Stream Synchronize | Blocks Host until work previously submitted to a Stream completes; it is coarser than Event-based ordering and overuse destroys overlap |
| `record_stream` | Tells the caching allocator that storage is asynchronously used by a specified Stream, primarily a non-allocation Stream, and must not be reused early; same-allocation-Stream-only use normally needs no extra record. It is not an execution dependency or Host synchronization |
| Graph Capture / NPUGraph | Captures a relatively stable launch sequence on a capture Stream for replay; the graph describes replayable work, while Streams carry submission and ordering |

## Ascend NPU Hardware

| Term | Definition |
|---|---|
| AI Core | Ascend NPU's compute unit (Da Vinci architecture) |
| AIC | Cube Core in an AI Core separation-mode group |
| AIV | Vector Core in an AI Core separation-mode group |
| Cube Unit | Matrix multiply-accumulate unit within AI Core |
| Vector Unit | Element-wise/SIMD unit within AI Core |
| Scalar Unit | Scalar/control flow unit within AI Core |
| MTE | Memory Transfer Engine — DMA unit |
| HBM | High Bandwidth Memory — main device memory (Global Memory / GM) |
| L2 | On-chip L2 cache between HBM and AI Cores |
| L1 | On-chip staging/reuse buffer, commonly serving larger Cube tiles |
| L0A/L0B | Near-Cube buffers for left/right matrix operand blocks |
| L0C | Cube accumulator/result buffer |
| UB | Unified Buffer — main on-chip Vector input/output/temporary workspace; it is not an alias for L0A/L0B/L0C |
| A1/B1 | Ascend C logical locations for larger left/right Cube operands, commonly mapped to L1 |
| C1 | Logical bias-input location mapped to L1 or UB depending on product; not Cube output |
| A2/B2 | Near-Cube logical locations for left/right blocks, commonly mapped to L0A/L0B |
| C2 | Near-compute logical bias-input location mapped to BT or L0C depending on product |
| CO1/CO2 | Cube-output logical stages: block-wise L0C result and final GM/UB result, respectively, subject to product mapping |
| HCCS | Huawei Cache Coherence System — inter-chip interconnect |
| RoCE | RDMA over Converged Ethernet — cross-node networking |

## Programming Models

| Term | Definition |
|---|---|
| JIT | Just-In-Time compilation: compile a kernel variant when an uncached specialization is first needed, then reuse matching in-process or disk cache entries |
| SPMD | Single Program Multiple Data — same code, different data per core |
| Program | The full computation specification |
| Grid | Logical set of Triton program instances; runtime maps them onto physical cores |
| Tile | A chunk of data processed by one program instance or one of its internal iterations |
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
| LocalTensor | Ascend C: typed view of on-chip Local Memory such as UB/L1/L0 according to its logical position |
| TPosition | Ascend C logical data-path location such as VECIN, A1, A2, C1, C2, CO1, or CO2 |
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
