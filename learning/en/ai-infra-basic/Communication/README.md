# AI Infra Communication: Tensor Semantics and GPU/NPU Data Paths

**English** | [简体中文](../../../zh/ai-infra-basic/Communication/README.md)

Communication in distributed inference includes at least three different jobs: combining TP partial results, moving MoE tokens to experts and back, and handing KV cache to another instance under PD disaggregation. Identify the payload, destination, and readiness contract before choosing a library.

Read this topic after [Parallel Strategy](../Parallel_Strategy/README.md), alongside [KV Transfer](../KV_Transfer/README.md). Existing Ascend walkthroughs cover concrete implementations; this topic supplies cross-platform concepts, equations, comparisons, and experimental methods.

## Reading Order

| File | Question You Should Be Able to Answer |
|---|---|
| [01-communication-primitives-and-costs.md](./01-communication-primitives-and-costs.md) | What do AllReduce, AllGather, ReduceScatter, and AllToAllV change? How do we estimate bytes and latency? |
| [02-links-rdma-and-memory.md](./02-links-rdma-and-memory.md) | Where do NVLink/HCCS, PCIe, IB/RoCE, RDMA, memory registration, and completion notifications fit? |
| [03-collective-libraries-gloo-nccl-hccl.md](./03-collective-libraries-gloo-nccl-hccl.md) | How do Gloo, NCCL, RCCL, HCCL, and `torch.distributed` relate? |
| [04-moe-deepep-and-zbal.md](./04-moe-deepep-and-zbal.md) | What do DeepEP dispatch/combine do, and how does SGLang NPU integrate ZBAL? |
| [05-kv-transfer-mooncake-and-nixl.md](./05-kv-transfer-mooncake-and-nixl.md) | How do Mooncake TE, Store, EP/PG, NIXL, and Ascend TransferEngine differ? |
| [06-sglang-paths-and-performance.md](./06-sglang-paths-and-performance.md) | How do we trace model calls into communication and diagnose slow ranks, links, and overlap? |
| [07-zbal-zero-buffer-source-dataflow.md](./07-zbal-zero-buffer-source-dataflow.md) | Source deep dive after Chapter 04: GVA/allocators, zero buffer, retained/avoided copies, HCCL delegation, and actual MoE address dataflow |
| [communication_semantics_demo.py](./communication_semantics_demo.py) | Verify reduction, concatenation, sharding, and MoE restoration with pure Python and no accelerator |

## Technology Map

These names are not mutually exclusive choices at one layer. The roles below describe this course's focus, not each project's entire feature set.

| Layer / Use | Technology | First Concept to Retain |
|---|---|---|
| Framework interface | `torch.distributed`, ProcessGroup | Ranks, communication groups, collectives, and asynchronous work handles |
| CPU collectives / coordination | Gloo | CPU tensor communication; SGLang's ordinary path also creates CPU coordination groups with it |
| Accelerator collectives | NCCL / RCCL / HCCL | NVIDIA GPUs, AMD GPUs, and Ascend NPUs respectively |
| Specialized MoE communication | DeepEP, Ascend DeepEP adaptations | Joint optimization of routing, dispatch, expert layout, and combine |
| NPU memory and communication | ZBAL / MemFabric | Allocator/operator cooperation; SGLang integrates both a backend and a DeepEP adapter |
| Asynchronous data transfer | Mooncake Transfer Engine, NIXL | Transfer registered memory blocks such as KV, weights, or embeddings |
| Distributed caching | Mooncake Store | Object, location, and cache-lifecycle management above transfer |
| Lower-level programming tools | UCX, NVSHMEM, MSCCL++ | Different tools for transport abstraction, one-sided access, and device communication |
| Physical links / networks | PCIe, NVLink/NVSwitch, HCCS, InfiniBand, RoCE | Topology, reachability, latency, and available bandwidth |

For example, one service might use Gloo for CPU coordination, HCCL for TP collectives, NPU DeepEP/ZBAL for MoE dispatch, and MemFabric TransferEngine for PD KV transfer. The installed versions and selected code paths determine which combinations actually run.

## Three Recurring Questions

1. **Semantics**: is the operation a sum, concatenation, routing permutation, or remote memory copy?
2. **Completion**: who guarantees Python return, device completion, consumer visibility, and scheduler readiness?
3. **Cost**: how many ranks participate, and how many useful bytes does each rank move? Are padding, quantization metadata, and slow-rank waits included?

## Running and Reading Conventions

Run from the repository root:

```bash
python3 learning/en/ai-infra-basic/Communication/communication_semantics_demo.py
```

The demo simulates semantics and estimates costs in one process; it **does not measure network performance**. Chapter 03 also includes a two-process Gloo example that can be saved and run with a distributed/Gloo-enabled PyTorch installation.

Sources checked: **2026-10-13**. SGLang call paths use this repository's `f249f7b` source as the baseline; relative links support reading alongside the code. Upstream communication libraries evolve quickly: in particular, do not mix DeepEP's `Buffer/Config` interface with newer `EPBuffer` APIs. Consult the linked official documentation for hardware matrices and installation versions.

The ZBAL deep dive was checked on **2026-10-14**. Read Chapter 07 after Chapter 04 to follow the actual allocator, ProcessGroup, and Ascend C kernels at a pinned MemFabric commit.

## Further Source Walkthroughs

- [LayerCommunicator, HCCL, and layer-boundary communication](../../sglang-ascend-npu/source-code-walkthrough/15-distributed-hccl-and-communication.md)
- [sgl-kernel-npu: DeepEP, HCCL, and MoE token paths](../../ascend-kernel-infra/sgl-kernel-npu/05-deepep-hccl-and-moe-kernel-path.md)
- [DeepEP low-latency and the layered A2 path](../../ascend-kernel-infra/sgl-kernel-npu/06-deepep-low-latency-and-layered-a2-path.md)

Each chapter links official documentation or source near the relevant claims. Start with Chapter 01, then select later chapters according to your TP, MoE, or PD problem.
