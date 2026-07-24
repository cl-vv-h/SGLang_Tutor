[中文](./README.md) | [English](./README_EN.md)

# Execution Graph

From computation graph concepts to CUDA/NPU Graph, `torch.compile`, static shape reuse, and replay dataflow in LLM serving.

## Key Files

| File | Content |
|---|---|
| [01-what-is-graph.md](./01-what-is-graph.md) | Computational graph fundamentals: nodes, edges, forward/backward, static vs dynamic shapes |
| [02-graph-execution-dataflow.md](./02-graph-execution-dataflow.md) | CUDA Graph capture/replay, NPUGraph, piecewise graph, and static batch dataflow |

## Why Graphs Matter for LLM Serving

- **Decode phase**: Each step processes only 1 new token per request, but the model forward involves many small kernel launches.
- **Kernel launch overhead**: Without graphs, each small operation requires a CPU→GPU kernel launch, adding significant latency.
- **Graph capture**: Record the entire forward pass once, then replay it — eliminating per-step kernel launch overhead.

## SGLang Integration

- `ModelRunner.init_device_graphs()` captures decode graphs
- `ModelRunner.init_piecewise_cuda_graphs()` captures finer-grained segment graphs
- `NPUGraph` and `torch.npu.graph()` handle Ascend NPU graph capture
- Shape padding ensures static shapes for graph replay compatibility
