[中文](./01-what-is-graph.md) | [English](./01-what-is-graph_EN.md)

# What is a Computational Graph

## 1. From Eager Execution to Graph Representation

In eager execution (PyTorch default), each operation executes immediately:

```python
a = x + y        # CPU immediately computes x+y
b = a * z        # CPU immediately computes a*z
c = matmul(b, W) # CPU immediately launches matmul kernel
```

This means one kernel launch per operation, each requiring CPU→GPU communication.

A computational graph separates **definition** from **execution**:

```text
Define:     c = matmul((x + y) * z, W)    # Just record ops, don't execute
Execute:    Run the entire graph at once    # Optimize and launch in one shot
```

## 2. Static vs Dynamic Graphs

| Property | Static Graph | Dynamic Graph (Eager) |
|---|---|---|
| Shape flexibility | Fixed shapes, known at graph-build time | Dynamic, changes each call |
| Optimization | Aggressive fusion, memory planning | Limited cross-op optimization |
| Debugging | Harder — ops aren't executed line-by-line | Easier — immediate results |
| Overhead | Build once, replay many times | Build + execute every call |

## 3. Why Graphs Matter for LLM Serving

LLM decode is a prime graph use case:

- **Identical structure every step**: Same layers, same operations, same dataflow
- **Small per-step work**: 1 new token per request, but many small kernel launches
- **Kernel launch overhead dominates**: Without graphs, CPU→GPU dispatch eats significant latency

By capturing the decode forward pass as a graph:
```text
Step 1: Build graph (capture all ops once)
Steps 2..N: Replay graph (zero CPU overhead per step)
```

## 4. CUDA Graph Mechanics

```python
# Capture phase
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    output = model(input_static)  # All ops recorded, not executed

# Replay phase
input_static.copy_(new_input)
g.replay()                         # Execute entire recorded graph
output_static.copy_(result)
```

Key constraints:
- All input/output tensors must have fixed addresses (static buffers)
- Shapes must match exactly between capture and replay
- GPU memory pointers cannot change

## 5. NPU Graph (Ascend)

Ascend NPU has analogous graph capture:

```python
g = torch.npu.NPUGraph()
with torch.npu.graph(g, pool=memory_pool):
    output = model(input_static)
g.replay()
```

NPU Graph uses `NPUGraph` and dedicated memory pools, following similar static-shape constraints.

## 6. Piecewise Graph

Full-model graphs have limitations:
- Large memory footprint (entire model's intermediate tensors pinned)
- Shape changes break the graph (dynamic batch sizes)

**Piecewise graph** splits the model into smaller graph segments:

```text
Full model: [Embed -> Layer0 -> Layer1 -> ... -> LayerN -> LM Head]
Piecewise:  [Graph0: Embed+Layer0] [Graph1: Layer1] ... [GraphN: LM Head]
```

Benefits:
- Each piece is smaller, faster to capture
- Pieces can be independently optimized
- Non-graph operations between pieces allow dynamic shapes

## 7. SGLang Graph Architecture

- `ModelRunner.init_device_graphs()` — Captures full decode graphs
- `ModelRunner.init_piecewise_cuda_graphs()` — Captures piecewise graphs
- `_forward_raw()` checks `forward_mode` and tries graph replay before falling back to eager
- `NPUGraph` handles Ascend NPU graph capture via `npu_piecewise_backend.py`
