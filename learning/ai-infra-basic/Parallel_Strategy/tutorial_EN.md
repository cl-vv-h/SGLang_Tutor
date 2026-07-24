[中文](./tutorial.md) | [English](./tutorial_EN.md)

# Parallel Strategy Tutorial

## 1. Why Parallelism in LLM Inference

Modern LLMs (70B+ parameters in BF16) need ~140GB+ of memory just to store weights — more than a single GPU. Even when a model fits, parallelism improves throughput by distributing the workload.

## 2. Tensor Parallelism (TP)

**How it works**: Split individual weight matrices across GPUs.

```text
Standard: Y = X @ W          # W: [H, 4H] must fit on one GPU
TP:       Y = [X @ W_0, X @ W_1]  # W split column-wise across 2 GPUs
          Y = all_reduce(Y_parts)  # Combine results
```

**Memory**: Each GPU holds `1/TP_size` of each weight matrix
**Communication**: All-reduce after each matrix multiply (every layer)
**Best for**: Models that don't fit on a single GPU
**Trade-off**: High communication volume — needs fast interconnects (NVLink, HCCS)

## 3. Pipeline Parallelism (PP)

**How it works**: Split model layers across GPUs.

```text
GPU 0: Layers 0-7
GPU 1: Layers 8-15
GPU 2: Layers 16-23
GPU 3: Layers 24-31
```

Micro-batches flow through the pipeline. While GPU 2 processes micro-batch 1, GPU 1 processes micro-batch 2.

**Communication**: Send/receive activations between stages (moderate)
**Best for**: Very deep models, combined with TP
**SGLang**: PP is used alongside TP; `pp_group.is_last_rank` controls sampling

## 4. Data Parallelism (DP)

**How it works**: Replicate the entire model, split the batch.

```text
GPU 0: Model replica, processes batch[0:B/2]
GPU 1: Model replica, processes batch[B/2:B]
```

**Communication**: All-reduce gradients (training only — inference has no DP communication)
**Best for**: Increasing throughput when model fits on one GPU
**SGLang**: `dp_size` controls DP; attention DP enables attention across DP ranks

## 5. Sequence/Context Parallelism (SP/CP)

**How it works**: Split long sequences across GPUs for attention computation.

```text
GPU 0: Processes positions 0..S/2
GPU 1: Processes positions S/2..S
All-to-all communication for attention softmax
```

**Best for**: Very long context inference (128K+ tokens)
**Communication**: All-to-all during attention

## 6. Expert Parallelism (EP)

**How it works**: Distribute MoE experts across GPUs.

```text
GPU 0: Experts 0-7
GPU 1: Experts 8-15
...
Each GPU routes tokens to the GPU holding the target expert
```

**Communication**: All-to-all for token dispatch/combine
**Best for**: Large MoE models
**Trade-off**: Load imbalance if some experts receive disproportionate tokens

## 7. Combining Strategies

Real deployments combine multiple strategies:

```text
Example: 8 GPUs, 70B MoE model
- TP = 2 (split matrices across GPU 0-1 and GPU 2-3)
- PP = 2 (GPU 0-3: layers 0-39, GPU 4-7: layers 40-79)
- EP = 4 (experts distributed within each PP stage)
```

## 8. SGLang Code References

- `python/sglang/srt/distributed/parallel_state.py` — Group initialization
- `python/sglang/srt/managers/scheduler.py` — TP/PP/DP rank-aware scheduling
- `python/sglang/srt/layers/communicator.py` — Layer-level communication abstraction
