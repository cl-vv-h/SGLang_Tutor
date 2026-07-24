**中文** | [English](./README_EN.md)

# Execution Graph

这一章解释生产推理中常见的 execution graph 优化，例如 CUDA Graph、torch.compile、静态 shape replay。它们解决的不是模型数学问题，而是减少 CPU launch overhead、稳定 kernel 调度，并让 decode loop 更接近固定模板。

## 本专题文件

| 文件 | 内容 |
|---|---|
| [01-what-is-graph.md](./01-what-is-graph.md) | 从“计算图是什么”讲到 SGLang 中 CUDA/NPU/CPU Graph 的 capture/replay 实现 |
| [02-graph-execution-dataflow.md](./02-graph-execution-dataflow.md) | 详细拆解 graph replay 时哪些数据进入 graph、如何调度、数据如何在 buffer/KV cache/output 之间流转 |

如果你刚开始看 SGLang 里的 `graph_runner`、`cuda_graph`、`piecewise_cuda_graph`，建议先读 [01-what-is-graph.md](./01-what-is-graph.md)。如果已经知道 graph 是什么，但不清楚 graph 里到底跑了哪些数据，继续读 [02-graph-execution-dataflow.md](./02-graph-execution-dataflow.md)。

## 为什么需要 Execution Graph

Decode 阶段每步输入通常只有一个 token。如果每一步都由 Python/CPU 逐个 launch kernel，会出现明显开销：

```text
CPU 构造 batch
CPU launch embedding kernel
CPU launch attention kernel
CPU launch MLP kernel
CPU launch sampling kernel
...
下一轮重复
```

当 batch 不大时，CPU launch overhead 会变得很显眼。CUDA Graph 的思路是把一段固定形状的 GPU 操作 capture 起来，后续只做 replay。

## CUDA Graph 的基本条件

CUDA Graph 通常要求：

1. kernel 序列相对固定。
2. 输入输出 tensor 地址稳定。
3. shape 稳定或落在可复用 bucket 中。
4. graph capture 期间不能有不支持 capture 的动态 CPU/GPU 行为。
5. 随机数、通信、内存分配等操作要特别小心。

因此 serving 系统常把 decode batch pad 到固定 batch size bucket，例如 `1, 2, 4, 8, 16, 32, 64`。

## Shape Bucket

线上 batch size 每轮都可能变化。如果每个 batch size 都 capture 一个 graph，数量会爆炸；如果只用一个最大 graph，又会浪费大量 padding。

常见折中是 bucket：

```text
真实 batch = 13 -> 使用 batch bucket 16
真实 batch = 17 -> 使用 batch bucket 32
真实 batch = 1  -> 使用 batch bucket 1
```

这样 replay 稳定，但多出来的 padding token 不参与真实输出。

## torch.compile 和 CUDA Graph 的区别

`torch.compile` 更关注图级编译、算子融合、降低 Python overhead。CUDA Graph 更关注 capture 固定 kernel launch 序列并 replay。它们可以组合，但关注点不同。

```text
torch.compile:
    改善模型 forward 的图编译和融合

CUDA Graph:
    改善每轮 decode 的 kernel launch replay
```

## Serving 中的难点

1. Batch 是动态的，但 graph 喜欢静态形状。
2. KV Cache block table 是动态的，但 graph replay 要地址稳定。
3. Sampling 参数可能每个请求不同。
4. LoRA、grammar、speculative decoding 会引入额外动态路径。
5. 多卡通信如果参与 capture，需要更谨慎地管理 stream 和 communicator。

## 和 SGLang 的连接点

- Decode loop 是 CUDA Graph 最有价值的场景，因为它重复执行且每轮 shape 相似。
- Scheduler 需要把请求组织到合适的 batch bucket，减少 padding 浪费。
- ModelRunner 需要提前准备可复用输入 buffer。
- Attention backend 需要支持 graph replay 下的 KV Cache 访问。
- LoRA、speculative decoding、structured output 等功能可能影响 graph capture 可用性。

## 阅读任务

1. 说明为什么 prefill 不一定比 decode 更适合 CUDA Graph。
2. 解释 batch size bucket 如何在性能和浪费之间折中。
3. 列出三个会破坏 graph capture 稳定性的动态因素。
4. 思考：如果线上请求很多都带不同 LoRA adapter，graph replay 会遇到什么问题。
