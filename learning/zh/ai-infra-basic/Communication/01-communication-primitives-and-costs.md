# 01. 通信原语、张量布局与成本模型

**简体中文** | [English](../../../en/ai-infra-basic/Communication/01-communication-primitives-and-costs.md)

## 1. 从两个 rank 的张量开始

rank 是通信组里的进程编号；一进程一卡很常见，但 rank 不是物理 device id。一个进程可以同时属于 TP、EP、PP 等不同组，同一个 collective 必须在约定的组内解释。

下面每个 rank 持有长度为 4 的向量。以 sum 为规约操作：

```text
rank 0: [1, 2, 3, 4]
rank 1: [10, 20, 30, 40]

AllReduce:
rank 0: [11, 22, 33, 44]
rank 1: [11, 22, 33, 44]

ReduceScatter (equal contiguous shards):
rank 0: [11, 22]
rank 1: [33, 44]

AllGather of those ReduceScatter outputs:
rank 0: [11, 22, 33, 44]
rank 1: [11, 22, 33, 44]
```

AllGather 是按 rank 顺序拼接，不做求和。若直接 AllGather 原始输入，结果长 8；只有先对相同逻辑张量做 ReduceScatter、再按同一分片顺序 AllGather，才在数学上恢复 AllReduce。浮点规约树不同可能带来舍入差异。[NCCL collective 语义](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)

| 原语 | 输入 → 输出 | 高频用途 |
|---|---|---|
| Broadcast | 一个 root 的数据 → 所有参与 rank | 初始化、复制小张量 |
| Reduce | 各 rank 对应元素规约 → root | 汇总指标 |
| AllReduce | 各 rank 对应元素规约 → 每个 rank | TP 局部和、训练梯度同步 |
| AllGather | 各 rank 分片 → 每个 rank 的完整拼接 | SP/TP 布局变换、训练参数收集 |
| ReduceScatter | 对应元素规约 → 每个 rank 一片 | 分片结果、训练梯度分片 |
| AllToAll / AllToAllV | 按目的 rank 切包 → 按来源 rank 收包 | EP token 交换；V 表示收发长度可变 |
| Send/Recv | 指定源与目的之间传递 | PP stage 激活、点对点协议 |
| Barrier | 等待所有参与者到达 | 初始化、调试；不替代数据依赖管理 |

## 2. AllToAllV：先数清楚每个方向有多少 token

令 `counts[src][dst]` 为源 rank 发给目标 rank 的记录数。例如：

```math
\mathrm{counts}=\left[\left[1,2\right],\left[3,0\right]\right]
```

```math
\text{rank 0}: \mathrm{input\_splits}=\left[1,2\right], \mathrm{output\_splits}=\left[1,3\right]
```

```math
\text{rank 1}: \mathrm{input\_splits}=\left[3,0\right], \mathrm{output\_splits}=\left[2,0\right]
```

rank 0 的输入有 3 条记录，输出有 4 条；rank 1 的输入有 3 条，输出有 2 条。每个发送方向必须匹配接收端对应来源的长度、dtype 与记录宽度。接收 splits 是 counts 的列，不是把自己的发送 splits 原样复用。零 token rank 也必须遵守协议，不能擅自跳过 collective。

MoE 的“记录”通常包括 hidden vector、expert id、原 token 索引和可能的量化 scale。传输 bytes 不能只按 token 个数估计。

## 3. 并行策略为什么产生这些通信

| 场景 | 分片导致的依赖 | 典型通信 |
|---|---|---|
| row-parallel linear | $`Y = \sum_{r} X_{r} W_{r}`$ | 局部结果 AllReduce，或下游接受分片时 ReduceScatter |
| column-parallel linear | `Y_r = X W_r` | 下游需要完整 Y 时 AllGather；可继续分片时不一定通信 |
| PP 推理 | 下一 stage 需要上一 stage 的激活 | Send/Recv |
| EP MoE | expert 分布在不同 rank | dispatch → expert compute → combine |
| CP attention | 本地 query 需要其他位置的 K/V 或中间结果 | ring P2P、gather 或其他算法，取决于 attention 实现 |
| DP 训练 | 副本梯度需要一致 | AllReduce / ReduceScatter |
| 独立 DP 推理 | 请求由不同副本各自处理 | 通常不需逐层同步梯度；共享 MoE 等组合另论 |
| PD 分离 | decode 需要 prefill 产生的 KV | 请求级数据块传输 |

这解释了为什么“用了多卡”并不等于“每层都做 AllReduce”。

## 4. 延迟模型：小包怕启动，大包怕带宽

先用一个可解释的近似：

```math
t_{\mathrm{transfer}}\approx n_{\mathrm{steps}}\cdot \alpha+\frac{\mathrm{bytes\_on\_critical\_link}}{B_{\mathrm{eff}}}
```

`alpha` 是每步启动/协议延迟，`B_eff` 是实际有效带宽；不要将所有网卡标称带宽直接相加。ring AllReduce 在均匀切块、通信均衡、忽略额外规约开销时，可估计为：

```text
P = participating ranks
M = full input bytes per rank
```

```math
\mathrm{bytes\_sent\_per\_rank}=\frac{2\cdot \left(P-1\right)}{P}\cdot M
```

```math
t_{\mathrm{ring}}\approx 2\cdot \left(P-1\right)\cdot \alpha+\frac{\mathrm{bytes\_sent\_per\_rank}}{B_{\mathrm{eff}}}
```

这里统计**单 rank 发送量**，不是全网字节量，也没有把发送与接收重复相加。ring 是一种算法；tree、分层算法、网络内规约会改变模型，不能把这个公式当成所有后端的实现。

算例：$`P=8`$、$`M=16 \mathrm{MiB}`$、$`\alpha=3 \mathrm{us}`$、$`B_{\mathrm{eff}}=50 \mathrm{GB}/s`$（十进制），每 rank 发送 28 MiB，估计约 **629 us**。若 $`M=4 \mathrm{KiB}`$，则约 **42.14 us**，几乎由启动项决定。以上是手工参数下的教学估算，不是硬件实测。

## 5. 重叠的边界

只有独立的计算与通信才可能重叠：

```math
\mathrm{serial}: t_{\mathrm{step}} = t_{\mathrm{compute}} + t_{\mathrm{comm}}
```

```math
\mathrm{ideal}: t_{\mathrm{step}} \ge  \mathrm{max}\left(t_{\mathrm{compute}}, t_{\mathrm{comm}}\right)
```

当下一层 GEMM 必须读取 AllReduce 输出时，二者存在真依赖；开两条 stream 不会消除它。重叠还可能争抢 SM/AICore、HBM、NIC，导致计算或通信各自变慢。最后应看端到端 TTFT、ITL、吞吐，而不是只看 trace 上两条彩色条相交。

## 6. 动手与自检

运行 [语义 demo](./communication_semantics_demo.py)，观察 AllGather 与 AllReduce 的不同输出，以及 ReduceScatter + AllGather 的等价结果。

1. 为什么 TP AllReduce 的元素必须对应同一批 token、同一输出位置？
2. 在 counts 算例中，rank 0 为什么接收 4 条记录？
3. 把 `P` 翻倍后，小包 ring AllReduce 为什么可能更慢？

下一讲：[链路、RDMA 与内存](./02-links-rdma-and-memory.md)。带宽指标的统计口径可进一步参考 [nccl-tests PERFORMANCE](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)。
