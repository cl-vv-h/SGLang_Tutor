# Parallel Strategy 教学入口

**简体中文** | [English](../../../en/ai-infra-basic/Parallel_Strategy/README.md)

这个目录整理 LLM 推理中常见的多卡并行策略。主讲义是 [tutorial.md](./tutorial.md)，配套 Python demo 用更小的张量和模型结构模拟 DP、TP、PP、SP/CP、EP 的切分对象和通信模式。

## 文件说明

| 文件 | 主题 | 核心问题 |
|---|---|---|
| [tutorial.md](./tutorial.md) | 并行策略总讲义 | DP/TP/PP/SP/EP 分别切什么、解决什么问题、通信代价是什么 |
| [01-dp-attention-principles.md](./01-dp-attention-principles.md) | DP Attention 原理 | 请求如何分到 Attention 子组，FFN/MoE 如何切换布局，partial/replica、padding、IDLE 和 KV 所有权如何保持正确 |
| [dp_attention_layout_demo.py](./dp_attention_layout_demo.py) | DP Attention CPU 模拟 | SUM_LEN/MAX_LEN gather、replica 去重与 reduce-scatter 返回为何数学等价；仅依赖标准库 |
| [dp_inference_demo.py](./dp_inference_demo.py) | Data Parallelism | 请求或 batch 维度如何分发到多个完整模型副本 |
| [tp_inference_demo.py](./tp_inference_demo.py) | Tensor Parallelism | Linear/Attention/FFN 的矩阵如何切到多个 rank |
| [pp_inference_demo.py](./pp_inference_demo.py) | Pipeline Parallelism | Transformer 层如何分段，hidden states 如何跨 stage 传递 |
| [sp_inference_demo.py](./sp_inference_demo.py) | Sequence/Context Parallelism | 长序列或长上下文如何按 sequence/context 维度拆分 |
| [ep_moe_demo.py](./ep_moe_demo.py) | Expert Parallelism | MoE token dispatch、expert 执行和结果 combine 如何配合 |

## 推荐阅读顺序

1. 先读 [tutorial.md](./tutorial.md) 的总览表，记住一句话：DP 切请求，TP 切矩阵，PP 切层，SP/CP 切序列或上下文，EP 切专家。
2. 按 `dp -> tp -> pp -> sp -> ep` 的顺序读 demo。这个顺序从最容易理解的横向扩容，逐步走向通信更复杂的并行。
3. 每读完一个策略，都回答三个问题：它切的对象是什么；它降低了哪类瓶颈；它引入了哪类通信。
4. 如果要理解 SGLang 的 `--enable-dp-attention`，继续读 [DP Attention 原理](./01-dp-attention-principles.md)，运行 CPU 布局模拟，再读 [端到端源码走读](../../sglang-source-reading/05-layer-communication/03-dp-attention-code-walkthrough.md)。

## 和 SGLang 的连接点

- TP/PP/DP 决定了 Scheduler 和 ModelRunner 进程组如何组织 rank。
- EP 与 MoE 模型相关，会引入 token dispatch/combine 和 all-to-all 通信。
- SP/CP 主要服务长上下文，和 KV Cache、attention backend、decode 阶段显存压力关系很强。
- 实际 serving 通常会组合多种策略，例如 TP + DP、TP + EP 或 PD disaggregation + TP。
- DP Attention 在同一个模型 world 内组合请求并行、Attention TP 和 FFN/MoE 并行；与普通 DP 的完整模型副本模式不同，`tp_size=4,dp_size=2` 的计算设备数也可能不同。
