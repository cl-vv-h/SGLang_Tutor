# Gated Delta Network

**简体中文** | [English](../../../en/ai-infra-basic/Gated_Delta_Network/README.md)

GDN 在本专题中指 **Gated Delta Network / Gated DeltaNet**，也就是 Qwen3-Next 等混合架构中使用的一类线性注意力层。它不是图像压缩里的 Generalized Divisive Normalization。GDN 的核心是用一个固定大小的 recurrent state 取代随上下文长度增长的 KV Cache：每个 token 根据 `q/k/v` 读取和写入状态，`g` 控制旧状态遗忘，`beta` 控制新信息写入强度。

## 学习顺序

| 顺序 | 文件 | 重点问题 |
|---|---|---|
| 1 | [01-gdn-math-and-state.md](./01-gdn-math-and-state.md) | GDN 的数学公式、每个变量含义、state update 为什么叫 delta rule |
| 2 | [02-gdn-layer-dataflow-and-shapes.md](./02-gdn-layer-dataflow-and-shapes.md) | 从 hidden states 到 `q/k/v/z/a/b`、conv、gating、state、output 的端到端 shape |
| 3 | [03-gdn-training-and-serving.md](./03-gdn-training-and-serving.md) | 哪些参数可训练、训练怎么做、prefill/decode/target verify 如何执行 |

## 一张总览图

![GDN layer dataflow](./assets/gdn-layer-dataflow.svg)

## 先记住的结论

| 问题 | 结论 |
|---|---|
| GDN 是 attention 吗 | 它是线性注意力/状态空间风格的 attention 替代层，通过 recurrent state 表示历史信息。 |
| GDN 的状态是参数吗 | 不是。`S_t` 或 `ssm_states` 是每个请求运行时缓存，类似 Mamba state，不被 optimizer 更新。 |
| 哪些东西可训练 | 输入投影、短卷积、`A_log`、`dt_bias`、output gate norm、out projection，以及模型其他 MLP/MoE/embedding 等权重。 |
| `a`、`b` 是参数吗 | 不是。它们是由 hidden states 经线性投影得到的 token 级激活；产生它们的投影权重是参数。 |
| `g`、`beta` 是参数吗 | 不是。`g = -exp(A_log) * softplus(a + dt_bias)`，`beta = sigmoid(b)`，它们是由参数和激活计算出来的控制信号。 |
| GDN 为什么适合长上下文 | 每层每请求状态大小约为 `num_value_heads * value_dim * key_dim`，不随历史 token 数线性增长。 |
| KDA 有何不同 | GDN 对每个 token/head 使用一个保留率标量；Kimi Delta Attention 在 key 通道上使用向量，因此状态各列可以独立衰减。 |

读完三个章节后，可继续学习 [Kimi Delta Attention](../Model_Architecture/09-kimi-delta-attention.md)，理解细粒度门控如何改变递归、kernel 与 serving 状态契约。

## 与 SGLang 源码阅读的关系

| 源码位置 | 作用 |
|---|---|
| `python/sglang/srt/models/qwen3_next.py` | `Qwen3GatedDeltaNet` 定义投影、卷积、`A_log/dt_bias`、norm 和 out projection |
| `python/sglang/srt/layers/radix_linear_attention.py` | 统一的 linear attention 层入口 |
| `python/sglang/srt/layers/attention/linear/gdn_backend.py` | GDN backend，区分 prefill、decode、target verify |
| `python/sglang/srt/layers/attention/fla/fused_gdn_gating.py` | 计算 `g` 和 `beta` 的 fused kernel |
| `python/sglang/srt/layers/attention/fla/fused_recurrent.py` | decode recurrent update 与 packed decode fast path |
| `python/sglang/srt/layers/attention/fla/chunk_fwd.py` | prefill chunk gated delta rule 的并行化核心 |
| `python/sglang/srt/configs/qwen3_next.py` | GDN 相关 head 数、head dim、conv kernel、hybrid layer 配置 |

## 参考资料

- [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464)
- [Qwen3-Next model card](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)
- [Qwen blog: Qwen3-Next](https://qwenlm.github.io/blog/qwen3-next/)
- [Qwen blog: Qwen3-Next, NVIDIA Blackwell, and FlashQLA](https://qwenlm.github.io/blog/qwen3-next-flashqla/)
- [Flash Linear Attention project](https://github.com/fla-org/flash-linear-attention)
- [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)
