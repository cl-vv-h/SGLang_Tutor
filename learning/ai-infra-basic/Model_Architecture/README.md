**中文** | [English](./README_EN.md)

# 主流大模型架构与实现原理

本专题独立讲解大语言模型本身的计算结构，不依赖任何训练或推理框架。内容从标准 Transformer 出发，依次展开 Attention、KV Cache、Sparse MoE、Multi-head Latent Attention、状态空间模型及代表性架构家族，并持续跟踪张量的形状变化和数据依赖。

## 本专题文件

| 文件 | 内容 |
|---|---|
| [01-decoder-only-transformer.md](./01-decoder-only-transformer.md) | Decoder-only Transformer 从 token 到 logits 的完整主干、残差结构和推理数据流 |
| [02-gqa-attention-shapes.md](./02-gqa-attention-shapes.md) | MHA/MQA/GQA、QKV 投影、QK Norm、RoPE、因果 Attention 和 KV Cache |
| [03-sparse-moe-routing.md](./03-sparse-moe-routing.md) | Router、Top-K、Dispatch、SwiGLU Expert、Combine 和 Expert Parallel |
| [04-multi-head-latent-attention.md](./04-multi-head-latent-attention.md) | MLA 的低秩压缩、解耦 RoPE、吸收矩阵和压缩 KV Cache 原理 |
| [05-architecture-families.md](./05-architecture-families.md) | Encoder、Encoder-Decoder、Dense Decoder、MoE、MLA+MoE 与 SSM Hybrid 的结构比较 |

## 统一符号

| 符号 | 含义 |
|---|---|
| `B` | batch size |
| `S` | sequence length |
| `T` | 变长序列打包后的 token 总数 |
| `H` | hidden size |
| `L` | layer 数量 |
| `Nq` | Query head 数量 |
| `Nkv` | Key/Value head 数量 |
| `D` | head dimension |
| `V` | vocabulary size |
| `E` | routed expert 数量 |
| `K` | 每个 token 选择的 expert 数量 |
| `I` | Dense FFN intermediate size |
| `Ie` | 单个 expert intermediate size |
| `Dc` | MLA 的 KV latent compression dimension |

标准 Multi-Head Attention 常满足 `H=Nq*D`；GQA 满足 `Nq>Nkv`；Sparse MoE 满足 `K<<E`；MLA 使用 `Dc` 维 latent state 替代逐 head 保存完整 K/V。

## 阅读顺序

1. 从 Decoder-only Transformer 建立 `token -> hidden states -> logits` 的主路径。
2. 展开 Attention，理解 token 之间如何交换信息，以及 MHA、GQA 和 KV Cache 的空间代价。
3. 展开 Sparse MoE，理解参数如何通过 Top-K 路由按 token 稀疏激活。
4. 学习 MLA，理解低秩 latent 表示如何改变 Attention 的投影结构和缓存状态。
5. 对比主要架构家族，建立“序列混合模块 + 通道变换模块 + 状态形式”的统一分析框架。
