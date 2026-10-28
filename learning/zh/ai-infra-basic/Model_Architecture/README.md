# 主流大模型架构与实现原理

**简体中文** | [English](../../../en/ai-infra-basic/Model_Architecture/README.md)

本专题独立讲解大语言模型本身的计算结构，不依赖任何训练或推理框架。内容从标准 Transformer 出发，依次展开 Attention、KV Cache、Sparse MoE、Multi-head Latent Attention、稀疏/压缩注意力、递归线性注意力、状态空间模型及代表性架构家族，并持续跟踪张量形状、缓存/状态形式与数据依赖。

## 本专题文件

| 文件 | 内容 |
|---|---|
| [01-decoder-only-transformer.md](./01-decoder-only-transformer.md) | Decoder-only Transformer 从 token 到 logits 的完整主干、残差结构和推理数据流 |
| [02-gqa-attention-shapes.md](./02-gqa-attention-shapes.md) | MHA/MQA/GQA、QKV 投影、QK Norm、RoPE、因果 Attention 和 KV Cache |
| [03-sparse-moe-routing.md](./03-sparse-moe-routing.md) | Router、Top-K、Dispatch、SwiGLU Expert、Combine 和 Expert Parallel |
| [04-multi-head-latent-attention.md](./04-multi-head-latent-attention.md) | 从一次完整 decode 出发，推导 MLA 低秩压缩、解耦 RoPE、K/V 矩阵吸收、缓存字节数和全张量流 |
| [05-architecture-families.md](./05-architecture-families.md) | 用“Token Mixer + Channel Mixer + Runtime State”还原 Encoder、Decoder、MoE、MLA、SSM、Hybrid 与多模态架构 |
| [06-efficient-attention-landscape.md](./06-efficient-attention-landscape.md) | 按缓存宽度、访问位置、递归状态和执行 kernel 四条独立轴比较 MHA/GQA/MLA、SWA/DSA/CSA/HCA、GDN/KDA |
| [07-deepseek-sparse-attention.md](./07-deepseek-sparse-attention.md) | 区分 indexer 与主 Attention 两次打分，逐 shape 追踪 top-k、paged 地址翻译、MLA 稀疏主核和训练流程 |
| [08-compressed-sparse-attention.md](./08-compressed-sparse-attention.md) | 用压缩时间线和数值例子讲解 CSA 重叠压缩、HCA、局部 SWA、条目就绪条件、异构缓存与 NPU 适配 |
| [09-kimi-delta-attention.md](./09-kimi-delta-attention.md) | 从可改写键值记忆推导 KDA 的逐通道衰减与 delta rule，并追踪卷积、chunk prefill、融合 decode 和 serving 状态 |
| [10-manifold-constrained-hyper-connections.md](./10-manifold-constrained-hyper-connections.md) | 普通 Residual → HC → mHC：多流读写、双随机约束、Sinkhorn 手算、完整 tensor flow、稳定性边界与 CPU 自检 |

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
| `k_attn` | 稀疏注意力选择的历史条目数 |
| `m` | 序列压缩步长 |
| `w` | 局部滑动窗口长度 |
| `M` | 递归线性注意力状态矩阵 |

标准 Multi-Head Attention 常满足 $`H=\mathrm{Nq}\cdot D`$；GQA 满足 $`\mathrm{Nq} \gt \mathrm{Nkv}`$；Sparse MoE 满足 `K<<E`；MLA 使用 `Dc` 维 latent state 替代逐 head 保存完整 K/V；稀疏注意力通常满足 `k_attn<<S`；递归注意力保存 `M` 而不是 token 长度的缓存。

## 阅读顺序

1. 从 Decoder-only Transformer 建立 `token -> hidden states -> logits` 的主路径。
2. 展开 Attention，理解 token 之间如何交换信息，以及 MHA、GQA 和 KV Cache 的空间代价。
3. 进入 04 后先读每篇开头的“阅读地图”和贯穿例子；它们给出后续公式的全局数据流与统一 shape 坐标。
4. 先学 MLA，再学 DSA 与 CSA/HCA，沿“压缩每个位置 → 选择 token 位置 → 先压缩序列再选择”的路线阅读。
5. 再读 KDA，对比“保存可寻址历史条目”与“把历史折叠到固定矩阵”这两类根本不同的状态契约。
6. 回到高效 Attention 全景，用缓存宽度、访问位置数、状态增长方式与 kernel 执行四个问题横向比较。
7. 展开 Sparse MoE，再用架构谱系章节把 Token Mixer、Channel Mixer 与 Runtime State 拼回完整模型。
8. 残差连接单独读 10：学过 01 后即可进入，不要求先学完 Attention 变体；mHC 改的是子层连接，不是另一种 Attention。
