# 主流大模型结构与实现：Qwen3-MoE

本专题以 Qwen3-MoE 的 SGLang 实现为主线，讲解现代 Decoder-only 大语言模型的结构、张量形状和推理数据流。Qwen3-MoE 同时包含 Transformer、Grouped-Query Attention、RoPE、KV Cache、Sparse MoE、Tensor Parallel 和 Expert Parallel 等主流组件。

## 本专题文件

| 文件 | 内容 |
|---|---|
| [01-decoder-only-transformer.md](./01-decoder-only-transformer.md) | 从 token 到 logits 的整机结构、Decoder Layer、残差连接和 prefill/decode 数据流 |
| [02-gqa-attention-shapes.md](./02-gqa-attention-shapes.md) | QKV 投影、QK Norm、RoPE、GQA、因果注意力、KV Cache 的逐步形状变化 |
| [03-sparse-moe-routing.md](./03-sparse-moe-routing.md) | Router、Top-K、Dispatch、SwiGLU Expert、Combine 和 Expert Parallel 数据流 |
| [04-sglang-qwen3-moe-execution.md](./04-sglang-qwen3-moe-execution.md) | Qwen3-MoE 在 SGLang 中的类依赖、forward 调用链、扁平 token 布局和并行执行 |

## 统一符号

| 符号 | 含义 |
|---|---|
| `B` | batch 中的请求数量 |
| `S` | 单条等长序列的 token 数；变长 batch 中用 `S_i` 表示第 `i` 条长度 |
| `T` | 当前 forward 中实际计算的 token 总数，`T = sum(S_i)` |
| `H` | hidden size |
| `L` | Decoder Layer 数量 |
| `Nq` | Query head 数量 |
| `Nkv` | Key/Value head 数量 |
| `D` | head dimension |
| `V` | vocabulary size |
| `E` | routed expert 数量 |
| `K` | 每个 token 选中的 expert 数量 |
| `Ie` | 单个 expert 的中间维度 |
| `Ptp` | attention tensor parallel world size |
| `Pep` | expert parallel world size |

在标准实现中常见 `H = Nq * D`。GQA 满足 `Nq > Nkv`，每组 `Nq / Nkv` 个 Query heads 共享一组 K/V head。

## 阅读顺序

1. 先阅读整机结构，建立 `input_ids -> hidden_states -> logits` 的主路径。
2. 再展开 Attention，理解为什么主路径始终保持 `[T,H]`，内部却会分裂成 Q/K/V 和 head 维度。
3. 然后展开 MoE，理解 token 如何从 `[T,H]` 变成 `T*K` 条 expert route，再合并回 `[T,H]`。
4. 最后对照 SGLang 源码，观察数学结构如何映射成 fused projection、paged KV cache、TP/EP 通信和推理专用数据布局。

## 对应源码

- `python/sglang/srt/models/qwen3_moe.py`：Qwen3-MoE 的 Attention、Decoder Layer、Sparse MoE 和 Causal LM 封装。
- `python/sglang/srt/models/qwen2_moe.py`：`Qwen3MoeModel` 继承的 embedding、layer loop、final norm 和 pipeline parallel 主干。
- `python/sglang/srt/layers/radix_attention.py`：模型层与 attention backend、KV Cache 之间的接口。
- `python/sglang/srt/layers/moe/`：Top-K、FusedMoE、dispatch/combine 和 expert parallel 实现。
- `python/sglang/srt/layers/linear.py`：QKV、Row Parallel、Column Parallel 等并行线性层。
