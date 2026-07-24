**中文** | [English](./01-decoder-only-transformer_EN.md)

# Decoder-only Transformer：从 Token 到下一个 Token

## 1. 模型解决的问题

Decoder-only 语言模型接收一个 token 序列：

```text
x_0, x_1, ..., x_(S-1)
```

并为每个位置计算下一个 token 的条件分布：

```text
P(x_t | x_0, x_1, ..., x_(t-1))
```

训练时可以并行计算序列中所有位置，但每个位置只能关注自己和更早的 token。推理时，prefill 一次处理 prompt，decode 每轮追加一个或少量 token。

现代生成式语言模型普遍采用 Decoder-only Transformer。注意力子层负责 token 间的信息交换，FFN 或 Sparse MoE 子层负责逐 token 的非线性特征变换。

## 2. 整机数据流

![Decoder-only Transformer 整机数据流](./assets/decoder-only-overview.svg)

整机计算可以写成：

```text
input_ids
  -> token embedding
  -> Decoder Layer 0
  -> Decoder Layer 1
  -> ...
  -> Decoder Layer L-1
  -> final RMSNorm
  -> LM Head
  -> logits
  -> sampling
  -> next token ids
```

各阶段的逻辑形状如下：

| 阶段 | 输入 | 输出 | 说明 |
|---|---:|---:|---|
| Token IDs | `[B,S]` 或 packed `[T]` | 同输入 | 整数词表索引 |
| Embedding | `[T]` | `[T,H]` | 按 token id 查询 embedding row |
| Decoder Layers | `[T,H]` | `[T,H]` | 重复 `L` 次，不改变主路径宽度 |
| Final RMSNorm | `[T,H]` | `[T,H]` | 对每个 token 的 hidden vector 独立归一化 |
| LM Head | `[M,H]` | `[M,V]` | `M` 是真正需要计算 logits 的 token 行数 |
| Sampling | `[M,V]` | `[B]` 或多个 token | 温度、Top-K、Top-P 等采样 |

训练通常令 `M=T`。在线自回归推理通常只需要每条请求最后一个有效位置的 logits，因此常见 `M=B`，无需为 prompt 的每一行都生成完整词表 logits。

## 3. 等长布局与 Packed Token 布局

训练教材通常把 hidden states 写成 `[B,S,H]`。当 batch 中的序列长度不同时，可以补齐到统一长度，也可以只打包有效 token：

```text
request 0: S_0 = 4
request 1: S_1 = 2
request 2: S_2 = 5
```

若全部 padding 到 `S_max=5`，形状是 `[3,5,H]`，其中 4 行是无效 padding。packed 表示只保留有效 token：

```text
T = 4 + 2 + 5 = 11
hidden_states.shape = [11,H]
```

packed 表示必须额外保存每条序列的起始偏移和长度。Attention 根据这些边界限制可见范围；线性层、归一化和 MoE 则可直接处理连续的 `[T,H]`。

两种表示的数学含义相同：

```text
dense training view: [B,S,H]
packed serving view: [T,H], T = sum_i S_i
```

## 4. Token Embedding

设 embedding 权重：

```text
W_embed: [V,H]
input_ids: [T]
```

embedding lookup 为每个 id 选择一行：

```text
X_0[t,:] = W_embed[input_ids[t],:]
X_0: [T,H]
```

这不是普通矩阵乘法，而是稀疏的行索引。每个 token 从一个整数变成长度为 `H` 的连续向量。

位置不直接加成一个 `[T,H]` 的绝对位置向量。Qwen3-MoE 在 Attention 内部通过 RoPE 把位置信息作用到 Q 和 K。

## 5. 一个 Decoder Layer

![Pre-Norm Decoder Layer](./assets/decoder-block.svg)

多数现代 Decoder-only 模型使用 pre-norm 残差结构。设该层输入为 `x_l [T,H]`：

```text
n_l = RMSNorm(x_l)                  [T,H]
a_l = CausalGQA(n_l)                [T,H]
r_l = x_l + a_l                     [T,H]
m_l = RMSNorm(r_l)                  [T,H]
f_l = SparseMoE(m_l)                [T,H]
x_(l+1) = r_l + f_l                 [T,H]
```

主路径始终是 `[T,H]`，使残差加法成立。Attention 和 MoE 内部可以改变维度，但返回前必须恢复 `[T,H]`。

### 5.1 RMSNorm

对单个 token vector `x in R^H`：

```text
rms(x) = sqrt(mean(x_i^2) + epsilon)
y_i = gamma_i * x_i / rms(x)
```

其中 `gamma [H]` 是可学习权重。RMSNorm 不混合不同 token，也不混合 hidden 维度的位置，只按每个 token 的 `H` 个元素计算尺度。

```text
input:  [T,H]
gamma:  [H]
output: [T,H]
```

### 5.2 残差连接

残差连接把子层输出加回主路径：

```text
r = x + sublayer(norm(x))
```

它要求两项形状完全一致。其作用是保留原始表示，并为深层网络提供短梯度路径。推理中它也决定了 kernel fusion 和临时 buffer 的组织方式。

## 6. Attention 与 MoE 的职责边界

### Attention：跨 token 混合

第 `t` 个 token 的 Attention 输出依赖可见历史 token：

```text
a_t = sum_j alpha_(t,j) * v_j, j <= t
```

因此 Attention 沿序列维度交换信息。它需要位置、因果 mask 和历史 KV Cache。

### MoE：逐 token 变换

MoE router 根据每个 token 的 hidden vector 选择 expert。每个 expert 内部的 MLP 不读取其他 token：

```text
f_t = sum_(e in TopK(t)) p_(t,e) * Expert_e(m_t)
```

MoE 沿特征维度做非线性变换；token 之间的耦合主要来自路由后的并行调度和通信，而不是数学上的 token attention。

## 7. Final RMSNorm 与 LM Head

经过 `L` 层后：

```text
X_L: [T,H]
X_final = RMSNorm(X_L): [T,H]
```

LM Head 权重：

```text
W_lm: [V,H]
```

若选出 `M` 行 hidden states：

```text
H_selected: [M,H]
logits = H_selected @ W_lm^T
logits: [M,V]
```

每个 `logits[m,v]` 是第 `m` 个待预测位置对词表 token `v` 的未归一化分数。概率为：

```text
P(v) = softmax(logits / temperature)[v]
```

`argmax` 是 greedy decoding；Top-K、Top-P 在候选子集上重新归一化后采样。

## 8. Prefill 数据流

设两条请求的新输入长度分别为 3 和 2：

```text
B = 2
extend lengths = [3,2]
T = 5
input_ids: [5]
positions: [5]
```

每层执行：

```text
hidden_states                 [5,H]
Q                             [5,Nq,D]
K_new, V_new                  [5,Nkv,D]
attention output              [5,H]
router logits                 [5,E]
top-k ids, top-k weights      [5,K], [5,K]
MoE output                    [5,H]
```

当前 K/V 被写入每层 KV Cache。Attention 对每条请求只访问自己的历史和当前可见前缀。packed 布局中虽然 token 行连续存放，但 metadata 阻止跨请求 attention。

## 9. Decode 数据流

普通 decode 每条活跃请求本轮输入一个 token：

```text
B_active = B
T = B
input_ids: [B]
positions: [B]
hidden_states: [B,H]
```

每层只计算新的 Q/K/V：

```text
Q_new: [B,Nq,D]
K_new: [B,Nkv,D]
V_new: [B,Nkv,D]
```

但每个 Query 需要读取该请求长度为 `L_ctx` 的历史 KV Cache：

```text
logical K_history: [L_ctx,Nkv,D]
logical V_history: [L_ctx,Nkv,D]
```

因此 decode 的算术输入 token 很少，历史 KV 读取却随上下文长度增长。Attention 常受 memory bandwidth 限制；MoE 则仍要执行 router 和选中 expert 的矩阵乘法。

## 10. 一个完整形状示例

以下数值用于展示形状关系，不代表某个特定 checkpoint：

```text
B=2, extend lengths=[3,2], T=5
H=4096, Nq=32, Nkv=8, D=128
E=64, K=4, Ie=1536, V=150000
```

单层主路径：

| 变量 | 形状 |
|---|---:|
| `input_ids` | `[5]` |
| `hidden_states` | `[5,4096]` |
| `q` | `[5,32,128]` |
| `k`, `v` | `[5,8,128]` |
| `attention_output` | `[5,4096]` |
| `router_logits` | `[5,64]` |
| `topk_ids`, `topk_weights` | `[5,4]` |
| dispatched expert rows | 最多 `5*4=20` 行，每行宽度 `4096` |
| `moe_output` | `[5,4096]` |

若只对每条请求最后一个位置计算 logits：

```text
selected_hidden: [2,4096]
logits: [2,150000]
sampled_ids: [2]
```

## 11. 参数、激活与运行时状态

理解模型执行时应区分三类数据：

| 类型 | 示例 | 生命周期 |
|---|---|---|
| 参数 | embedding、QKV、expert、LM Head 权重 | 模型加载后长期驻留 |
| 当前激活 | hidden states、Q、router logits、expert output | 一次 forward 或一个子层 |
| 跨轮状态 | 每层 KV Cache、请求长度、cache indices | 跨多个 decode step |

Decoder Layer 的输出不会作为独立历史永久保存；decode 重用的是每层 K/V，而不是过去所有层的 hidden states。

## 12. 关键结构约束

1. 残差主路径必须保持 `[T,H]`。
2. `Nq*D` 通常等于 `H`，而 `Nkv*D` 可以显著小于 `H`。
3. 因果 Attention 不能读取未来 token，也不能跨请求读取 token。
4. MoE 的参数量可随 expert 数量增长，但单 token 只激活 Top-K experts。
5. Prefill 计算多个新 token，decode 通常每请求只计算一个新 token。
6. KV Cache 属于请求运行时状态，不属于模型权重。
