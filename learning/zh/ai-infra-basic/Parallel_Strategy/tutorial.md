# 大模型推理并行策略整理：DP / TP / PP / SP / EP

常见并行策略包括：

| 缩写 | 名称 | 中文名 | 切分对象 | 主要解决问题 | 典型通信方式 |
|---|---|---|---|---|---|
| DP | Data Parallelism | 数据并行 | 请求 / batch | 提高吞吐 | 推理阶段通常少通信 |
| TP | Tensor Parallelism | 张量并行 | 单层内部矩阵 / hidden 维度 | 单层太大，单卡算不动 | all-reduce / all-gather |
| PP | Pipeline Parallelism | 流水线并行 | Transformer 层 | 模型层数太多，单卡放不下 | send / recv hidden states |
| SP / CP | Sequence / Context Parallelism | 序列 / 上下文并行 | sequence / context 维度 | 长上下文、KV Cache 压力大 | all-gather KV / broadcast Q + all-reduce / ring attention |
| EP | Expert Parallelism | 专家并行 | MoE Experts | Expert 太多，单卡放不下所有 Expert | all-to-all dispatch / combine |

一句话记忆：

> DP 切请求，TP 切矩阵，PP 切层，SP/CP 切序列或上下文，EP 切专家。

---

# 1. DP：Data Parallelism，数据并行

## 1.1 DP 切的是什么？

DP 切的是：

```text
请求 / batch / 数据
```

DP 不切模型。

每个 DP rank 上都有一份完整模型副本。

例如：

```text
rank 0:
    完整模型副本

rank 1:
    完整模型副本

rank 2:
    完整模型副本
```

如果来了多个请求：

```text
request A -> rank 0
request B -> rank 1
request C -> rank 2
```

每个 rank 独立处理自己的请求。

## 1.2 DP 解决什么问题？

DP 主要解决：

```text
请求量大，需要提高吞吐
```

如果一个模型单卡可以放下，但是在线请求很多，那么可以复制多份模型，每张 GPU 处理一部分请求。

例如：

```text
1 张 GPU:
    一次只能服务 1 份模型副本

4 张 GPU + DP:
    可以同时服务 4 份模型副本
```

所以 DP 更像是“横向扩容”。

## 1.3 DP 推理时每个 rank 上有什么？

每个 DP rank 都有完整模型参数：

```text
rank 0:
    完整 embedding
    完整 attention
    完整 FFN
    完整 lm_head
    request A 的 KV Cache

rank 1:
    完整 embedding
    完整 attention
    完整 FFN
    完整 lm_head
    request B 的 KV Cache
```

不同 DP rank 通常处理不同请求。

## 1.4 DP 推理阶段通信多吗？

推理阶段 DP 通常几乎不需要 rank 间通信。

因为每个 rank 都有完整模型，可以独立完成自己的 forward / decode。

```text
rank 0:
    独立推理 request A

rank 1:
    独立推理 request B
```

训练阶段 DP 需要同步梯度：

```text
all-reduce gradients
```

但推理阶段没有反向传播，所以 DP 通信很少。

## 1.5 DP 的优点

```text
1. 实现简单。
2. 推理阶段通信少。
3. 可以直接提高系统吞吐。
4. 适合模型单卡能放下、请求量大的场景。
```

## 1.6 DP 的缺点

```text
1. 每张 GPU 都要放完整模型，显存占用大。
2. 如果模型单卡放不下，DP 不能解决。
3. 不能降低单个请求的延迟。
4. 对单个超长请求或超大模型没有直接帮助。
```

总结：

> DP 解决的是吞吐问题，不解决单模型太大放不下的问题。

---

# 2. TP：Tensor Parallelism，张量并行

## 2.1 TP 切的是什么？

TP 切的是：

```text
单层内部的大矩阵
```

尤其是 Transformer 里的 Linear 层：

```text
Attention:
    q_proj
    k_proj
    v_proj
    o_proj

FFN:
    up_proj
    gate_proj
    down_proj

LM Head:
    hidden -> vocab
```

这些矩阵很大，单卡存储和计算压力很高，所以 TP 把矩阵切到多个 rank 上共同计算。

## 2.2 TP 解决什么问题？

TP 主要解决：

```text
单层矩阵太大，单卡放不下或算不动
```

例如：

```text
hidden_size = 8192
intermediate_size = 28672
```

FFN 权重很大：

```text
up_proj:
    [8192, 28672]

down_proj:
    [28672, 8192]
```

如果用 TP=2，可以让两个 rank 各保存一部分权重、计算一部分结果。

## 2.3 TP 的核心：Column Parallel 和 Row Parallel

TP 中最重要的是两种切法：

```text
ColumnParallelLinear
RowParallelLinear
```

它们的区别在于：

```text
Column Parallel:
    按输出维度切矩阵，最后可能需要 all-gather 拼接输出。

Row Parallel:
    按输入维度切矩阵，最后需要 all-reduce 求和合并输出。
```

## 2.4 ColumnParallelLinear

普通 Linear：

```text
Y = X @ W
```

假设：

```text
X.shape = [B, S, H]
W.shape = [H, O]
Y.shape = [B, S, O]
```

Column Parallel 按输出维度 `O` 切：

```text
W = [W0 | W1]
```

于是：

```text
rank 0:
    Y0 = X @ W0
    Y0.shape = [B, S, O/2]

rank 1:
    Y1 = X @ W1
    Y1.shape = [B, S, O/2]
```

完整输出：

```text
Y = concat(Y0, Y1)
```

如果后续需要完整输出，就要：

```text
all_gather
```

但是在高性能 TP 中，很多时候不会马上 all-gather，而是保持分片状态继续往后算。

## 2.5 RowParallelLinear

Row Parallel 按输入维度切。

完整计算：

```text
Y = X @ W
```

如果：

```text
X = [X0 | X1]

W =
[
  W0
  W1
]
```

那么：

```text
Y = X0 @ W0 + X1 @ W1
```

每个 rank：

```text
rank 0:
    partial_Y0 = X0 @ W0
    partial_Y0.shape = [B, S, O]

rank 1:
    partial_Y1 = X1 @ W1
    partial_Y1.shape = [B, S, O]
```

注意：

```text
partial_Y0 和 partial_Y1 的 shape 都是完整输出维度 [B, S, O]
```

它们不是拼接关系，而是加法贡献关系。

所以完整输出是：

```text
Y = partial_Y0 + partial_Y1
```

因此通信方式是：

```text
all_reduce_sum
```

## 2.6 TP 在 FFN 中怎么用？

普通 FFN：

```text
x
 ↓
up_proj: hidden_size -> intermediate_size
 ↓
activation
 ↓
down_proj: intermediate_size -> hidden_size
```

TP 中通常是：

```text
up_proj:
    Column Parallel

activation:
    local shard 本地计算

down_proj:
    Row Parallel
```

具体来说：

```text
rank 0:
    local_intermediate_0 = x @ W_up_0
    local_intermediate_0.shape = [B, S, I/2]

    local_intermediate_0 = activation(local_intermediate_0)

    partial_out_0 = local_intermediate_0 @ W_down_0
    partial_out_0.shape = [B, S, H]

rank 1:
    local_intermediate_1 = x @ W_up_1
    local_intermediate_1.shape = [B, S, I/2]

    local_intermediate_1 = activation(local_intermediate_1)

    partial_out_1 = local_intermediate_1 @ W_down_1
    partial_out_1.shape = [B, S, H]

all_reduce:
    out = partial_out_0 + partial_out_1
```

所以 FFN 中的 TP 规律是：

```text
先用 Column Parallel 把中间维度切开；
中间 activation 本地算；
再用 Row Parallel 把局部贡献合并回 hidden_size。
```

## 2.7 TP 在 Attention 中怎么用？

普通 Attention：

```text
x
 ↓
q_proj / k_proj / v_proj
 ↓
multi-head attention
 ↓
o_proj
```

TP 中通常是：

```text
q_proj / k_proj / v_proj:
    Column Parallel，按 attention head 切

local heads attention:
    每个 rank 只算自己负责的 heads

o_proj:
    Row Parallel，all-reduce 合并
```

例如：

```text
hidden_size = 128
num_heads = 8
TP = 2
```

那么：

```text
rank 0:
    head 0, 1, 2, 3

rank 1:
    head 4, 5, 6, 7
```

每个 rank 只生成自己的 Q/K/V heads：

```text
rank 0:
    q0/k0/v0.shape = [B, 4, S, head_dim]

rank 1:
    q1/k1/v1.shape = [B, 4, S, head_dim]
```

然后各自本地做 attention。

最后经过 `o_proj` 时，每个 rank 计算一部分输出贡献：

```text
rank 0:
    partial_out_0.shape = [B, S, hidden_size]

rank 1:
    partial_out_1.shape = [B, S, hidden_size]

all_reduce:
    out = partial_out_0 + partial_out_1
```

## 2.8 TP 的典型通信

| 场景 | 通信方式 | 原因 |
|---|---|---|
| ColumnParallelLinear 需要完整输出 | all-gather | 不同 rank 持有输出维度的不同切片 |
| RowParallelLinear 合并输出 | all-reduce | 不同 rank 是同一个输出的加法贡献 |
| Attention 的 o_proj | all-reduce | 不同 head 的输出贡献需要合并 |
| FFN 的 down_proj | all-reduce | intermediate shard 的输出贡献需要合并 |

## 2.9 TP 的优点

```text
1. 可以把大矩阵切到多卡。
2. 降低单卡参数显存。
3. 降低单卡计算压力。
4. 适合 hidden_size 很大、FFN 很大的模型。
5. 对单个请求也能并行加速。
```

## 2.10 TP 的缺点

```text
1. 每层都可能需要通信，通信频繁。
2. all-reduce 延迟会影响推理性能。
3. TP size 太大时，通信开销可能超过计算收益。
4. 实现复杂，需要严格管理 tensor shape 和 process group。
```

总结：

> TP 是“切一层里的矩阵”，通过 all-gather 或 all-reduce 把局部结果合并。

---

# 3. PP：Pipeline Parallelism，流水线并行

## 3.1 PP 切的是什么？

PP 切的是：

```text
Transformer 层
```

例如一个模型有 40 层，可以切成 4 个 stage：

```text
stage 0:
    embedding + block 0~9

stage 1:
    block 10~19

stage 2:
    block 20~29

stage 3:
    block 30~39 + final_norm + lm_head
```

每个 stage 放在不同 rank 或 rank group 上。

## 3.2 PP 解决什么问题？

PP 主要解决：

```text
模型层数太多，单卡放不下完整模型
```

例如一个 80 层的大模型，如果单卡放不下全部层，可以让每张卡只保存一部分层。

## 3.3 PP 推理流程

假设 2-stage PP：

```text
rank 0 / stage 0:
    embedding + 前半 Transformer Blocks

rank 1 / stage 1:
    后半 Transformer Blocks + final_norm + lm_head
```

prefill / full forward 流程：

```text
input_ids
  ↓
rank 0: stage0
  ↓ send hidden states
rank 1: stage1
  ↓
logits
```

stage 间传递的是：

```text
hidden states
shape = [B, S, H]
```

## 3.4 PP 自回归推理流程

完整生成不是只 forward 一次，而是自回归循环。

无 KV Cache 的简单版：

```text
generated = prompt

for step in max_new_tokens:
    rank 0:
        hidden = stage0(generated)
        send hidden to rank 1

    rank 1:
        logits = stage1(hidden)
        next_token = argmax(logits[:, -1, :])
        send next_token back to rank 0

    rank 0:
        generated = concat(generated, next_token)
```

有 KV Cache 的高性能版本：

```text
Prefill:
    每个 stage 处理完整 prompt；
    每个 stage 建立自己层的 KV Cache。

Decode:
    每一步只处理一个新 token；
    token hidden 从 stage0 流到 stage1；
    最后 stage 生成 next_token；
    next_token 再反馈给 stage0。
```

## 3.5 PP 中每个 rank 上有什么？

每个 PP rank 只保存部分层。

例如：

```text
rank 0:
    embedding
    block 0~19

rank 1:
    block 20~39

rank 2:
    block 40~59

rank 3:
    block 60~79
    final_norm
    lm_head
```

## 3.6 PP 的典型通信

PP 的主要通信是：

```text
send / recv hidden states
```

例如：

```text
stage 0 -> stage 1:
    hidden_states [B, S, H]

stage 1 -> stage 2:
    hidden_states [B, S, H]
```

自回归生成中还可能有：

```text
last stage -> first stage:
    next_token [B, 1]
```

## 3.7 PP 的 pipeline bubble

如果只有一个请求：

```text
time 1:
    stage 0 忙
    stage 1 空闲

time 2:
    stage 0 空闲
    stage 1 忙
```

这就是 pipeline bubble。

为了提高利用率，通常会使用：

```text
micro-batch
multi-request pipeline
continuous batching
```

让多个请求或多个 micro-batch 同时在不同 stage 上流动。

## 3.8 PP 的优点

```text
1. 可以把很深的模型切到多卡。
2. 降低单卡参数显存。
3. 适合层数很多的大模型。
4. 可以和 TP / DP / EP 组合。
```

## 3.9 PP 的缺点

```text
1. stage 间需要传递 hidden states。
2. 存在 pipeline bubble。
3. 单请求延迟可能增加，因为必须顺序经过所有 stage。
4. 自回归 decode 调度复杂。
5. KV Cache 分散在不同 stage，需要独立维护。
```

总结：

> PP 是“按层切模型”，hidden states 像流水线一样从前一个 stage 流到后一个 stage。

---

# 4. SP / CP：Sequence Parallelism / Context Parallelism

## 4.1 SP 和 CP 的关系

SP 和 CP 经常一起讨论。

```text
SP: Sequence Parallelism
    更偏训练或激活按 sequence 维切分。

CP: Context Parallelism
    更偏长上下文推理中，把 context / KV Cache 按 sequence 维分片。
```

它们的核心都可以理解为：

```text
切 sequence / context 维度
```

## 4.2 SP/CP 切的是什么？

切的是：

```text
sequence length / context length
```

完整 hidden：

```text
x.shape = [B, S, H]
```

SP=2 后：

```text
rank 0:
    x[:, 0:S/2, :]
    shape = [B, S/2, H]

rank 1:
    x[:, S/2:S, :]
    shape = [B, S/2, H]
```

## 4.3 SP/CP 解决什么问题？

主要解决：

```text
长上下文带来的显存和计算压力
```

尤其是：

```text
1. 长序列 hidden states
2. Attention 中的 K/V Cache
3. 长上下文 prefill
4. 超长 context decode
```

例如：

```text
S = 128K
H = 8192
layers = 80
```

如果每个 rank 都持有完整 context 的 KV Cache，显存压力会非常大。

CP 可以让：

```text
rank 0:
    KV for token 0~32767

rank 1:
    KV for token 32768~65535

rank 2:
    KV for token 65536~98303

rank 3:
    KV for token 98304~131071
```

## 4.4 Prefill 阶段怎么做？

Prefill 阶段要处理完整 prompt。

如果 sequence 被切分：

```text
rank 0:
    token 0~3

rank 1:
    token 4~7
```

每个 rank 本地可以计算自己的：

```text
local Q/K/V
```

但 Attention 中，某个 token 需要看它之前的所有 token。

例如 rank 1 的 token 4 需要看：

```text
token 0~4
```

其中 token 0~3 在 rank 0 上。

所以 attention 需要跨 rank 通信。

## 4.5 简单实现：all-gather KV

最直观的实现方式是：

```text
每个 rank 计算 local K/V；
all-gather 所有 rank 的 K/V；
每个 rank 用 local Q attend global K/V。
```

也就是：

```text
rank 0:
    local_q0
    local_k0 / local_v0
    all_gather -> full_k / full_v
    output0 = Attention(local_q0, full_k, full_v)

rank 1:
    local_q1
    local_k1 / local_v1
    all_gather -> full_k / full_v
    output1 = Attention(local_q1, full_k, full_v)
```

优点：

```text
实现简单，逻辑直观。
```

缺点：

```text
每个 rank 都 materialize 全量 KV，长上下文场景非常贵。
```

## 4.6 更工业化实现：KV 不动，移动 Q，分布式 softmax

更接近工业实现的思路是：

```text
KV 按 sequence 分片保留在各 rank；
为了计算某一段 Q 的输出，把 Q 广播给所有 rank；
每个 rank 用本地 KV shard 算局部贡献；
通过 all-reduce 合并。
```

完整 attention：

```text
Out = softmax(QK^T)V
```

如果：

```text
K = [K0; K1]
V = [V0; V1]
```

那么：

```text
QK^T = [QK0^T, QK1^T]
```

最终：

```text
Out = P0V0 + P1V1
```

其中：

```text
P = softmax([QK0^T, QK1^T])
```

每个 rank 可以计算局部贡献：

```text
rank 0:
    local_scores0 = Q @ K0^T
    local_out0 = exp(local_scores0 - global_max) @ V0

rank 1:
    local_scores1 = Q @ K1^T
    local_out1 = exp(local_scores1 - global_max) @ V1
```

然后：

```text
global_max = all_reduce_max(local_max)
global_sum = all_reduce_sum(local_sum)
global_out = all_reduce_sum(local_out)

context = global_out / global_sum
```

这里用 all-reduce 是因为：

```text
Out = local_out0 + local_out1
```

也就是不同 rank 算的是同一个输出的加法贡献。

## 4.7 Decode 阶段怎么做？

Decode 阶段每次只处理最新 token：

```text
Q_new.shape = [B, heads, 1, head_dim]
```

历史 KV 很长：

```text
KV_cache.shape = [B, heads, history_len, head_dim]
```

如果 KV Cache 按 context 分片：

```text
rank 0:
    KV for context shard 0

rank 1:
    KV for context shard 1
```

decode 时：

```text
1. 最新 token 的 Q_new 广播到所有 rank。
2. 每个 rank 用本地 KV shard 计算局部 attention。
3. 使用 distributed softmax + all-reduce 合并 context。
4. 得到最新 token 的输出。
```

这种方式比 all-gather 完整 KV 更合理，因为：

```text
Q_new 很小，历史 KV 很大。
```

所以长上下文 decode 中，更倾向于：

```text
移动 Q，不移动 KV。
```

## 4.8 all-gather KV 和 broadcast Q + all-reduce 的区别

### all-gather KV

```text
K_full = concat(K0, K1)
V_full = concat(V0, V1)
```

这是拼接关系，所以使用：

```text
all-gather
```

### broadcast Q + all-reduce

每个 rank 计算：

```text
rank 0:
    local_out0 = P0 @ V0

rank 1:
    local_out1 = P1 @ V1
```

完整输出：

```text
Out = local_out0 + local_out1
```

这是加法贡献关系，所以使用：

```text
all-reduce
```

总结：

```text
拼接关系 -> all-gather
加法贡献关系 -> all-reduce
```

## 4.9 SP/CP 的典型通信

| 实现方式 | 通信方式 |
|---|---|
| 简单 all-gather KV | all-gather K/V |
| 分布式 attention | broadcast Q + all-reduce max/sum/out |
| Ring Attention | ring send/recv K/V blocks |
| Blockwise Attention | 分块通信 + online softmax |

## 4.10 SP/CP 的优点

```text
1. 降低长上下文 KV/hidden 的单卡显存压力。
2. 适合超长 context prefill。
3. 适合长上下文 decode 中分布式 KV Cache。
4. 可以和 TP / PP / DP / EP 组合。
```

## 4.11 SP/CP 的缺点

```text
1. Attention 跨 rank 依赖复杂。
2. 简单 all-gather KV 通信量大。
3. 分布式 softmax 实现复杂。
4. Ring / block attention 调度难度高。
5. decode 阶段和 KV Cache manager 强相关。
```

总结：

> SP/CP 是“切 sequence / context”，每个 rank 保存一段 token 或 KV；attention 通过通信完成跨 context 的全局注意力。

---

# 5. EP：Expert Parallelism，专家并行

## 5.1 EP 切的是什么？

EP 切的是：

```text
MoE Experts
```

MoE 模型中，一个普通 FFN 被替换成多个 Expert：

```text
Expert 0
Expert 1
Expert 2
Expert 3
...
```

每个 Expert 通常就是一个 FFN。

EP 把这些 Experts 分布到不同 rank 上。

例如：

```text
num_experts = 4
EP = 2

rank 0:
    Expert 0
    Expert 1

rank 1:
    Expert 2
    Expert 3
```

## 5.2 EP 解决什么问题？

MoE 的特点是：

```text
总参数量很大；
但每个 token 只激活少量 experts。
```

例如：

```text
总共有 64 个 Experts；
每个 token 只选 Top-1 或 Top-2 Experts。
```

如果每张 GPU 都保存全部 Experts，显存压力非常大。

EP 让每张 GPU 只保存一部分 Experts：

```text
EP = 8
num_experts = 64

每个 rank 只保存 8 个 experts。
```

## 5.3 MoE 在 Transformer 里处于什么位置？

普通 Transformer Block：

```text
x
 ↓
Self-Attention
 ↓
Residual
 ↓
FFN
 ↓
Residual
```

MoE Transformer Block：

```text
x
 ↓
Self-Attention
 ↓
Residual
 ↓
MoE FFN
    ├── Router
    ├── Expert 0
    ├── Expert 1
    ├── Expert 2
    └── Expert 3
 ↓
Residual
```

所以 MoE 通常替代的是：

```text
Transformer Block 里的 FFN 部分
```

Attention 仍然存在。

## 5.4 EP 中 Router 和 Expert 的关系

EP 中通常是：

```text
Router:
    每个 EP rank 上都有完整一份，参数一致。

Experts:
    分片放在不同 rank 上。
```

例如：

```text
rank 0:
    Router
    Expert 0
    Expert 1

rank 1:
    Router
    Expert 2
    Expert 3
```

Router 输出的是：

```text
global expert_id
```

不是本地 expert_id。

例如：

```text
token -> expert 3
```

即使这个 token 当前在 rank 0，rank 0 也可以把它发送到 rank 1，因为 expert 3 在 rank 1。

## 5.5 EP 的核心流程

假设当前 rank 本地有一些 token hidden：

```text
x.shape = [T, H]
```

EP 流程：

```text
1. Router:
    expert_ids = router(x)

2. 根据 expert_id 找 expert 所在 rank:
    dst_rank = expert_id // experts_per_rank

3. Dispatch:
    all-to-all 把 token 发到 expert rank

4. Local Expert Compute:
    每个 rank 用本地 experts 处理收到的 tokens

5. Combine:
    all-to-all 把 expert output 发回 token 原 owner rank

6. Restore:
    owner rank 根据 token 原始 index 恢复顺序
```

## 5.6 为什么 EP 要 all-to-all？

因为每个 rank 的 token 都可能路由到任意 expert。

例如：

```text
rank 0 的 token:
    token0 -> expert 3 -> rank 1
    token1 -> expert 0 -> rank 0

rank 1 的 token:
    token2 -> expert 1 -> rank 0
    token3 -> expert 2 -> rank 1
```

所以：

```text
每个 rank 都可能向每个 rank 发送 token。
```

这就是 all-to-all 通信模式。

## 5.7 Dispatch buffer 里为什么有多个张量？

Dispatch 不只是发送 token hidden，还要发送元数据。

通常需要：

| 张量 | 作用 |
|---|---|
| `send_tokens` | token hidden，真正给 expert 计算的数据 |
| `send_indices` | token 在原 owner rank 的原始位置，用于返回后恢复顺序 |
| `send_expert_ids` | token 要进入哪个 global expert |
| `send_gates` | router gate 权重，用于加权 expert output |
| `send_mask` | 标记这个槽位是否是真实 token，还是 padding |

每个 dispatch slot 实际表达的是：

```text
RouteItem = {
    token_hidden,
    original_index,
    expert_id,
    gate,
    valid_mask
}
```

这些信息需要分成多个 tensor 发送，是因为它们的数据类型不同：

```text
token_hidden: float
original_index: int / long
expert_id: int / long
gate: float
mask: bool
```

## 5.8 Expert 输出后如何恢复原 token 顺序？

Router 会把 token 打散到不同 expert。

例如原始顺序：

```text
token0, token1, token2, token3
```

Router 后：

```text
token0 -> expert3
token1 -> expert0
token2 -> expert2
token3 -> expert1
```

Expert 计算时顺序可能变成：

```text
expert0: token1
expert1: token3
expert2: token2
expert3: token0
```

所以输出不能简单 concat，而要根据原始 index scatter 回去：

```text
out[0] = token0_output
out[1] = token1_output
out[2] = token2_output
out[3] = token3_output
```

这就是为什么需要 `send_indices`。

## 5.9 EP 中 token 初始在哪个 rank？

这不是 EP 自己决定的。

EP 层开始时，每个 rank 已经有一批 local hidden：

```text
rank 0:
    local tokens A

rank 1:
    local tokens B
```

这些 local tokens 的来源由前面的并行布局决定：

```text
DP:
    请求被分到哪个 replica。

SP:
    token 按 sequence 维切到哪个 rank。

TP:
    hidden 可能按 hidden 维分片，进入 MoE 前可能需要 gather 或 TP-aware routing。

PP:
    token hidden 当前流到哪个 pipeline stage。
```

EP 只负责：

```text
owner rank -> expert rank -> owner rank
```

也就是说：

```text
token 进入 MoE 时在哪个 rank；
MoE 输出后还回到哪个 rank。
```

中间只是在 expert 计算时临时发送到其他 rank。

## 5.10 最简单 DP + EP 中 token 怎么分？

假设：

```text
DP = 2
EP = 2
world_size = 4
```

可以组织为：

```text
DP replica 0:
    rank 0, rank 1

DP replica 1:
    rank 2, rank 3
```

request A 被 DP scheduler 分到 replica 0。

那么 request A 只在：

```text
rank 0, rank 1
```

这个 EP group 内处理。

在 EP group 内，最简单方式是按 token 平均切分：

```text
request A tokens = [t0,t1,t2,t3,t4,t5,t6,t7]

rank 0:
    t0,t1,t2,t3

rank 1:
    t4,t5,t6,t7
```

然后每个 rank 对自己手里的 tokens 做 Router，再 all-to-all 到 expert rank。

但平均切只是简单实现，不是唯一方式。

## 5.11 EP 的优点

```text
1. 支持超大 MoE 参数量。
2. 每个 rank 只保存部分 experts。
3. 每个 token 只激活少量 experts，计算量可控。
4. 适合大规模稀疏模型。
```

## 5.12 EP 的缺点

```text
1. all-to-all 通信复杂且开销大。
2. Router 可能导致负载不均衡。
3. 需要 capacity factor / token dropping。
4. 实现复杂。
5. 和 TP / PP / SP / DP 组合时需要复杂 process group。
```

总结：

> EP 是“Router 复制，Expert 分片；token 全局路由到 expert rank，算完再返回 owner rank”。

---

# 6. 五种并行策略如何组合？

真实大模型中，经常组合：

```text
DP × PP × TP × EP × SP/CP
```

不是简单相加，而是多维进程网格。

例如：

```text
DP = 2
PP = 2
TP = 2
EP = 4
```

总进程数可能是：

```text
2 × 2 × 2 × 4 = 32
```

每个 global rank 都有多个逻辑身份：

```text
global_rank = 17

可能对应:
    dp_rank = 1
    pp_rank = 0
    tp_rank = 1
    ep_rank = 2
```

不同层使用不同 process group：

```text
普通 Linear:
    TP group

Pipeline stage 间:
    PP group

MoE dispatch:
    EP group

请求副本:
    DP group

长上下文 attention:
    SP/CP group
```

---

# 7. 各并行策略的核心区别

## 7.1 从“切什么”看

```text
DP:
    切请求 / batch

TP:
    切矩阵 / hidden 维度

PP:
    切 Transformer 层

SP/CP:
    切 sequence / context 维度

EP:
    切 MoE experts
```

## 7.2 从“通信什么”看

```text
DP:
    推理阶段通常少通信

TP:
    all-reduce / all-gather 矩阵计算结果

PP:
    send / recv hidden states

SP/CP:
    通信 K/V 或 Q，做分布式 attention

EP:
    all-to-all 发送 token 到 experts
```

## 7.3 从“解决什么问题”看

```text
DP:
    请求多，提升吞吐

TP:
    单层太大，矩阵算不动

PP:
    模型太深，层数太多放不下

SP/CP:
    上下文太长，KV / activation 太大

EP:
    MoE experts 太多，expert 参数放不下
```

---

# 8. 推理场景中如何选择？

## 8.1 模型单卡能放下，请求量大

优先考虑：

```text
DP
```

原因：

```text
复制多份模型，提高吞吐。
```

## 8.2 模型单层矩阵太大

优先考虑：

```text
TP
```

原因：

```text
切分大矩阵，让多个 rank 共同计算一层。
```

## 8.3 模型层数太多，单卡放不下

优先考虑：

```text
PP
```

原因：

```text
按层切模型，每个 rank 只保存部分层。
```

## 8.4 上下文很长

优先考虑：

```text
SP / CP
```

原因：

```text
切分 sequence / context，降低 KV Cache 和 attention 压力。
```

## 8.5 MoE 模型 expert 很多

优先考虑：

```text
EP
```

原因：

```text
不同 rank 保存不同 experts。
```

---

# 9. 一个完整例子

假设部署一个 MoE LLM：

```text
模型:
    64 layers
    hidden_size = 8192
    num_experts = 64
    每层 MoE top-2
    context length = 128K
```

可能配置：

```text
DP = 2
PP = 4
TP = 4
EP = 8
CP = 2
```

可以理解为：

```text
DP:
    两个副本处理不同请求。

PP:
    64 层切成 4 个 stage，每个 stage 16 层。

TP:
    每个 stage 内 dense / attention 矩阵用 4-way TP。

EP:
    MoE experts 在 8 个 expert ranks 上分布。

CP:
    128K context 的 KV Cache 按 context 维分片。
```

---

# 10. 最终总结

五种并行策略可以这样记：

```text
DP 切请求；
TP 切矩阵；
PP 切层；
SP/CP 切序列或上下文；
EP 切专家。
```

进一步说：

```text
DP:
    多个完整模型副本处理不同请求。

TP:
    多个 rank 共同计算同一层的大矩阵。

PP:
    不同 rank 保存不同层，hidden states 流水线传递。

SP/CP:
    不同 rank 保存不同 token/context 的 hidden 或 KV，通过分布式 attention 合并。

EP:
    不同 rank 保存不同 experts，token 被 all-to-all 路由到目标 expert。
```

它们的本质区别在于：

```text
1. 切分维度不同。
2. 每个 rank 保存的东西不同。
3. 通信原语不同。
4. 适合解决的问题不同。
```

最终，大模型推理通常不是只用一种并行，而是根据：

```text
模型参数规模
层数
hidden size
context length
MoE expert 数量
吞吐目标
延迟目标
GPU 数量和互联拓扑
```

把 DP / TP / PP / SP / EP 组合成一个多维并行网格。
