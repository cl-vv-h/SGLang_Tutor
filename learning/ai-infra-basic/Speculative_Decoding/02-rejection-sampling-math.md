# 02. 投机采样的数学原理

## 1. 问题定义

设当前前缀为 `x`。Target model 给出的目标分布是：

```text
p(v) = P_target(v | x), v in vocabulary
```

Draft path 给出的提案分布是：

```text
q(v) = P_draft(v | x)
```

目标：先从 `q` 中采样候选 `y`，但最终输出的 token 分布必须等于 `p`。

如果直接输出 `y ~ q`，分布会变成 draft distribution，模型质量会改变。严格 speculative sampling 用拒绝采样修正这一点。

## 2. 单 token 拒绝采样规则

先从 draft 分布采样：

```text
y ~ q
```

然后用 target 分布检查这个候选：

```text
accept_prob(y) = min(1, p(y) / q(y))
```

采样一个均匀随机数 `u ~ Uniform(0,1)`：

```text
if u <= accept_prob(y):
    output y
else:
    output z ~ r
```

其中 residual distribution `r` 定义为：

```text
r(v) = max(p(v) - q(v), 0) / Z
Z = sum_u max(p(u) - q(u), 0)
```

这条规则的含义是：

1. 如果 draft 对 token `v` 的概率不超过 target，即 `q(v) <= p(v)`，那么 draft 提出的 `v` 总是可以接受。
2. 如果 draft 对 `v` 的概率高于 target，即 `q(v) > p(v)`，只能接受其中 `p(v)/q(v)` 的比例。
3. 被拒绝时，说明 draft 在某些 token 上分配了过多概率，需要从 target 比 draft 更偏好的 token 集合中补回来。

## 3. 图解：min 部分与 residual 部分

![拒绝采样概率分解](./assets/rejection-sampling.svg)

对每个 token，目标概率 `p(v)` 可以拆成两部分：

```text
p(v) = min(p(v), q(v)) + max(p(v) - q(v), 0)
```

其中：

```text
min(p, q)             来自接受 draft token
max(p - q, 0)         来自拒绝后的 residual sample
```

只要这两部分加起来正好等于 `p`，最终输出就保持 target distribution。

## 4. 单 token 证明

对任意 token `v`，最终输出 `v` 的概率由两部分组成。

第一部分：draft 采样到 `v` 且被接受：

```text
P(output=v, accept)
= q(v) * min(1, p(v)/q(v))
= min(q(v), p(v))
```

第二部分：某个 draft token 被拒绝，然后 residual sample 采到 `v`：

```text
P(output=v, reject)
= P(reject) * r(v)
```

先计算拒绝概率：

```text
P(reject)
= 1 - sum_u min(q(u), p(u))
```

因为 `sum p = sum q = 1`，有：

```text
1 - sum_u min(q(u), p(u))
= sum_u max(p(u) - q(u), 0)
= Z
```

所以：

```text
P(output=v, reject)
= Z * max(p(v)-q(v),0) / Z
= max(p(v)-q(v),0)
```

两部分相加：

```text
P(output=v)
= min(q(v),p(v)) + max(p(v)-q(v),0)
= p(v)
```

因此，虽然候选来自 draft 分布 `q`，最终输出分布仍然是 target 分布 `p`。

## 5. 多 token 链式验证

投机解码不是只猜一个 token，而是猜一串：

```text
y_1, y_2, ..., y_K
```

Draft 的链式分布为：

```text
q_i(v) = P_draft(v | x, y_1, ..., y_(i-1))
y_i ~ q_i
```

Target verify 一次算出每个位置的目标分布：

```text
p_i(v) = P_target(v | x, y_1, ..., y_(i-1))
```

验证从 `i=1` 开始顺序进行：

```text
for i in 1..K:
    accept y_i with probability min(1, p_i(y_i)/q_i(y_i))
    if rejected:
        sample correction z from residual r_i(v)
        commit y_1...y_(i-1), z
        stop this speculative round
```

如果 `K` 个 draft token 全部接受，则 target verify 已经给出了下一个位置的分布：

```text
p_(K+1)(v) = P_target(v | x, y_1, ..., y_K)
```

这时可以额外采样一个 bonus token：

```text
z ~ p_(K+1)
commit y_1...y_K, z
```

一轮最多提交 `K+1` 个 token。

## 6. 为什么 target verify 可以并行

链式验证的接受判断是顺序的，但 target logits 可以并行算出来。原因是 target verify 的每个位置都使用固定候选前缀：

```text
position 1 sees: x
position 2 sees: x, y_1
position 3 sees: x, y_1, y_2
...
position K sees: x, y_1, ..., y_(K-1)
```

这些条件前缀在 draft 阶段已经确定，因此 target 可以像一次短 prefill 一样并行处理 `y_1...y_K`。验证时只是在 target logits 上顺序决定接受到哪里。

这也是 speculative decoding 能加速的关键：

```text
sequential decision
parallel target computation
```

## 7. Greedy decoding 特例

当 temperature 为 0 或严格 greedy 时，目标分布退化成：

```text
p(v*) = 1, v* = argmax target_logits
p(v)  = 0, v != v*
```

这时不需要概率比值。验证规则变成：

```text
if draft_token_i == target_argmax_i:
    accept
else:
    reject and output target_argmax_i
```

也就是说 greedy speculative decoding 接受的是 draft 和 target 完全一致的最长前缀。

## 8. Top-k、Top-p、temperature 与 grammar

严格保持目标分布时，`p` 必须是最终想要服务的 target 分布。也就是说，所有 sampling processor 都应该被纳入 `p` 的定义：

```text
raw target logits
  -> temperature
  -> repetition penalty
  -> top-k / top-p
  -> grammar mask
  -> softmax
  -> p
```

Draft 分布 `q` 可以不同，但 verifier 必须知道候选 token 在 `q` 下的概率，才能计算 `p(y)/q(y)`。

### Grammar 约束

若 grammar 规定某个 token 非法，则：

```text
p(token) = 0
```

如果 draft 提出非法 token，它会被拒绝。工程上更好的做法是让 draft 也使用同样的 grammar mask，这样可以减少无效候选和拒绝率。

### Top-p / Top-k

Top-p 和 Top-k 会把部分 token 的 target 概率截断为 0。如果 draft 经常提出 target 截断后的 token，接受率会下降。严格实现仍然正确，但速度收益会变差。

## 9. Tree verification 的数学视角

Tree verification 不是改变接受规则，而是一次 target forward 验证更多候选条件前缀。候选节点 `n` 表示一条路径：

```text
path(n) = [y_1, y_2, ..., y_d]
```

target 在节点 `n` 上计算的是：

```text
P_target(. | x, y_1, ..., y_(d-1))
```

验证时只能提交一条从 root 出发的路径。系统会根据候选树结构、target 分布、draft 分布和采样随机数决定接受路径的前缀长度。树的价值在于提供更多分叉，让 target 更容易在一次 verify 中找到可接受的连续路径。

工程上，tree verification 必须额外维护：

```text
node_id -> token_id
node_id -> parent_id
node_id -> position
node_id -> attention visible ancestors
node_id -> path retrieval index
```

因此它比线性链更难实现，但在高质量 draft 或多分支 draft 中收益更高。

## 10. 数值与边界情况

### q(y) 很小

如果 `q(y)` 很小而 `p(y)` 较大，则 `p(y)/q(y)` 很大，接受概率截断为 1。因为 draft 很少提出这个 token，一旦提出通常应当接受。

### q(y)=0

若 `q(y)=0`，draft 不会采样到 `y`。如果 target 认为 `y` 有概率，它只能通过 residual distribution 在拒绝后被采样出来。

### residual normalization 很小

如果 `p` 和 `q` 非常接近：

```text
Z = sum max(p-q,0)
```

会很小。这代表拒绝概率很低。实现中需要注意浮点误差和归一化稳定性。

### 概率与 logits

理论公式写在概率空间，但实际系统常持有 logits。计算时一般会：

```text
target_logits -> log_softmax -> target_logprobs
draft_logits  -> log_softmax -> draft_logprobs
```

接受判断可以写成：

```text
log(u) <= log p(y) - log q(y)
```

这样比直接除法更稳定。

## 11. 伪代码

```python
def speculative_sample(prefix, draft, target, K):
    draft_tokens = []
    draft_probs = []

    state = prefix
    for i in range(K):
        q = draft.next_distribution(state)
        y = sample(q)
        draft_tokens.append(y)
        draft_probs.append(q)
        state = state + [y]

    target_probs = target.verify(prefix, draft_tokens)  # K + 1 rows

    accepted = []
    for i, y in enumerate(draft_tokens):
        p_i = target_probs[i]
        q_i = draft_probs[i]

        if uniform() <= min(1.0, p_i[y] / q_i[y]):
            accepted.append(y)
            continue

        residual = normalize(max(p_i - q_i, 0))
        correction = sample(residual)
        return accepted + [correction]

    bonus = sample(target_probs[K])
    return accepted + [bonus]
```

这段伪代码描述的是严格线性 speculative sampling。实际 serving 系统还要处理 batch、KV Cache、tree mask、CUDA Graph、stop condition、logprob 和输出流。
