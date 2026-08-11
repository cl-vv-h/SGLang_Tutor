# 03. GDN 的训练、Prefill、Decode 与 Serving 状态管理

## 1. 哪些是可训练参数

以 SGLang 的 `Qwen3GatedDeltaNet` 为例，GDN 层内可训练参数包括：

| 参数 | 代码模块/字段 | 形状直觉 | 作用 |
|---|---|---:|---|
| `in_proj_qkvz.weight` | `MergedColumnParallelLinear` | `[2*H*K+2*HV*V, H_model]` | 从 hidden 投影出 q/k/v/z |
| `in_proj_ba.weight` | `MergedColumnParallelLinear` | `[2*HV, H_model]` | 从 hidden 投影出 b/a gate 激活 |
| `conv1d.weight` | `ColumnParallelLinear` 后 reshape | `[2*H*K+HV*V, C]` 的等价布局 | q/k/v 路径的短 causal convolution |
| `A_log` | `nn.Parameter` | `[HV]` | log-space decay 参数 |
| `dt_bias` | `nn.Parameter` | `[HV]` | decay/time-step 偏置 |
| `norm.weight` | `RMSNormGated` | `[V]` 或实现相关 | 输出 gate/norm 的缩放参数 |
| `out_proj.weight` | `RowParallelLinear` | `[H_model, HV*V]` | 把 GDN 多 head 输出投回 hidden size |

不是可训练参数的对象：

| 对象 | 原因 |
|---|---|
| `q/k/v/z/a/b` | 当前 token 的 activation，由投影计算得到 |
| `g/beta` | 由 `A_log/dt_bias/a/b` 计算得到的控制信号 |
| `S_t` / `ssm_states` | 每个请求的运行时状态缓存，不进入 optimizer |
| `conv_states` | causal conv 的运行时窗口缓存，不进入 optimizer |
| `cache_indices` | 请求到状态槽位的索引，不是模型数值参数 |

## 2. 训练目标是什么

GDN 本身不是一个单独训练目标。它作为 decoder layer 的一部分参与 causal language modeling。

训练样本：

```text
input_ids:  [B,S]
labels:     [B,S]
```

模型输出：

```text
logits: [B,S,Vocab]
```

loss：

```text
loss = CrossEntropy(logits[:, :-1, :], labels[:, 1:])
```

反向传播会更新 GDN 层内参数：

```text
loss
  -> lm_head
  -> decoder layers
  -> GDN out_proj / norm / projections / conv / A_log / dt_bias
```

但不会更新运行时 state：

```text
S_t 是 forward 过程中的中间状态；
它参与梯度计算，但不是 optimizer 持有的 Parameter。
```

## 3. 训练时 state 怎么处理

训练通常使用 teacher forcing，一次给一段序列。对每个训练样本，初始状态一般是零或由前一段切块传入：

```text
S_0 = zeros([B, HV, V, K])
```

然后对序列递推：

```text
for t in 1..S:
    S_t, o_t = GDNStep(S_(t-1), q_t, k_t, v_t, g_t, beta_t)
```

如果直接这样顺序跑，训练长序列会很慢。高效训练会用 chunk/chunkwise parallel 算法。

## 4. 为什么训练需要 chunk 并行

GDN recurrence 是顺序定义的：

```text
S_t depends on S_(t-1)
```

这对 decode 很自然，因为 decode 本来一次一个 token。但训练/prefill 需要一次处理长序列：

```text
T = 4096, 32768, ...
```

如果每个 token 一个 kernel，GPU 并行度会很差。因此 GDN 使用 chunk 算法：

```text
把序列切成 chunk
在 chunk 内利用 lower-triangular 结构并行求解
在 chunk 间传递 summary state
```

![GDN chunk prefill](./assets/gdn-chunk-prefill.svg)

## 5. chunk gated delta rule 的直觉

单步递推里有：

```text
S_t = exp(g_t) * S_(t-1)
      + beta_t * (v_t - exp(g_t) * S_(t-1) @ k_t) outer k_t
```

一个 chunk 内的 token 会互相影响。第 `i` 个 token 的写入会影响第 `j>i` 个 token 的预测。因此 chunk 内需要处理一个因果的下三角依赖。

SGLang 的 `chunk_fwd.py` 中可以看到核心阶段：

```text
1. compute beta * K @ K^T lower triangular matrix
2. apply gate scaling exp(g_i - g_j)
3. solve lower triangular system, 得到 (I + A)^-1
4. recompute write vectors w/u
5. 合并 chunk state 并计算 output
```

为什么出现 `K @ K^T`？

```text
因为每个写入方向是 k_i，
后续 token 用 k_j 读取旧写入时会出现 k_j dot k_i。
```

也就是 chunk 内 token-to-token 的相互作用可以组织成一个下三角 Gram-like 矩阵。

## 6. 推理 prefill 的流程

prefill 处理 prompt 或新增长段 tokens：

```text
hidden_states: [T,H_model]
query_start_loc: [N+1]
```

端到端流程：

```text
1. 投影:
       hidden -> mixed_qkv, z, a, b

2. causal conv:
       mixed_qkv -> conv(mixed_qkv)
       写入每条请求最后 C-1 个 conv window

3. split:
       q [1,T,H,K]
       k [1,T,H,K]
       v [1,T,HV,V]

4. gating:
       g [1,T,HV]
       beta [1,T,HV]

5. chunk_gated_delta_rule:
       读取 initial_state
       并行计算所有 token 输出
       得到 final_state

6. state tracking:
       把每条请求的 final_state 写回 ssm_states[cache_indices]

7. output:
       core_attn_out + z -> RMSNormGated -> out_proj
```

prefill 的结果包括两类：

| 输出 | 生命周期 |
|---|---|
| 当前 token 的 hidden output | 继续流向后续层和 logits |
| 每条请求的 final GDN state | 保存到请求状态池，供后续 decode 使用 |

## 7. 推理 decode 的流程

decode 通常每请求一个 token：

```text
hidden_states: [B,H_model]
```

流程：

```text
1. 投影得到 mixed_qkv/z/a/b
2. causal_conv1d_update 更新 conv_states
3. 从 ssm_states[cache_indices] 读取旧状态
4. 计算 g/beta
5. 执行一次 recurrent update
6. 把新状态写回 ssm_states[cache_indices]
7. 输出 core_attn_out -> gated norm -> out_proj
```

decode 的核心是常数时间状态更新：

```text
每个 token 不需要扫描所有历史 token；
只读写固定大小的 [HV,V,K] state。
```

这就是 GDN 在长上下文 serving 中有吸引力的原因。

## 8. packed decode fast path

SGLang 的 Triton GDN backend 支持 packed decode：

```text
mixed_qkv: [B, 2*H*K + HV*V]
a:         [B, HV]
b:         [B, HV]
ssm_states:[num_slots, HV, V, K]
```

packed kernel 在一个 kernel 中完成：

```text
1. 从 mixed_qkv 取 q/k/v
2. 对 q/k 做 L2 norm
3. 计算 g 和 beta
4. 读取旧 state
5. 执行 delta-rule update
6. 写回 state
7. 输出 [B,1,HV,V]
```

这样避免了多个中间 tensor 和 kernel launch：

```text
split q/k/v
gating kernel
recurrent update kernel
```

合并为一条更紧的 decode fast path。

## 9. target verify 与临时状态

投机解码 target verify 中，一条请求可能同时验证多个 draft token。GDN 必须沿候选路径推进状态，但不能提前污染正式状态。

需要区分：

| 状态 | 用途 |
|---|---|
| `ssm_states` | 已提交 token 的正式状态 |
| `intermediate_states_buffer` | draft token 或 tree verify 的临时状态 |
| `intermediate_conv_window` | draft token 对 conv window 的临时推进 |

verify kernel 常使用：

```text
disable_state_update=True
```

含义：

```text
计算每个候选 token 的 target 输出；
缓存必要的中间状态；
等 accept/reject 后再提交正确路径。
```

这与投机解码中临时 KV 的管理思想一致。

## 10. Serving 中的状态管理

GDN serving 至少有两类状态：

```text
conv_states:
    短卷积窗口，保存最近 C-1 个 q/k/v projection 输入

ssm_states:
    GDN recurrent state，保存压缩历史记忆
```

它们由请求池和 cache indices 管理：

```text
request_id -> state slot -> conv_states[slot], ssm_states[slot]
```

当请求结束：

```text
释放 state slot
```

当 prefix cache 命中：

```text
不仅要恢复 token 前缀；
还要恢复该前缀对应的 conv state 和 ssm state。
```

当进行状态迁移或 PD 分离：

```text
GDN state 需要像 KV Cache 一样被视作请求运行时状态。
```

## 11. GDN 与 KV Cache 的显存比较

标准 attention 的 KV Cache 主体：

```text
KV bytes ∝ context_len * num_layers * num_kv_heads * head_dim
```

GDN state 主体：

```text
GDN bytes ∝ num_linear_layers * HV * V * K
```

它不随 context_len 线性增长。

但要注意：

```text
混合模型通常不是所有层都是 GDN。
full attention 层仍然需要 KV Cache。
GDN 层还需要 conv state 和 ssm state。
```

所以实际显存是：

```text
total_runtime_cache =
    full_attention_KV_cache
    + GDN_ssm_states
    + GDN_conv_states
    + allocator/prefix/spec metadata
```

## 12. 训练与 serving 的差异

| 维度 | 训练 | Serving |
|---|---|---|
| 输入 | 长序列 teacher forcing | prefill 多 token，decode 单 token |
| 状态 | 中间激活参与反传，通常不跨样本永久保存 | 每个请求持久保存 conv/ssm state |
| kernel | 需要 forward + backward | 多数只需要 forward |
| 并行策略 | chunk parallel、sequence parallel、activation checkpoint | chunk prefill、packed decode、CUDA graph、state pool |
| 正确性重点 | 梯度、loss、数值稳定 | 状态提交、prefix cache、target verify、batch 调度 |

SGLang 中部分 GDN kernels 明确是 inference/serving 路径，并不实现 backward。训练 GDN 模型时需要使用支持反向传播的训练框架或专用 FLA kernels。

## 13. 常见排障点

| 现象 | 可能原因 |
|---|---|
| prefill 正常，decode 异常 | conv state 或 ssm state 没正确写回 |
| 单条请求正常，continuous batching 异常 | `cache_indices` 与请求槽位错位 |
| target verify 后输出漂移 | 临时 state 提交路径错误 |
| 长上下文退化 | gate 衰减过强/过弱，状态压缩能力不足，或 full attention 间隔不合适 |
| 多卡 shape mismatch | `H/HV/K/V` 被 TP 切分后与 packed projection 不一致 |
| 只在某些 GPU 后端失败 | GDN backend 对 prefill/decode/verify 支持程度不同 |

## 14. 小结

1. GDN 的可训练参数在投影、conv、`A_log/dt_bias`、norm 和 out projection；状态不是参数。
2. 训练通过 causal LM loss 反传到 GDN 参数，长序列依赖 chunk 并行算法。
3. prefill 用 chunk gated delta rule，decode 用 recurrent update，packed decode 会进一步融合 split/gating/update。
4. target verify 必须使用临时状态，避免 draft token 污染正式 request state。
5. GDN 降低长上下文 cache 随长度增长的问题，但引入了 state pool、conv state 和混合层调度的新复杂度。
