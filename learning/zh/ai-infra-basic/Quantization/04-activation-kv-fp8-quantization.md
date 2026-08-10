# 04. Activation、KV Cache、FP8 与 MXFP8 量化

## 1. activation 量化为什么更难

权重是固定的，activation 是每个请求、每个 token、每层都变化的。

| 差异 | 权重 | activation |
|---|---|---|
| 是否固定 | 固定 | 随输入变化 |
| 能否离线慢慢校准 | 可以 | 只能估计，或运行时动态计算 |
| outlier 是否稳定 | 相对稳定 | token、层、任务之间变化大 |
| 运行时 scale 成本 | 无或很低 | 需要 per-token/per-block amax 和 scale |

因此 activation 量化经常使用动态 scale：

```text
对当前 token hidden vector:
    amax = max(abs(x))
    scale = amax / qmax
    q_x = round(x / scale)
```

读法：

```text
每个 token 现场看自己的最大值；
根据最大值决定缩放；
再把这一行 activation cast 到 INT8/FP8/INT4。
```

## 2. per-token activation quantization

设当前 packed hidden states：

```text
X: [T, H]
```

per-token scale：

```text
scale_x: [T]
```

每个 token 行独立量化：

```text
q_x[t, h] = round(X[t, h] / scale_x[t])
```

| 优点 | 缺点 |
|---|---|
| 每个 token 自己适应范围 | 需要额外求 amax |
| 比 per-tensor 更抗 outlier | scale 读取和广播增加 kernel 复杂度 |
| 适合 decode/prefill 混合 batch | 不同 token 的 scale 会影响 GEMM 实现 |

per-token 常用于 W8A8 INT8 或 FP8 activation 动态量化。

## 3. per-token-group activation quantization

有时一个 token 的 `H` 维太宽，只用一个 scale 仍然太粗，可以沿 hidden 维再分组：

```text
X:       [T, H]
group:   G
scales: [T, ceil(H / G)]
```

元素恢复：

```text
group_id = floor(h / G)
X_hat[t,h] = q_x[t,h] * scales[t, group_id]
```

这比 per-token 更准，但 scale metadata 更多，kernel 需要在 tile 内按 group 读取 scale。

## 4. W8A8 INT8 的典型数据流

```text
BF16 hidden_states [T,H]
    -> dynamic quant
INT8 activation [T,H] + per-token scale [T]
    -> INT8/BF16-aware GEMM with INT8 weight
output BF16/FP16 [T,O]
```

如果权重是 per-channel INT8：

```text
W_q:       [O,H] int8
scale_w:   [O]
X_q:       [T,H] int8
scale_x:   [T]
```

矩阵乘法核心先得到整数累加：

```text
acc[t,o] = sum_h X_q[t,h] * W_q[o,h]
```

再恢复尺度：

```text
Y[t,o] ≈ acc[t,o] * scale_x[t] * scale_w[o]
```

读法：

```text
INT8 GEMM 负责快算整数乘加；
scale_x 和 scale_w 决定这个整数累加值代表多大的真实浮点值。
```

## 5. W4A4 为什么更激进

W4A4 同时压权重和 activation 到 4-bit。相比 W4A16，它多了 activation 误差；相比 W8A8，它少了一半的 activation 表示能力。

对 Linear：

```text
Y_hat = dequant(A4) @ dequant(W4)
```

误差项包括：

```text
Y_hat - Y ≈ X @ e_w + e_x @ W + e_x @ e_w
```

W4A4 要成功，通常需要：

| 条件 | 说明 |
|---|---|
| 更细 activation scale | per-token 或 per-token-group |
| outlier 抑制 | SmoothQuant、QuaRot、Hadamard/rotation、clipping |
| 硬件支持 | INT4/FP4 tensor core 或 NPU int4 matmul |
| fused kernel | 单独 dequant 再 GEMM 会吃掉收益 |
| 质量回归 | 长上下文、代码、数学和工具调用都要测 |

## 6. FP8 activation

FP8 比 INT8 更适合范围变化较大的数据，因为 exponent 提供动态范围。

常见形式：

```text
X_bf16 -> cast/quantize -> X_fp8 + scale
```

| 方案 | scale |
|---|---|
| delayed scaling | 使用历史 amax 决定本轮 scale |
| current scaling | 使用当前 tensor 的 amax |
| per-token FP8 | 每个 token 或 row 一个 scale |
| block FP8 | 每个 block 一个 scale |
| MXFP8 | 每 32 个连续值一个 E8M0 scale |

在推理中，动态 current scaling 更直观：

```text
amax = max(abs(X))
scale = amax / fp8_max
X_fp8 = cast(X / scale)
```

但大模型 serving 更关心：

```text
求 amax 的开销
scale tensor 的布局
GEMM 是否原生支持该 FP8/MXFP8 格式
输出 dtype 和后续 residual/norm 是否匹配
```

## 7. MXFP8 activation 数据流

MXFP8 使用 block scale。对 `[T,H]` activation，常见 rowwise block 是：

```text
每个 token 行沿 H 维每 32 个元素一个 block
```

```text
X_bf16: [T,H]
    -> split H into blocks of 32
    -> each block compute amax
    -> each block choose E8M0 scale
    -> cast payload to FP8 E4M3

X_mxfp8_payload: [T,H] uint8/float8 storage
X_mxfp8_scale:   [T,H/32] uint8 E8M0
```

计算时：

```text
for each tile:
    load FP8 payload
    load E8M0 block scale
    tensor core matmul with block-scaled operands
    accumulate/output BF16/FP32 depending on kernel
```

MXFP8 的优势：

| 优势 | 说明 |
|---|---|
| 比 per-tensor FP8 更抗局部 outlier | 每 32 个值有自己的 scale |
| scale metadata 紧凑 | E8M0 只有 8-bit |
| 适合硬件原生 block scaling | Blackwell 等硬件可直接加速 |

代价：

| 代价 | 说明 |
|---|---|
| shape 对齐严格 | last dim 常要能被 32 整除，kernel 还有 tile 对齐要求 |
| rowwise/columnwise 不等价 | 需要按 GEMM operand 方向分别准备 |
| scale dtype 特殊 | E8M0/UE8M0 常以 `uint8` 形式出现 |

## 8. KV Cache 量化

KV Cache 存的是每层 attention 的历史 K/V：

```text
K_cache: [num_blocks, block_size, num_kv_heads, head_dim]
V_cache: [num_blocks, block_size, num_kv_heads, head_dim]
```

未量化时通常是 BF16/FP16。量化后可能是：

```text
FP8 KV + scale
INT8 KV + scale
MXFP8 KV + block scale
```

KV Cache 显存估算：

```text
bytes_per_token =
    num_layers * 2 * num_kv_heads * head_dim * bytes_per_element
```

其中 `2` 表示 K 和 V。

例子：

```text
num_layers = 80
num_kv_heads = 8
head_dim = 128

BF16 bytes_per_element = 2
FP8  bytes_per_element = 1
```

```text
BF16 KV per token = 80 * 2 * 8 * 128 * 2 = 327680 bytes ≈ 320 KB
FP8  KV per token = 80 * 2 * 8 * 128 * 1 = 163840 bytes ≈ 160 KB
```

不含 scale 和分页元数据时，FP8 KV 主体显存约减半。

## 9. KV Cache 为什么敏感

attention logit：

```text
score = Q @ K^T / sqrt(head_dim)
```

如果 K 被量化：

```text
K_hat = K + e_k
score_hat = Q @ K_hat^T / sqrt(D)
          = score + Q @ e_k^T / sqrt(D)
```

读法：

```text
K 的误差会直接进入 attention score；
score 再经过 softmax；
softmax 对相对差异敏感，尤其长上下文中候选位置很多。
```

V 的误差进入加权求和：

```text
out = softmax(score) @ V
out_hat = softmax(score_hat) @ (V + e_v)
```

K 误差会影响“看哪里”，V 误差会影响“读到什么”。K 的量化通常更需要谨慎。

## 10. KV Cache scale 粒度

| 粒度 | scale 形状直觉 | 优缺点 |
|---|---|---|
| per-tensor | 每层 K/V 一个 scale | metadata 少，但精度粗 |
| per-layer | 每层单独 scale | 比全局更稳 |
| per-head | 每层每个 head 一个 scale | 适应不同 head 分布 |
| per-token | 每个 token 一个 scale | 准，但 scale 多 |
| per-block/page | 每个 KV page 一个 scale | 和 paged KV cache 匹配 |
| per-channel/head_dim | 沿 head_dim 分组 | 更准，kernel 更复杂 |
| MXFP8 block | 每 32 个连续值一个 E8M0 scale | 精度和 metadata 折中，但有对齐要求 |

KV Cache 量化还必须考虑 prefix cache 和 KV transfer：

```text
如果 prefix cache 存的是 FP8 KV，
复用时必须带着 scale 一起复用；
跨机器传输时也必须传 payload + scale + dtype metadata。
```

## 11. FP8 KV 的常见运行时路径

```text
prefill/decode 产生 K_new, V_new: BF16/FP16
    -> quantize to FP8/INT8/MXFP8
    -> 写入 KV Cache memory pool

attention 读取历史 K/V:
    -> kernel 内部按 scale 解释低 bit KV
    -> 计算 QK^T、softmax、weighted sum
```

有两种实现风格：

| 风格 | 说明 |
|---|---|
| attention kernel 原生消费 FP8 KV | 最理想，减少读取带宽 |
| 读取后局部反量化 | 实现简单，但可能多一次转换和临时 buffer |

如果 attention backend 不支持该 KV dtype，系统可能 fallback 到 BF16 KV 或报错。

## 12. activation/KV 量化的验证指标

| 指标 | 为什么看 |
|---|---|
| perplexity | 基础语言建模质量 |
| 长上下文 QA | KV 量化误差随长度累积 |
| passkey/retrieval | attention 是否还能精确找远处信息 |
| 代码生成 | 对 logits 排序和长依赖敏感 |
| 数学推理 | 多步推理中小误差容易放大 |
| 重复率/退化率 | 量化后可能更容易 loop |
| logprob drift | 分析 logits 分布偏移 |
| TTFT/ITL/TPS | 量化是否真带来 serving 收益 |
| 显存曲线 | KV 量化收益是否符合预期 |

## 13. 小结

1. activation 量化难在分布随输入变化，动态 scale 很常见。
2. W8A8 的核心是 activation outlier 处理，SmoothQuant 和 per-token scale 都围绕这个问题。
3. W4A4 更依赖硬件、scale 粒度和 outlier 抑制。
4. FP8 用 exponent 提供范围，MXFP8 再用每 32 值 block scale 提高局部精度。
5. KV Cache 量化显存收益大，但 K/V 误差会反复进入 attention，必须做长上下文验证。
