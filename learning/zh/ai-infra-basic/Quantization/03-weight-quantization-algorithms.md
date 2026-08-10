# 03. 权重量化算法：从 RTN 到 GPTQ、AWQ、AutoRound、HQQ、NF4

## 1. 权重量化为什么最常见

权重有三个特点：

| 特点 | 对量化的意义 |
|---|---|
| 模型加载后长期不变 | 可以离线慢慢量化，运行时直接加载低 bit 权重 |
| 占常驻显存大 | 4-bit 权重通常能显著降低模型显存 |
| 每次 forward 都会读 | decode 阶段权重读取带宽是重要瓶颈 |

Weight-only quantization 的基本形式是：

```text
y = x_fp16 @ dequant(q_w, scale_w, zero_w)
```

activation 仍是 BF16/FP16，因此它通常比 W8A8、W4A4 更容易保质量。

## 2. 离线权重量化的一般流程

```text
1. 选择量化对象:
       q_proj/k_proj/v_proj/o_proj、gate/up/down、MoE experts、lm_head 等

2. 准备校准数据:
       几百到几千条 prompt，覆盖目标任务和长度分布

3. 收集统计:
       activation、Hessian 近似、channel 重要性、layer sensitivity

4. 计算量化参数:
       scale、zero point、group size、clipping、rounding offset

5. 生成低 bit 权重:
       INT4/INT8/FP8/NF4 payload + scale metadata

6. 打包和重排:
       packed W4、Marlin layout、CUTLASS layout、NPU int4pack、MoE expert layout

7. 保存配置:
       quantization_config.json、safetensors、compressed-tensors metadata

8. 质量与性能验证:
       perplexity、任务集、长上下文、吞吐、TTFT、ITL、显存
```

## 3. RTN：Round-To-Nearest

RTN 是最简单的量化：

```text
q = round(w / scale)
w_hat = q * scale
```

| 优点 | 缺点 |
|---|---|
| 快，不需要校准数据 | 低 bit 下误差大 |
| 容易实现 | 不知道哪些权重对输出更重要 |
| 可作为 baseline | 对 3-bit/4-bit 大模型通常不够稳 |

RTN 的隐含假设是：

```text
每个权重元素的误差同等重要。
```

这在 LLM 中通常不成立。不同 channel、不同层、不同 expert 的重要性差异很大。

## 4. GPTQ：用二阶信息决定量化误差怎么传播

GPTQ 的目标不是让每个权重元素都最接近原值，而是让量化后的层输出尽量接近原输出。

对一个 Linear：

```text
Y = X @ W
Y_hat = X @ W_hat
```

希望最小化：

```text
loss = || X @ W - X @ W_hat ||^2
```

把误差写成：

```text
DeltaW = W_hat - W
loss = || X @ DeltaW ||^2
```

这里 `X` 是校准 activation。若某个权重方向对应的 activation 很大，那个方向的量化误差会更影响输出。

GPTQ 使用近似二阶信息：

```text
H ≈ X^T X
```

| 符号 | 含义 |
|---|---|
| `X` | 当前层输入 activation，来自校准数据 |
| `H` | Hessian 或二阶敏感性近似 |
| `DeltaW` | 权重量化误差 |

直观读法：

```text
如果某个输入方向经常被激活，H 在这个方向上会更大；
这个方向上的权重量化误差更容易影响输出；
GPTQ 量化一个权重后，会用二阶信息补偿剩余权重。
```

### 4.1 GPTQ 的简化流程

```text
for each layer:
    收集校准 activation X
    计算 H ≈ X^T X
    按列或 block 遍历权重
    对当前权重做低 bit rounding
    用 H 的逆或近似逆把误差传播到未量化部分
    保存 qweight、scale、zero point、group index
```

### 4.2 GPTQ 关注点

| 参数 | 含义 | 影响 |
|---|---|---|
| `bits` | 权重 bit 数 | 4-bit 常见，3/2-bit 更难 |
| `group_size` | 每组共享 scale 的输入维长度 | 小 group 更准但 metadata 多 |
| `desc_act` | 是否按 activation 重要性重排 | 可能提升质量，但影响 kernel 兼容 |
| `sym` | 对称量化 | kernel 更简单 |
| backend | Marlin、ExLlama、CUTLASS、NPU 等 | 决定 packed layout 和速度 |

## 5. AWQ：保护 activation 重要的权重通道

AWQ 的关键观察：

```text
不是所有权重同等重要；
少量 salient weights 对输出影响很大；
判断 salient channel 时应看 activation，而不是只看 weight 本身。
```

AWQ 不直接保存混合精度权重，而是通过等价缩放保护重要通道。

设 Linear：

```text
Y = X @ W
```

引入一个按 channel 的正 scale `s`：

```text
Y = (X / s) @ (W * s)
```

这在数学上等价，因为一个方向缩小、另一个方向放大，乘积不变。

| 变换 | 作用 |
|---|---|
| `W * s` | 放大重要权重通道，使其量化相对误差更小 |
| `X / s` | 抵消权重放大，保持原始函数不变 |
| 搜索 `s` | 根据校准 activation 找到质量和量化误差的折中 |

读法：

```text
AWQ 不是真的让 1% 权重保持 FP16；
它把重要 channel 放大，让 uniform INT4 量化时这些 channel 的信息不那么容易丢。
```

### 5.1 AWQ 的简化流程

```text
for each quantized Linear:
    收集输入 activation
    找到 activation 重要的 channel
    搜索 per-channel scale s
    对 W * s 做 INT4/INT3 量化
    运行时用等价缩放或融合后的参数恢复计算语义
```

### 5.2 AWQ 与 GPTQ 的区别

| 维度 | GPTQ | AWQ |
|---|---|---|
| 核心思想 | 用二阶信息补偿 rounding 误差 | 用 activation 感知缩放保护重要通道 |
| 是否需要校准数据 | 需要 | 需要 |
| 是否反向传播 | 不需要 | 不需要 |
| 典型格式 | W4A16、W3A16 | W4A16 |
| 工程特点 | 可能有 act-order/reorder | 强调硬件友好和少重排 |

## 6. SmoothQuant：把 activation 难题迁移到 weight

SmoothQuant 主要解决 W8A8，尤其是 activation outlier 导致 INT8 activation 难量化的问题。

核心等价变换：

```text
Y = X @ W
Y = (X / s) @ (diag(s) @ W)
```

也可以按 channel 写成：

```text
X_smooth[:, i] = X[:, i] / s_i
W_smooth[i, :] = W[i, :] * s_i
```

读法：

```text
activation 中某些 channel 有 outlier，导致 A8 scale 被拉大；
权重通常比 activation 更容易量化；
所以把一部分幅度从 activation 挪到 weight；
平滑后的 activation 更适合 INT8，权重承受增加的量化难度。
```

SmoothQuant 的超参数常写作 `alpha`：

```text
s_i = max(|X_i|)^alpha / max(|W_i|)^(1 - alpha)
```

| `alpha` 趋势 | 效果 |
|---|---|
| 更接近 0 | 更少迁移 activation 难度 |
| 更接近 1 | 更多把 activation 范围迁移给 weight |

SmoothQuant 是理解 W8A8 的关键：activation 量化的核心难题不是 bit 数本身，而是 outlier 和动态范围。

## 7. AutoRound / SignRound：优化 rounding 决策

普通 rounding 是：

```text
q = round(w / scale)
```

但 `0.49` 一定要向下、`0.51` 一定要向上吗？对单个元素看是这样，对整层输出不一定。

AutoRound/SignRound 类方法把 rounding 看成可优化变量：

```text
w_hat = quantize(w, rounding_offset)
目标: 让层输出或模型输出误差最小
```

简化流程：

```text
加载原模型和校准数据
冻结原始权重
为 rounding 或 clipping 引入可优化变量
短步数优化这些变量
导出普通 INT4/INT8 checkpoint
```

它介于 PTQ 和 QAT 之间：

| 特点 | 说明 |
|---|---|
| 不做完整训练 | 不更新原始模型权重 |
| 需要少量优化 | 比 RTN/GPTQ/AWQ 更耗时 |
| 推理无额外开销 | 导出后仍是普通量化权重 |
| 对低 bit 有帮助 | 尤其 2-bit、3-bit、4-bit |

## 8. HQQ：无校准数据的快速权重量化

HQQ 的定位是 calibration-free weight quantization。它不依赖外部校准样本，而是在权重本身上求解低 bit 近似。

常见目标可以理解成：

```text
找到 qweight、scale、zero_point
使 dequant(qweight, scale, zero_point) 接近原始 W
```

HQQ 使用 half-quadratic optimization 快速求解，支持 8、4、3、2、1 bit 等配置。

| 优点 | 风险 |
|---|---|
| 不需要校准数据 | 无法利用目标任务 activation 分布 |
| 量化速度快 | 极低 bit 仍需认真验证 |
| 可用于多模态和 PEFT | 生产推理速度取决于后端 kernel |

HQQ 与 GPTQ/AWQ 的核心差别：

```text
GPTQ/AWQ:
    用校准 activation 判断哪些误差更重要

HQQ:
    不使用校准数据，主要从权重近似本身优化
```

## 9. NF4 与 QLoRA

QLoRA 使用的 NF4 是一种 4-bit 非均匀码本。它不是均匀 INT4。

均匀 INT4 的格点类似：

```text
-1.0, -0.875, -0.75, ..., 0.875, 1.0
```

NF4 的直觉是：

```text
预训练权重经过归一化后常近似零中心正态分布；
正态分布的大多数值靠近 0；
所以 16 个码本值应该在 0 附近更密，在尾部更疏。
```

QLoRA 还使用 double quantization：

```text
第一次量化:
    W -> W_NF4 + quantization constants

第二次量化:
    quantization constants -> 更低 bit 的 constants
```

读法：

```text
不仅权重本身压到 4-bit；
连每个 block 的 scale/常数也再压一次，进一步降低显存。
```

QLoRA 的典型计算：

```text
base_out = X_bf16 @ dequant(W_nf4)
lora_out = X_bf16 @ A_bf16 @ B_bf16
Y = base_out + lora_out
```

| 适合 | 不适合 |
|---|---|
| 低显存微调 | 只追求最高生产推理吞吐时直接照搬 |
| 冻结 base、训练 LoRA | 没有高效 4-bit kernel 的 serving |
| PEFT 实验 | 未验证 merged adapter 后的质量 |

## 10. 低 bit 算法怎么选择

| 目标 | 优先考虑 |
|---|---|
| 快速拿到 W4A16 baseline | AWQ 或 GPTQ |
| 有较好校准数据，追求质量 | GPTQ、AWQ、AutoRound |
| 不想准备校准数据 | HQQ、bitsandbytes/NF4 |
| W8A8 serving | SmoothQuant、FP8/PTQ、动态 activation quant |
| 极低 bit 2/3-bit | AutoRound、GPTQ 变体、QuIP/旋转类方法，需要强验证 |
| QLoRA 微调 | NF4 + double quant + LoRA |
| MoE expert 权重量化 | 关注 expert layout、EP/TP 分片、对应 fused MoE kernel |

## 11. 校准数据为什么重要

校准数据决定算法看到的 activation 分布。

```text
如果校准数据都是短英文问答，
但线上主要跑长中文代码生成，
那么 activation 分布、router 分布、KV 分布都可能不同。
```

好的校准集通常需要覆盖：

| 维度 | 示例 |
|---|---|
| 语言 | 中文、英文、代码、多语言 |
| 长度 | 短问答、长上下文、长输出 |
| 任务 | 对话、数学、代码、工具调用、RAG |
| 模态 | 文本、图像、多模态输入 |
| batch 行为 | prefill-heavy、decode-heavy、MoE 路由分布 |

## 12. 哪些层不该轻易量化

常见保留高精度的部分：

| 层或张量 | 原因 |
|---|---|
| embedding | token 表大且访问稀疏，量化收益和 kernel 支持不一定好 |
| lm_head | 直接影响 logits，词表大，误差影响 sampling |
| final norm | 对 logits 前分布敏感 |
| router/gating | MoE expert 选择对小误差敏感 |
| very small layers | kernel overhead 可能超过收益 |
| 特定敏感层 | 有些 layer 对 perplexity 或任务质量影响异常大 |

敏感层分析常通过逐层量化和质量回归来做：

```text
for each layer:
    只量化这一层
    跑校准评估
    记录质量下降
```

质量下降大的层可以保持 BF16/FP16，或者使用更高 bit。

## 13. 权重量化输出里通常有什么

一个 W4A16 Linear 可能包含：

```text
weight_packed:  uint8/int32 packed INT4
weight_scale:   float16/float32/float8 scale
weight_zp:      optional zero point
g_idx:          optional group index or act-order metadata
quant_config:   bits, group_size, sym, backend, packing_format
```

一个 FP8 Linear 可能包含：

```text
weight:            float8_e4m3fn
weight_scale:      per-tensor/per-channel float32
weight_scale_inv:  blockwise scale inverse
input_scale:       static activation scale, optional
```

一个 MXFP8 block quant Linear 可能包含：

```text
weight:            FP8 payload
weight_scale_inv:  uint8 E8M0/UE8M0 block scales
weight_block_size: [1, 32]
```

## 14. 小结

1. RTN 是最简单 baseline，但不考虑权重重要性。
2. GPTQ 用二阶信息降低量化误差对层输出的影响。
3. AWQ 用 activation 统计保护重要 channel，强调硬件友好。
4. SmoothQuant 把 activation outlier 难题迁移到 weight，是 W8A8 的核心思想之一。
5. AutoRound/SignRound 优化 rounding 决策，导出后通常没有额外推理开销。
6. HQQ 不需要校准数据，适合快速量化，但仍要看 kernel 与质量。
7. NF4/QLoRA 面向低显存微调，不等于天然高吞吐生产推理格式。
