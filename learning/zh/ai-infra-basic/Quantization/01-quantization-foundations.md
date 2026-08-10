# 01. 量化基础：从浮点数到低 bit 表示

## 1. 量化在解决什么问题

LLM 推理中最贵的资源通常有三类：

| 资源 | 量化能改善什么 | 典型例子 |
|---|---|---|
| 模型常驻显存 | 权重从 BF16/FP16 降到 8-bit、4-bit 或更低 | 70B BF16 约 140 GB，W4 权重主体约 35 GB，加上 scale 后略高 |
| 显存带宽 | GEMM、attention、MoE 每次读取的数据更少 | decode 阶段反复读权重和 KV Cache |
| 计算吞吐 | 某些硬件对 INT8/FP8/FP4 有更高 tensor core 吞吐 | W8A8、FP8 GEMM、MXFP8 GEMM |

量化的核心不是“压缩文件”，而是重新定义一个近似计算：

```text
原始计算:
    y = x_float @ w_float

量化计算:
    q_w, scale_w, zero_w = quantize(w_float)
    q_x, scale_x, zero_x = quantize(x_float)   # 如果量化 activation
    y ≈ dequant(q_x, scale_x, zero_x) @ dequant(q_w, scale_w, zero_w)
```

高性能推理不会真的把整个 `q_w` 先反量化成大块 FP16 权重再做 GEMM，而是在 kernel 内部融合：

```text
读取低 bit 权重 -> 解包/反量化局部 tile -> tensor core/GEMM -> 输出 BF16/FP16/FP32
```

## 2. 最基本的均匀量化

最常见的 INT 量化是 affine quantization。它用一个缩放系数 `scale` 和一个整数偏移 `zero_point`，把实数映射到整数格点。

```text
量化:
    q = clamp(round(x / scale) + zero_point, q_min, q_max)

反量化:
    x_hat = (q - zero_point) * scale
```

| 符号 | 含义 |
|---|---|
| `x` | 原始浮点值 |
| `q` | 存储或计算时使用的整数值 |
| `x_hat` | 反量化后近似恢复的浮点值 |
| `scale` | 一个整数步长代表多少真实数值 |
| `zero_point` | 真实值 0 对应到整数网格中的位置 |
| `q_min, q_max` | 当前 bit-width 能表示的整数范围 |

读法很简单：

```text
先把 x 除以 scale，得到“落在哪个整数格点附近”；
再四舍五入到整数；
如果超出可表示范围，就截断到 q_min/q_max；
恢复时再把整数格点乘回 scale。
```

一个 4-bit unsigned 量化的整数范围是：

```text
q_min = 0
q_max = 15
```

一个 4-bit signed 对称量化的常见范围是：

```text
q_min = -8
q_max = 7
```

为什么 signed INT4 不是 `-7..7`？因为 4 个 bit 一共有 16 个编码，二补码天然覆盖 `-8..7`。有些 kernel 为了对称性会不用 `-8`，但这会浪费一个编码，需要看具体格式。

## 3. 对称量化与非对称量化

### 3.1 对称量化

对称量化令 `zero_point = 0`，整数范围围绕 0。

```text
q = clamp(round(x / scale), q_min, q_max)
x_hat = q * scale
```

scale 常按最大绝对值计算：

```text
scale = max(abs(x)) / q_max_abs
```

| 优点 | 缺点 |
|---|---|
| 反量化简单，kernel 友好 | 如果数据分布偏正或偏负，会浪费一侧表示范围 |
| 适合权重，因为权重常接近零中心分布 | outlier 会把 scale 拉大，使多数小值精度变差 |

### 3.2 非对称量化

非对称量化允许 `zero_point != 0`，更适合非零中心分布。

```text
scale = (x_max - x_min) / (q_max - q_min)
zero_point = round(q_min - x_min / scale)
```

| 优点 | 缺点 |
|---|---|
| 对偏移分布更省格点 | kernel 需要处理 zero point，计算更复杂 |
| activation 有时更适合非对称 | zero point 的 dtype、广播形状和 fused kernel 支持更麻烦 |

## 4. scale 粒度

同样是 INT4，精度差异可能非常大，因为 scale 的粒度不同。

| 粒度 | scale 形状示例 | 直观含义 | 常见场景 |
|---|---:|---|---|
| per-tensor | `[1]` | 整个张量共用一个 scale | 最简单、最快，但最粗 |
| per-channel | `[O]` | 每个输出通道一个 scale | Linear 权重常用 |
| per-group | `[O, I/group_size]` | 每个输出通道沿输入维按组分 scale | W4A16/GPTQ/AWQ 常用 |
| per-block | `[O/block_n, I/block_k]` | 矩阵按 tile 共享 scale | FP8 block quant、MXFP8 |
| per-token | `[T]` 或 `[T, group]` | 每个 token 的 activation 动态 scale | W8A8/FP8 activation |
| per-head/layer | `[L,H]` 或 `[layer,head]` | 给 KV Cache 或 attention 相关张量单独 scale | KV Cache quant |
| per-expert | `[E,...]` | 每个 MoE expert 独立 scale | MoE 权重量化 |

粒度越细，通常越准，因为不同区域不用被同一个 outlier 绑架。但粒度越细，metadata 越多，kernel 也越难写。

### 4.1 group size 的含义

设 Linear 权重逻辑形状：

```text
W: [O, I]
```

如果做 per-group W4，`group_size = 128`，则每个输出通道每 128 个输入维度共享一组 scale：

```text
qweight: [O, I] 的逻辑 INT4 值
scale:   [O, ceil(I / 128)]
```

计算某个元素 `W[o, i]` 的近似值：

```text
group_id = floor(i / 128)
W_hat[o, i] = dequant(qweight[o, i], scale[o, group_id], zero_point[o, group_id])
```

group size 越小，scale 越细，质量通常越好；但 scale 数量和内存访问会增加。常见 group size 有 `32`、`64`、`128`、`256`，具体取决于模型、硬件和 kernel。

## 5. scale 怎么选

### 5.1 MinMax

MinMax 是最直接的方法：

```text
scale = (x_max - x_min) / (q_max - q_min)
```

它的问题是极端 outlier 会把范围拉得很大。

```text
大多数值: -0.2 到 0.2
少数 outlier: 12.0
```

如果整个 tensor 共用一个 scale，绝大多数小值会挤在很少的整数格点里，实际有效 bit 变低。

### 5.2 Percentile / clipping

为了避免少量 outlier 支配 scale，可以先选择截断阈值：

```text
clip_min, clip_max = percentile(x, 0.1%), percentile(x, 99.9%)
```

然后只按截断后的范围量化：

```text
x_clipped = clamp(x, clip_min, clip_max)
```

这会牺牲 outlier 的精度，换取多数值更细的量化格点。

### 5.3 MSE 搜索

也可以搜索一个截断阈值，使反量化误差最小：

```text
error = mean((x - dequant(quantize(x, scale)))^2)
```

读法：

```text
尝试多个 scale 或 clipping 阈值；
每次把张量量化再反量化；
选择平均平方误差最小的那一个。
```

MSE 更稳，但需要校准样本或额外搜索时间。

## 6. 静态量化、动态量化和离线量化

| 类型 | qparams 什么时候确定 | 典型场景 | 优缺点 |
|---|---|---|---|
| 离线权重量化 | 模型转换时 | GPTQ、AWQ、AutoRound、NF4 | 运行时简单，权重已压缩 |
| 静态 activation 量化 | 校准阶段确定 | W8A8 INT8、部分 FP8 | 运行时快，但分布漂移时风险高 |
| 动态 activation 量化 | 每次 forward 现场计算 | per-token INT8/FP8、MXFP8 activation | 更准，但多了求 amax/scale 的 kernel 成本 |
| QAT | 训练时模拟量化误差 | 极低 bit 或质量要求高 | 质量好但训练成本高 |

LLM serving 中常见组合是：

```text
权重: 离线量化并保存
activation: decode/prefill 运行时动态量化
KV Cache: 创建或写入 cache 时量化
```

## 7. 量化误差来自哪里

量化误差可以写成：

```text
e = x_hat - x
```

但真正影响模型输出的不是单个元素误差，而是这些误差进入矩阵乘法、attention 和 softmax 后的结果。

对 Linear：

```text
y = x @ W
y_hat = (x + e_x) @ (W + e_w)
```

展开后：

```text
y_hat ≈ x @ W
      + x @ e_w
      + e_x @ W
      + e_x @ e_w
```

| 项 | 含义 |
|---|---|
| `x @ W` | 原始结果 |
| `x @ e_w` | 权重量化误差被 activation 放大 |
| `e_x @ W` | activation 量化误差被权重放大 |
| `e_x @ e_w` | 两边都量化后的二阶小项 |

这解释了为什么只做 weight-only 往往比 W4A4 更容易保质量：activation 不量化时，少了 `e_x @ W` 和 `e_x @ e_w`。

## 8. PTQ 与 QAT 的区别

| 路线 | 全称 | 是否重新训练 | 常见算法 | 适合场景 |
|---|---|---:|---|---|
| PTQ | Post-Training Quantization | 否 | RTN、GPTQ、AWQ、SmoothQuant、AutoRound、HQQ | 部署已有模型 |
| QAT | Quantization-Aware Training | 是 | fake quant、STE、量化感知 finetune | 极低 bit、质量要求很高 |
| QAT-lite | 少量校准或短时优化 | 少量 | AutoRound/SignRound 类 rounding 优化 | 介于 PTQ 和 QAT 之间 |

PTQ 的优势是快，不需要原始训练流程；QAT 的优势是模型能适应量化噪声，但成本更高。

## 9. 读量化配置时先看什么

拿到一个量化模型，先读这些字段：

| 字段 | 说明 |
|---|---|
| `quant_method` | GPTQ、AWQ、fp8、mxfp8、bitsandbytes、compressed-tensors、modelopt 等 |
| `bits` / `weight_bits` | 权重 bit 数 |
| `activation_scheme` | static、dynamic、per-token、per-tensor 等 |
| `group_size` | 权重沿输入维共享 scale 的组大小 |
| `sym` / `zero_point` | 对称还是非对称 |
| `weight_block_size` | block quant 的二维 tile，比如 `[1,32]`、`[128,128]` |
| `kv_cache_quant_algo` | KV Cache 是否 FP8/INT8/MXFP8 |
| `ignored_layers` | 哪些层不量化，常见于 embedding、lm_head 或敏感层 |
| `packed_modules_mapping` | QKV、gate/up 等 fused 权重如何对应原始子模块 |

很多线上问题不是数学错，而是这些字段和实际 checkpoint tensor 布局不一致。

## 10. 小结

1. 量化由三件事组成：低 bit 值、scale/zero point、消费这些值的 kernel。
2. bit 数只说明容量，scale 粒度决定了大量精度表现。
3. weight-only 容易落地，W8A8/FP8 需要更强的 activation 处理，W4A4 更依赖硬件和算法。
4. `group_size`、`packed`、`scale_inv`、`weight_block_size` 都是运行时协议，不是文档装饰。
5. 判断量化方案是否靠谱，要同时看质量、显存、吞吐、延迟、kernel 支持和 fallback 路径。
