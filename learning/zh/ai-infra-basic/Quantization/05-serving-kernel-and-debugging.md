# 05. Serving 中的量化数据流、kernel 与排障

## 1. 从 checkpoint 到 forward 的全链路

量化模型在 serving 引擎中通常经历这条链路：

```text
config.json / quantization_config.json
    -> 解析 quant method
    -> 构建模型层时选择 quant method
    -> create_weights 分配量化参数
    -> weight_loader 加载 checkpoint tensor
    -> process_weights_after_loading 做 pack/reorder/scale 整理
    -> forward 时调用 quant_method.apply
    -> fused quantized kernel 执行 GEMM/MoE/attention
```

| 阶段 | 最容易出错的点 |
|---|---|
| config 解析 | `quant_method`、`bits`、`group_size`、`weight_block_size` 与 checkpoint 不一致 |
| 权重分配 | 逻辑 shape 和 packed physical shape 混淆 |
| 权重加载 | TP 分片 offset 未按 pack factor 调整 |
| post-process | scale 转置、interleave、swizzle、format cast 漏做 |
| forward | backend 不支持当前 dtype，fallback 或报错 |
| 输出 | bias、all-reduce、residual dtype 不匹配 |

## 2. Linear 的量化权重创建

普通 BF16 Linear 只需要：

```text
weight: [O,I] bf16
bias:   [O]
```

W4A16 可能需要：

```text
qweight: [O,I/pack_factor] int32/uint8
scales:  [O,I/group_size]
zeros:   optional [O,I/group_size]
g_idx:   optional [I]
bias:    [O]
```

FP8/MXFP8 可能需要：

```text
weight:            [O,I] fp8
weight_scale:      [O] or [block_n, block_k]
weight_scale_inv:  [ceil(O/block_n), ceil(I/block_k)]
input_scale:       optional
```

MoE expert 还会多一维：

```text
w13_weight: [num_local_experts, 2*intermediate, hidden]
w2_weight:  [num_local_experts, hidden, intermediate]
```

量化后可能变成：

```text
w13_weight_packed: [E, 2*intermediate, hidden/2] uint8
w13_scale:         [E, 2*intermediate, hidden/group]
w2_weight_packed:  [E, hidden, intermediate/2] uint8
w2_scale:          [E, hidden, intermediate/group]
```

## 3. packed W4 的 loader 视角

设逻辑权重：

```text
W: [O,I]
```

如果沿 `I` 维打包，每个 `uint8` 存两个 W4：

```text
W_packed: [O,I/2]
pack_factor = 2
```

如果 tensor parallel 按输入维切 RowParallel：

```text
rank 0 负责 I 的前半
rank 1 负责 I 的后半
```

加载切片时必须把逻辑 offset 转成 packed offset：

```text
physical_offset = logical_offset / pack_factor
physical_size   = logical_size / pack_factor
```

如果不转换，会出现三类问题：

| 问题 | 现象 |
|---|---|
| shape mismatch | 加载时直接报尺寸不一致 |
| silent wrong weight | shape 能对上，但 nibble 对应错位，输出异常 |
| TP rank 不一致 | 单卡能跑，多卡质量崩或 all-reduce 后异常 |

## 4. 为什么要 reorder、swizzle、interleave

checkpoint 格式通常面向保存和通用加载；kernel 格式面向硬件读取。

```text
checkpoint layout:
    便于保存、分片、跨框架交换

kernel layout:
    便于 tensor core tile 读取、coalesced memory access、减少 bank conflict
```

常见 post-process：

| 操作 | 作用 |
|---|---|
| pack | 把低 bit 值塞进 `uint8/int32` |
| unpack | 调试或 fallback 时恢复逻辑低 bit |
| transpose | 适配 GEMM 的 A/B operand 布局 |
| interleave | 把 scale 或 weight 按 kernel tile 交错排列 |
| swizzle | 改变内存顺序，让硬件读取更连续 |
| format cast | NPU/GPU 专用格式转换，例如 int4pack |
| padding | 让 hidden/intermediate 满足 32、64、128、256 对齐 |

这些操作不改变数学含义，但改变物理存储。如果一个 checkpoint 已经是 kernel layout，再重复处理一次也会错。

## 5. fused dequant GEMM 的执行

W4A16 的高性能执行通常不是：

```text
qweight -> full dequant W_bf16 [O,I]
X_bf16 @ W_bf16
```

而是：

```text
for each GEMM tile:
    load packed W4 tile
    unpack nibbles
    load group scales / zero points
    dequant tile to register/shared memory
    multiply with BF16 activation tile
    accumulate
    write BF16/FP16 output
```

这叫 fused dequant GEMM。收益来自：

| 收益 | 说明 |
|---|---|
| 少读权重 | HBM 中读 W4 packed，而不是 BF16 |
| 少临时内存 | 不生成完整反量化权重 |
| tile 级反量化 | scale 在小范围内复用 |
| tensor core 友好 | kernel 可按硬件 tile 设计 |

## 6. W8A8 / FP8 GEMM 的执行

W8A8 INT8：

```text
X_bf16
    -> dynamic quant -> X_int8 + scale_x

W_int8 + scale_w
    -> INT8 GEMM -> int32 accumulation
    -> apply scale_x * scale_w
    -> output bf16/fp16
```

FP8：

```text
X_bf16
    -> FP8 cast + activation scale

W_fp8 + weight scale
    -> FP8 tensor core GEMM
    -> output bf16/fp16/fp32
```

MXFP8：

```text
X_bf16
    -> split into 32-value blocks
    -> FP8 payload + E8M0 block scales

W_mxfp8
    -> FP8 payload + E8M0 block scales

kernel:
    consumes payload and block scales together
```

对 MXFP8，shape 对齐经常是硬条件：

```text
K dimension % 32 == 0
scale last dim = K / 32
某些 MoE kernel 还要求 N、K、intermediate 对齐到 128/256
```

## 7. MoE 量化的额外复杂度

MoE 有多组 expert 权重：

```text
gate/up projection: [E, 2I, H]
down projection:    [E, H, I]
```

每个 token 只进 Top-K expert：

```text
router -> topk_ids, topk_weights
dispatch hidden_states to experts
expert GEMM
combine outputs
```

量化后复杂度增加：

| 问题 | 说明 |
|---|---|
| per-expert scale | 每个 expert 的分布不同，scale 不能简单共享 |
| EP/TP 分片 | expert parallel 和 tensor parallel 同时影响 weight layout |
| grouped GEMM | 多个 expert 的 token 数不同，需要 grouped kernel |
| padding | 每个 expert 的 `M/N/K` 维度要满足量化 kernel 对齐 |
| router dtype | router logits 通常保持 BF16/FP32，不能随便低 bit |
| w13 融合 | gate 和 up 合并后 scale 与 shard mapping 更复杂 |

## 8. KV Cache 量化的数据流

```text
Attention layer 产生:
    K_new, V_new: [T, Nkv, D] bf16

写 cache 前:
    K_q, K_scale = quantize(K_new)
    V_q, V_scale = quantize(V_new)

KV memory pool 保存:
    K_q/V_q payload
    K_scale/V_scale metadata
    block/page index
    dtype info

Attention 读取:
    按 request 的 cache index 找历史页
    读取 payload + scale
    kernel 内部解释为低 bit KV
```

如果有 prefix cache：

```text
prefix entry = token ids + KV payload + KV scale + dtype + layout version
```

如果有 KV transfer：

```text
sender 必须发送 payload 和 scale
receiver 必须按同一 dtype/layout 写入本地 memory pool
```

只传低 bit payload、不传 scale，接收端无法恢复真实 K/V。

## 9. 量化通信

多卡 tensor parallel 中，RowParallelLinear 常需要 all-reduce：

```text
每个 rank 计算部分输出
all-reduce 求和
```

通信也可以量化：

```text
partial_output_bf16
    -> quantize communication tensor
    -> all-reduce low bit / compressed representation
    -> dequant
```

收益：

```text
降低 interconnect 带宽
```

风险：

```text
all-reduce 是求和，量化误差会跨 rank 叠加；
如果 scale 选择不好，可能引入明显数值偏差。
```

通信量化通常比权重量化更依赖 workload、batch size、网络拓扑和容错策略。

## 10. 显存估算

### 10.1 权重显存

```text
weight_bytes =
    num_params * bits_per_weight / 8
    + scale_bytes
    + zero_point_bytes
    + layout_padding
```

以 70B 为例：

```text
BF16 主体:
    70B * 2 bytes = 140 GB

W4 主体:
    70B * 4 / 8 = 35 GB
```

如果 group size 是 128，scale 用 FP16，粗略 scale overhead：

```text
每 128 个权重 1 个 scale
scale overhead per param = 2 / 128 = 0.015625 bytes
70B scale overhead ≈ 1.09 GB
```

还要加 zero point、padding、MoE 特殊 metadata、embedding/lm_head 未量化部分等。

### 10.2 KV Cache 显存

```text
kv_bytes =
    batch_total_tokens
    * num_layers
    * 2
    * num_kv_heads
    * head_dim
    * bytes_per_kv_element
    + kv_scale_bytes
    + allocator_metadata
```

KV 量化对长上下文特别有效，因为 `batch_total_tokens` 会持续增长。

## 11. 性能判断

量化收益要分阶段看：

| 阶段 | 主要瓶颈 | 量化可能带来的收益 |
|---|---|---|
| 模型加载 | 磁盘、CPU、GPU copy、post-process | 低 bit 文件小，但 post-process 可能更久 |
| Prefill | 大矩阵 GEMM、attention | W8A8/FP8/MXFP8 可能加速 |
| Decode | 权重读取、KV 读取、小 batch kernel overhead | weight-only 和 KV quant 可能收益明显 |
| MoE | expert GEMM、dispatch/combine、EP 通信 | expert W4/FP8 可降显存和带宽 |
| 多卡 | all-reduce/all-to-all | 通信量化可能有收益 |

不要只看 tokens/sec。至少同时看：

```text
TTFT: 首 token 延迟
ITL: inter-token latency
TPS: 总吞吐
显存峰值
GPU 利用率
kernel fallback 比例
质量指标
```

## 12. 常见故障与定位

| 现象 | 可能原因 | 定位方式 |
|---|---|---|
| 加载时报 shape mismatch | packed physical shape 与 logical shape 混淆 | 打印 checkpoint tensor shape、param shape、pack factor |
| 能跑但输出乱码 | scale/zero point 方向错、nibble 顺序错、layout 重排错 | 用小层做 dequant 对比，检查最大误差 |
| 单卡正常多卡异常 | TP shard offset 没处理 pack factor 或 scale shard | 对比每 rank 加载区间 |
| 显存没下降 | 层被 ignored、fallback 到 BF16、临时 dequant buffer 太大 | 打印实际 parameter dtype 和 allocator 统计 |
| 吞吐下降 | kernel 不支持低 bit，走 fallback；动态 scale 开销大 | profiler 看 kernel 名称和时间 |
| 长上下文质量下降 | KV Cache scale 太粗或 K 量化误差大 | 跑 passkey/retrieval，比较 FP16 KV |
| MoE 质量异常 | expert scale、w13 gate/up 顺序、router dtype、expert offset 错 | 单独测试 routed expert 输出 |
| 某些 batch size 报错 | graph/static shape 或 kernel alignment 不满足 | 检查 `H/I/O` 是否被 32/64/128 整除 |

## 13. 最小化数值对齐测试

调试量化层时，先不用完整模型。对单层做：

```text
1. 随机生成小输入 X
2. 加载原始 FP 权重 W
3. 加载量化权重 qW + scale
4. 手写 reference dequant 得到 W_hat
5. 比较:
       Y_ref = X @ W
       Y_q_ref = X @ W_hat
       Y_kernel = quant_kernel(X, qW, scale)
6. 分别检查:
       max_abs(Y_q_ref - Y_kernel)
       max_abs(Y_ref - Y_q_ref)
```

读法：

```text
Y_q_ref 和 Y_kernel 差很大:
    kernel layout、pack、scale 使用方式可能错。

Y_ref 和 Y_q_ref 差很大:
    量化算法本身质量不够，或 scale 粒度/校准数据有问题。
```

## 14. 质量回归建议

量化上线前至少分三层验证：

| 层级 | 内容 |
|---|---|
| 数值层 | 单层输出误差、整模型 logits drift、perplexity |
| 任务层 | 代码、数学、中文、英文、RAG、工具调用、多模态 |
| serving 层 | 多 batch、长上下文、prefix cache、KV transfer、spec decode、LoRA 混用 |

对采样模型，还要固定随机种子并比较分布，而不是只比较完全相同输出。

```text
greedy:
    可比较 token match rate

sampling:
    应比较 logprob、KL drift、任务指标和人工 spot check
```

## 15. 读源码时的关键文件类型

不同 serving 引擎名字不同，但职责通常类似：

| 文件类型 | 关注点 |
|---|---|
| quant config | 支持哪些 quant method、bits、dtype、capability |
| linear layer | quant_method 如何创建权重和 forward |
| parameter class | packed parameter、scale parameter 如何加载和分片 |
| weight loader | checkpoint tensor 到 runtime parameter 的映射 |
| fp8/int4 kernel wrapper | scale layout、activation quant、output dtype |
| MoE runner | expert quant info、grouped GEMM、dispatch/combine |
| KV cache method | KV dtype、scale 加载、attention backend 支持 |
| server args | 用户如何开启 quantization、KV dtype、backend |

## 16. 小结

1. 量化 serving 是配置、权重、layout、kernel、分片、cache 的共同协议。
2. packed W4 改变物理 shape，TP 分片和 loader 必须按 `pack_factor` 处理。
3. 高性能来自 fused kernel，而不是先反量化成完整 BF16 权重。
4. MXFP8/FP8 block quant 的 scale dtype、block shape、row/column layout 都是硬约束。
5. 排障时先区分“kernel 实现错”和“量化算法质量不够”，不要直接在整模型里盲猜。
