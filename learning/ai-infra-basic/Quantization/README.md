**中文** | [English](./README_EN.md)

# Quantization

这一章解释推理量化。量化的目标是降低显存、提升带宽效率和加速 kernel，但它会引入精度误差、kernel 选择复杂度和部署兼容性问题。

## 常见量化对象

| 对象 | 例子 | 主要收益 | 主要风险 |
|---|---|---|---|
| Weight | INT8、INT4、GPTQ、AWQ | 降低模型权重显存和读取带宽 | 精度下降、kernel 依赖 |
| Activation | W8A8、FP8 activation | GEMM 更快，带宽更低 | 需要校准或动态 scale |
| KV Cache | FP8 KV、INT8 KV | 长上下文显存更低 | Attention 精度和稳定性 |
| LoRA adapter | 量化 base + FP16 LoRA | 低显存微调和 serving | merge/计算路径更复杂 |

## Weight-only Quantization

Weight-only 只量化权重，activation 通常仍是 FP16/BF16。它适合降低模型加载显存和权重读取带宽。

```text
FP16 Linear:
    y = x_fp16 @ w_fp16

INT4 weight-only:
    w_int4 + scale -> dequant
    y = x_fp16 @ dequant(w_int4)
```

高性能 kernel 不会真的把完整权重先反量化成 FP16 大矩阵，而是在 GEMM 过程中融合 dequant。

## W8A8 / FP8

W8A8 表示 weight 和 activation 都用 8-bit 表示。FP8 则常见于新一代 GPU 上的高吞吐 GEMM。

关键是 scale：

```text
real_value ≈ quant_value * scale
```

scale 可以是 per-tensor、per-channel、per-token、per-group。粒度越细，精度通常越好，但 metadata 和 kernel 成本越高。

## GPTQ 和 AWQ 的直觉

GPTQ 更像基于二阶信息的 post-training quantization，试图让量化后的权重对输出影响更小。

AWQ 更强调保护重要 activation channel，对权重做缩放和量化，让关键通道误差更小。

从 serving 视角看，重点不是背公式，而是知道模型权重格式会决定 loader、scale 布局和 kernel 路径。

## KV Cache Quantization

KV Cache 量化适合长上下文场景，因为 KV 显存会随 token 数线性增长。但它更敏感：

1. Decode 每步都读取历史 KV，误差会持续影响 attention。
2. K 和 Q 的点积对 scale 比较敏感。
3. 不同 layer/head 的分布可能差异明显。
4. Prefix cache 和 KV transfer 都要理解 KV dtype。

## 和 SGLang 的连接点

- Weight loader 需要识别模型权重格式和 quant config。
- Linear kernel 会根据 quant method 选择 GPTQ/AWQ/FP8/W8A8 等路径。
- MoE、attention、lm_head 可能有不同量化支持程度。
- KV Cache dtype 会影响 memory pool、attention backend 和 transfer backend。
- Benchmark 时要同时看吞吐、延迟、显存和质量回归。

## 阅读任务

1. 比较 weight-only quantization 和 W8A8 的区别。
2. 解释 per-channel scale 为什么通常比 per-tensor 更准。
3. 思考 KV Cache 量化为什么比权重量化更容易影响长文本质量。
4. 做性能评估时，除了 TPS，还应该检查哪些质量和稳定性指标。
