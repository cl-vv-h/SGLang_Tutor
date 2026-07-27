# Attention Kernel 教学入口

这个目录用纯 PyTorch 教学代码解释 attention kernel 优化背后的核心想法。重点不是替代 CUDA/Triton 实现，而是把计算顺序、softmax 稳定性、KV Cache 访问和 decode 阶段的瓶颈讲清楚。

## 文件说明

| 文件 | 主题 | 适合关注 |
|---|---|---|
| [flash_attention_tutorial.py](./flash_attention_tutorial.py) | FlashAttention forward 教学版 | 分块 attention、online softmax、避免显式保存完整 attention matrix |
| [flash_decoding_tutorial.py](./flash_decoding_tutorial.py) | FlashDecoding 教学版 | decode 单 token 查询、KV block partial result、跨 block 合并 |

## 推荐阅读顺序

1. 先看 `flash_attention_tutorial.py` 的 `naive_attention`，确认普通 attention 的输入输出形状。
2. 再看 `flash_attention_forward_tutorial`，理解为什么可以边扫描 K/V block 边维护 softmax 的 `m` 和 `l`。
3. 然后看 `flash_decoding_tutorial.py` 的 `naive_decode_attention`，把 prefill 阶段的大矩阵 attention 和 decode 阶段的单 token attention 区分开。
4. 最后看 `flash_decoding_attention`，理解长上下文 decode 如何按 KV block 拆分，并在结尾合并 partial softmax 结果。

## 和 SGLang 的连接点

- SGLang 的 `ModelRunner` 会根据 forward mode 选择 prefill 或 decode 路径。
- prefill 更像大块矩阵计算，decode 更容易被 KV Cache 读取和 memory bandwidth 限制。
- FlashAttention 解释了为什么 attention backend 要关心分块、数值稳定和显存读写。
- FlashDecoding 解释了长上下文 decode 时为什么要围绕 KV block 做并行化。

## 运行示例

```bash
python learning/ai-infra-basic/Attention_Kernel/flash_attention_tutorial.py
python learning/ai-infra-basic/Attention_Kernel/flash_decoding_tutorial.py
```
