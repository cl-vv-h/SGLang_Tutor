# AI Infra 基础专题

这个目录用于补齐阅读 SGLang 源码前后的基础知识。它不直接复刻生产级实现，而是用更小的 Python demo 和讲义拆开 LLM serving 中最常见的机制：attention kernel、LoRA 适配器、分布式并行和调度优化。

## 目录结构

| 目录 | 内容 | 建议先看 |
|---|---|---|
| [Attention_Kernel](./Attention_Kernel/) | FlashAttention 与 FlashDecoding 的教学版实现 | [Attention_Kernel/README.md](./Attention_Kernel/README.md) |
| [LoRA](./LoRA/) | LoRA、QLoRA、DoRA、AdaLoRA 的简化训练与模块替换 demo | [LoRA/README.md](./LoRA/README.md) |
| [Parallel_Strategy](./Parallel_Strategy/) | DP、TP、PP、SP/CP、EP 推理并行策略 | [Parallel_Strategy/README.md](./Parallel_Strategy/README.md) |
| [Schedule_Optimization](./Schedule_Optimization/) | Prefill/Decode、KV Cache、Chunked Prefill 调度模拟 | [Schedule_Optimization/README.md](./Schedule_Optimization/README.md) |

## 建议学习路线

1. 先读 [Schedule_Optimization/README.md](./Schedule_Optimization/README.md)，理解 LLM 推理为什么分成 prefill 和 decode。
2. 再读 [Attention_Kernel/README.md](./Attention_Kernel/README.md)，把 attention 的计算形状、内存访问和 kernel 优化联系起来。
3. 然后读 [Parallel_Strategy/README.md](./Parallel_Strategy/README.md)，理解多卡推理时到底切的是请求、矩阵、层、序列还是专家。
4. 最后读 [LoRA/README.md](./LoRA/README.md)，理解 adapter 训练、合并、量化和动态 rank 分配如何服务于微调和多 LoRA serving。

## 与 SGLang 源码阅读的关系

- `Schedule_Optimization` 对应 Scheduler、continuous batching、prefill/decode 混排和 chunked prefill。
- `Attention_Kernel` 对应 ModelRunner、attention backend、KV cache 读写和 decode attention。
- `Parallel_Strategy` 对应 TP/PP/DP/EP rank 组织、通信模式和多进程执行。
- `LoRA` 对应 LoRA adapter 注册、热加载、batch 约束、LoRA memory pool 和 kernel 执行路径。

## 运行方式

这些 demo 主要依赖 Python 与 PyTorch。建议在仓库根目录运行，方便后续把输出和源码阅读笔记对应起来：

```bash
python learning/ai-infra-basic/Schedule_Optimization/prefill_decode_demo.py
python learning/ai-infra-basic/Schedule_Optimization/chunked_prefill_with_fakeLLM_tutorial.py
python learning/ai-infra-basic/Attention_Kernel/flash_attention_tutorial.py
python learning/ai-infra-basic/Attention_Kernel/flash_decoding_tutorial.py
python learning/ai-infra-basic/LoRA/lora_tutorial.py
```

并行策略目录里的 demo 适合逐个打开源码阅读；部分分布式示例需要多进程或多 GPU 环境，不建议一上来直接运行全部文件。

## 六周学习节奏

1. 推理基础：Transformer 推理流程、Prefill/Decode、KV Cache、Attention shape、batch inference。
2. 跑通 SGLang：安装、启动 server、OpenAI-compatible API、offline engine、sampling 参数。
3. 调度与 batching：请求队列、continuous batching、prefill batch、decode batch、延迟和吞吐权衡。
4. KV Cache 与 RadixAttention：cache block/page、prefix cache、cache hit/miss、cache eviction。
5. 高级优化：tensor parallel、pipeline parallel、expert parallel、quantization、speculative decoding、multi-LoRA batching、PD disaggregation。
6. 源码阅读和小型贡献：从 launch_server、request object、scheduler、model runner、KV cache manager 到 sampler 串起主链路。
