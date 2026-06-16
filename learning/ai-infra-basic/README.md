# AI Infra 基础专题

这个目录用于补齐阅读 SGLang 源码前后的基础知识。它不直接复刻生产级实现，而是用讲义和小型 Python demo 拆开 LLM serving 中最常见的机制：推理流程、调度、KV Cache、attention kernel、执行图、并行、KV 传输、投机解码、量化、LoRA 和 benchmark/profiling。

## 目录结构

| 顺序 | 目录 | 内容 | 建议先看 |
|---|---|---|---|
| 1 | [Inference_Basics](./Inference_Basics/) | Transformer 推理、Prefill/Decode、吞吐与延迟基础 | [README.md](./Inference_Basics/README.md) |
| 2 | [Schedule_Optimization](./Schedule_Optimization/) | Continuous batching、Chunked Prefill、调度权衡 | [README.md](./Schedule_Optimization/README.md) |
| 3 | [KV_Cache_Memory](./KV_Cache_Memory/) | KV Cache 布局、分页、prefix cache、显存估算 | [README.md](./KV_Cache_Memory/README.md) |
| 4 | [Attention_Kernel](./Attention_Kernel/) | FlashAttention 与 FlashDecoding 的教学版实现 | [README.md](./Attention_Kernel/README.md) |
| 5 | [Execution_Graph](./Execution_Graph/) | 从计算图概念到 CUDA/NPU Graph、torch.compile、静态形状复用 | [01-what-is-graph.md](./Execution_Graph/01-what-is-graph.md) |
| 6 | [Parallel_Strategy](./Parallel_Strategy/) | DP、TP、PP、SP/CP、EP 推理并行策略 | [README.md](./Parallel_Strategy/README.md) |
| 7 | [KV_Transfer](./KV_Transfer/) | PD 分离、KV sender/receiver、远程 KV cache | [README.md](./KV_Transfer/README.md) |
| 8 | [Speculative_Decoding](./Speculative_Decoding/) | Draft/Target、verify、accept token、EAGLE/NGRAM | [README.md](./Speculative_Decoding/README.md) |
| 9 | [Quantization](./Quantization/) | Weight-only、W8A8/FP8、KV quant、校准与误差 | [README.md](./Quantization/README.md) |
| 10 | [LoRA](./LoRA/) | LoRA、QLoRA、DoRA、AdaLoRA 和多 LoRA serving | [README.md](./LoRA/README.md) |
| 11 | [Benchmark_Profiling](./Benchmark_Profiling/) | TTFT/ITL/TPS、压测、profiling、瓶颈定位 | [README.md](./Benchmark_Profiling/README.md) |

## 建议学习路线

1. 先读 [Inference_Basics](./Inference_Basics/) 和 [Schedule_Optimization](./Schedule_Optimization/)，建立 prefill/decode、batching、延迟和吞吐的基本模型。
2. 再读 [KV_Cache_Memory](./KV_Cache_Memory/) 和 [Attention_Kernel](./Attention_Kernel/)，理解为什么 LLM serving 常常被显存、KV 读取和 attention backend 限制。
3. 然后读 [Execution_Graph](./Execution_Graph/) 和 [Parallel_Strategy](./Parallel_Strategy/)，理解生产推理如何减少 CPU overhead、稳定形状并扩到多卡。
4. 接着读 [KV_Transfer](./KV_Transfer/) 和 [Speculative_Decoding](./Speculative_Decoding/)，理解高阶 serving 优化如何围绕“更快拿到 token”和“更好使用不同资源”展开。
5. 最后读 [Quantization](./Quantization/)、[LoRA](./LoRA/) 和 [Benchmark_Profiling](./Benchmark_Profiling/)，把模型压缩、adapter serving 和性能验证闭环串起来。

## 与 SGLang 源码阅读的关系

- `Inference_Basics` 对应请求生命周期、forward mode、sampling 和 token generation loop。
- `Schedule_Optimization` 对应 Scheduler、waiting/running queue、continuous batching、chunked prefill。
- `KV_Cache_Memory` 对应 KV cache manager、memory pool、RadixAttention、prefix cache、HiCache。
- `Attention_Kernel` 对应 attention backend、prefill attention、decode attention、KV block 读写。
- `Execution_Graph` 对应 CUDA graph、静态 batch、graph capture/replay 和 shape padding。
- `Parallel_Strategy` 对应 TP/PP/DP/EP rank 组织、通信模式和多进程执行。
- `KV_Transfer` 对应 PD disaggregation、bootstrap、prealloc、KV sender/receiver 和 transfer backend。
- `Speculative_Decoding` 对应 draft worker、target verify、`spec_info`、accept token 和 grammar/sampling 后处理。
- `Quantization` 对应 weight loader、quant method、kernel 选择、FP8/W8A8/GPTQ/AWQ 等执行路径。
- `LoRA` 对应 adapter 注册、热加载、batch 约束、LoRA memory pool 和 LoRA kernel。
- `Benchmark_Profiling` 对应 benchmark scripts、metrics、trace、CUDA/NVTX profiling 和线上调参。

## 运行方式

已有可运行 demo 主要依赖 Python 与 PyTorch。建议在仓库根目录运行，方便后续把输出和源码阅读笔记对应起来：

```bash
python learning/ai-infra-basic/Schedule_Optimization/prefill_decode_demo.py
python learning/ai-infra-basic/Schedule_Optimization/chunked_prefill_with_fakeLLM_tutorial.py
python learning/ai-infra-basic/Attention_Kernel/flash_attention_tutorial.py
python learning/ai-infra-basic/Attention_Kernel/flash_decoding_tutorial.py
python learning/ai-infra-basic/LoRA/lora_tutorial.py
```

并行策略目录里的 demo 适合逐个打开源码阅读；部分分布式示例需要多进程或多 GPU 环境，不建议一上来直接运行全部文件。
