# Mamba / State Space Model 教学入口

这个目录专门解释 Mamba 以及 SGLang 中围绕 Mamba 出现的运行时概念：`mamba state`、`MambaPool`、`mamba_scheduler_strategy`、`mamba_track_indices`、Mamba radix cache、hybrid attention backend 等。

## 本专题文件

| 文件 | 内容 |
|---|---|
| [01-mamba-and-sglang-state.md](./01-mamba-and-sglang-state.md) | 从 State Space Model 和 Mamba 基本原理讲到 SGLang 中 Mamba state 的内存、调度和数据流 |
| [02-mamba-principles.md](./02-mamba-principles.md) | 专门讲 Mamba 的模型原理、Mamba2 block 的 forward 实现、prefill/decode kernel 路径 |
| [03-mamba-radix-cache.md](./03-mamba-radix-cache.md) | 专门讲 Mamba 与 Radix Cache 的关系、代码依赖、状态插入/命中/释放流程 |

## 推荐先理解的背景

在读这一讲之前，建议先有三个基础印象：

1. Transformer attention 会为每个历史 token 保存 KV Cache。
2. Decode 阶段每轮只生成一个或少量 token，但必须携带历史上下文状态。
3. SGLang 的 Scheduler 会把请求、cache、forward mode 和 graph replay 组织到同一条执行链路上。

如果这些概念还不稳，可以先读：

- [Inference_Basics](../Inference_Basics/README.md)
- [KV_Cache_Memory](../KV_Cache_Memory/README.md)
- [Execution_Graph](../Execution_Graph/README.md)

## 和 SGLang 的连接点

- `python/sglang/srt/configs/mamba_utils.py`：Mamba state shape、dtype、每请求显存估算。
- `python/sglang/srt/mem_cache/memory_pool.py`：`MambaPool`、`HybridReqToTokenPool` 中 Mamba state 分配和释放。
- `python/sglang/srt/layers/attention/mamba/`：Mamba2 forward、metadata、selective scan/conv kernel。
- `python/sglang/srt/server_args.py`：`mamba_scheduler_strategy`、`mamba_track_interval`、`max_mamba_cache_size` 等参数校验。
- `python/sglang/srt/mem_cache/mamba_radix_cache.py`：Mamba 场景下 prefix/radix cache 的状态保存。
- `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`：Mamba track 信息如何进入 graph/piecewise graph buffer。

## 阅读目标

读完本专题后，你应该能回答：

1. Mamba 为什么不需要像 attention 那样保存完整 KV Cache？
2. Mamba 的 `conv state` 和 `SSM/temporal state` 分别是什么？
3. SGLang 为什么要为每个请求分配 `mamba_pool_idx`？
4. `mamba_scheduler_strategy=no_buffer/extra_buffer` 大致在解决什么问题？
5. 为什么 Mamba 与 radix cache、chunked prefill、speculative decoding、graph replay 都会发生耦合？
