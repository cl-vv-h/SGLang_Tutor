# Schedule Optimization 教学入口

这个目录用小型 demo 解释 LLM serving 调度中最基础、也最容易混淆的部分：Prefill/Decode 的差异、KV Cache 的作用，以及 Chunked Prefill 为什么能改善长 prompt 对系统延迟的影响。

## 文件说明

| 文件 | 主题 | 核心问题 |
|---|---|---|
| [prefill_decode_demo.py](./prefill_decode_demo.py) | 带 KV Cache 的 tiny causal LM | prefill 一次处理 prompt，decode 每次处理一个新 token，KV Cache 如何避免重复计算 |
| [chunked_prefill_with_fakeLLM_tutorial.py](./chunked_prefill_with_fakeLLM_tutorial.py) | Chunked Prefill 调度模拟 | 长 prompt 如何拆成多个 chunk，decode 请求如何穿插执行 |

## 推荐阅读顺序

1. 先读 `prefill_decode_demo.py` 中的 `LayerKVCache`、`prefill`、`decode` 和 `greedy_generate`。
2. 对比 `greedy_generate_without_cache`，确认没有 KV Cache 时每步 decode 都会重复计算整个上下文。
3. 再读 `chunked_prefill_with_fakeLLM_tutorial.py` 中的 `Request`、`ScheduledTask` 和 `ChunkedPrefillScheduler`。
4. 最后观察 demo 输出：长 prefill 被切开后，短 decode 请求可以更早获得调度机会。

## 和 SGLang 的连接点

- Scheduler 的核心任务是把等待队列中的请求组织成 prefill batch 和 decode batch。
- Continuous batching 的价值在于每一步都能动态接纳新请求，而不是等整个静态 batch 全部结束。
- Chunked Prefill 能减少长 prompt 独占计算资源的时间，更适合线上多租户和长短请求混合场景。
- KV Cache 是 decode 高效执行的前提，也是 prefix cache、RadixAttention、HiCache 等机制的基础。

## 运行示例

```bash
python learning/ai-infra-basic/Schedule_Optimization/prefill_decode_demo.py
python learning/ai-infra-basic/Schedule_Optimization/chunked_prefill_with_fakeLLM_tutorial.py
```
