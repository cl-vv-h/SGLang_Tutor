**中文** | [English](./README_EN.md)

# Benchmark Profiling

这一章讲如何判断一个 serving 系统到底快不快，以及慢在哪里。Benchmark 给出外部指标，profiling 帮你定位内部瓶颈。两者缺一不可。

## 常见指标

| 指标 | 含义 | 主要受什么影响 |
|---|---|---|
| QPS | 每秒完成请求数 | 请求长度、并发、调度、模型大小 |
| TPS | 每秒输出 token 数 | decode batch、attention backend、并行策略 |
| TTFT | Time To First Token | 排队、tokenize、prefill、KV transfer |
| ITL | Inter-token Latency | decode step、KV 读取、batch 大小 |
| E2E latency | 请求总耗时 | prompt 长度、输出长度、排队时间 |
| GPU utilization | GPU 利用率 | batch 是否足够、CPU overhead、通信等待 |
| KV cache usage | KV 显存占用 | 并发、上下文长度、prefix cache 命中 |

## Workload 设计

一个 benchmark 不能只测单一 prompt。至少要区分：

```text
短 prompt + 短输出: chat 问答
短 prompt + 长输出: 写作 / 代码生成
长 prompt + 短输出: RAG / 文档问答
长 prompt + 长输出: agent / 长文生成
高并发短请求: latency-sensitive
低并发长请求: throughput / memory-sensitive
```

不同 workload 会把瓶颈推向完全不同的位置。

## 压测时容易踩的坑

1. 只看平均延迟，不看 p50/p90/p99。
2. prompt/output 长度分布不符合真实业务。
3. 客户端生成请求太慢，反而成为瓶颈。
4. 没有区分排队时间和模型执行时间。
5. warmup 不够，把首次编译、graph capture、权重加载算进稳态。
6. 忽略失败请求、超时请求和取消请求。

## Profiling 层次

从外到内可以分成：

```text
业务指标:
    QPS / TTFT / ITL / latency distribution

runtime 指标:
    waiting queue size / running batch size / prefill tokens / decode tokens

GPU 指标:
    kernel time / memory bandwidth / SM utilization / HBM usage

通信指标:
    all-reduce / all-to-all / KV transfer time / network bandwidth
```

不要一上来就看 kernel trace。先用外部指标判断是 TTFT 问题、ITL 问题还是吞吐问题，再往内定位。

## 常见瓶颈和症状

| 症状 | 可能原因 | 优先检查 |
|---|---|---|
| TTFT 很高 | prefill 排队、长 prompt、chunked prefill 未开启 | waiting queue、prefill tokens、admission |
| ITL 抖动 | decode batch 不稳定、CPU overhead、通信等待 | decode batch size、CUDA graph、NCCL |
| GPU 利用率低 | batch 太小、CPU launch overhead、sampling 慢 | batch 统计、graph replay、CPU profile |
| 显存 OOM | KV Cache 太多、prefix cache 未回收 | KV usage、block table、max running requests |
| TP 扩展差 | all-reduce 开销大、batch 太小 | NCCL trace、per-layer time |
| MoE 抖动 | expert imbalance、all-to-all 慢 | expert load、dispatch/combine time |

## 和 SGLang 的连接点

- Benchmark scripts 负责构造请求、记录 TTFT/ITL/TPS。
- Scheduler 日志能帮助区分 prefill 和 decode token 数。
- Runtime metrics 可以暴露 queue、batch、cache、throughput 等状态。
- CUDA/NVTX profiling 能把 ModelRunner、attention、MLP、sampling 和通信拆开看。
- 调参要有闭环：改参数、测 workload、看指标、再决定是否保留。

## 阅读任务

1. 设计一组能覆盖短问答、RAG、长输出的 benchmark workload。
2. 解释为什么平均延迟可能掩盖线上问题。
3. 遇到 TTFT 高但 ITL 正常时，你会先看哪些指标。
4. 遇到 GPU 利用率低但请求很多时，你会怀疑哪些环节。
