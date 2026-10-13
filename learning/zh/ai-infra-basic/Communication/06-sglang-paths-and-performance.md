# 06. 从 SGLang 调用路径到通信性能排查

**简体中文** | [English](../../../en/ai-infra-basic/Communication/06-sglang-paths-and-performance.md)

## 1. 沿三条数据路径读源码

### 1.1 TP / 层边界 collective

```text
model / LayerCommunicator
  -> communication_op.py
  -> GroupCoordinator + selected TP/attention/MoE group
  -> NpuCommunicator / PyNccl / custom all-reduce / other path
  -> device backend / kernel
```

入口：[communicator.py](../../../../python/sglang/srt/layers/communicator.py)、[communication_op.py](../../../../python/sglang/srt/distributed/communication_op.py)、[parallel_state.py](../../../../python/sglang/srt/distributed/parallel_state.py)。NPU wrapper 见 [npu_communicator.py](../../../../python/sglang/srt/distributed/device_communicators/npu_communicator.py)。

记录调用所在层、group 成员、输入 shape/dtype、tensor 是否分片、结果是否原地写回。然后看 `GroupCoordinator.all_reduce()` 的分支，而不只看启动参数。自定义 kernel、对称内存和 graph 模式可能改变执行路径。

### 1.2 MoE EP

```text
router/top-k -> token dispatcher -> dispatch handle/layout
            -> local expert compute -> combine -> token order restored
```

入口：[token_dispatcher/deepep.py](../../../../python/sglang/srt/layers/moe/token_dispatcher/deepep.py) 和 [base.py](../../../../python/sglang/srt/layers/moe/token_dispatcher/base.py)。先确定真实 `Buffer` 来自 `deep_ep` 还是 `zbal`，再看模式、计数、路由量与 completion hook。

### 1.3 PD 与远端 KV

```text
prefill/decode queues -> destination allocation + request metadata
                     -> transfer worker / engine
                     -> completion protocol -> decode ready
```

入口：[prefill.py](../../../../python/sglang/srt/disaggregation/prefill.py)、[decode.py](../../../../python/sglang/srt/disaggregation/decode.py)、[mooncake/conn.py](../../../../python/sglang/srt/disaggregation/mooncake/conn.py)、[ascend/transfer_engine.py](../../../../python/sglang/srt/disaggregation/ascend/transfer_engine.py)。HiCache 远端缓存则另见 [Mooncake Store 集成](../../../../python/sglang/srt/mem_cache/storage/mooncake_store/README.md)。

## 2. 按问题选择验证路径

| 当前问题 | 第一条基线 | 然后比较 | 保持相同的条件 |
|---|---|---|---|
| CPU 协调 / 初学 collective | 两进程 Gloo 正确性 | 多进程、不同 tensor 大小 | 成员、调用顺序、dtype |
| NVIDIA / AMD TP | NCCL / RCCL microbenchmark | 框架 custom 路径、拓扑与消息大小 | rank 映射、进程数、算法口径 |
| Ascend TP | HCCL 正确性与性能基线 | 已兼容的 ZBAL 算子/内存模式 | NPU 型号、CANN、dtype、buffer 大小 |
| MoE EP | dispatch→expert→combine 正确性 | DeepEP 模式、ZBAL 适配、融合路径 | top-k、路由分布、有效 token 与容量 |
| PD/KV | 端到端块内容与请求生命周期正确 | TE/NIXL/Ascend 路径、分块与并发 | 模型 cache 格式、P/D TP、prefix 命中率 |

“支持”是能够正确执行；“更快”必须在目标工作负载上测得。某项目 microbenchmark 的峰值不等于模型服务收益。

## 3. 先定义测量边界

一个实用分解是：

```text
t_observed = t_queue + t_pack + t_wait_for_peers
           + t_transfer + t_unpack + t_consumer_dependency
```

各项可能重叠，这只是定位账本，不能把重叠区间重复相加当成总耗时。

- **microbenchmark**：验证原语和基础链路，使用代表性的小/中/大消息；区分冷启动、预热后稳态与长期尾延迟。
- **模型 trace**：看各 rank 到达通信的时间、通信所在 stream、依赖事件、HBM/计算资源竞争。
- **服务 benchmark**：固定输入/输出长度分布与并发，看 TTFT、ITL、吞吐、p95/p99 和错误率。

设备工作是异步的；计时应覆盖实际完成，使用匹配 backend 的设备事件或明确同步边界。跨节点 trace 的绝对时间比较需要时钟对齐；否则先比较本 rank 区间和协议事件。

## 4. 带宽数字要写清口径

```text
algorithm bandwidth = logical payload bytes / elapsed time
ring AllReduce bus bandwidth ≈ algorithm bandwidth * 2*(P-1)/P
```

后一项是 [nccl-tests 的 AllReduce 统计口径](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)，不是“实际捕获到了这么多 NIC 字节”，也不应直接套到 MoE 或 KV transfer。分清 per-rank / aggregate、单向 / 双向、GB/s / GiB/s、Gb/s，100 Gb/s 的理论换算是 12.5 GB/s，尚未扣协议损耗。

对 MoE 至少同时记录有效 token、top-k assignments、跨 rank/跨节点发送量、padding 与 scale bytes；对 KV 记录命中前后实际传输 bytes。否则减少 padding 的优化可能被误判为“网络变慢”。

## 5. 从症状到证据

| 症状 | 优先检查 | 为什么 |
|---|---|---|
| collective 超时 / hang | 各 rank 首个异常、group/order、shape/splits、设备绑定 | 一张卡没进入操作，其他卡会一起等 |
| 小包慢、大包带宽正常 | 启动、host 调度、同步、graph 路径 | 小包更容易受 latency 项支配 |
| 大包也慢 | NIC/NUMA、实际 transport、staging、链路拥塞 | 可能根本没有走预期数据通路 |
| 只有一个 rank 慢 | 前置计算、路由倾斜、资源共享、网络热点 | collective 时长可能包含等慢 peer 的时间 |
| 开启 overlap 后 ITL 变差 | 核心/HBM/NIC 争抢、额外同步 | 重叠可使两项工作都变慢 |
| PD 尾延迟高 | 预分配队列、碎片块、注册、重试、完成计数 | payload 带宽只覆盖交接的一部分 |
| 切 ZBAL 后 OOM/初始化失败 | 包/硬件矩阵、allocator/mix mode、预算、初始化顺序 | 通信与内存分配耦合 |
| 量化通信后结果偏差 | scale 布局、规约精度、误差容限 | 字节减少不自动保证数值等价 |

诊断时保留一个可复现基线，每次只改一个能解释的变量。增加 timeout 可以用于判断初始化是否很慢，但不能修复调用顺序错误；加 barrier 也可能隐藏原来的重叠问题。

## 6. 一份可复用的实验记录

```text
source commits / package versions:
device model + driver + CUDA or CANN + torch adapter:
nodes / ranks / TP-EP-PP groups / rank-to-device-NIC mapping:
operation + actual backend + graph/stream mode:
input shape / dtype / useful bytes / padding / routing skew:
warmup / iterations / timing boundary / concurrency:
correctness tolerance + failures:
latency median / p95 / p99 + bandwidth definition:
TTFT / ITL / throughput before and after:
```

Ascend 性能采集方法另见 [Benchmark Profiling](../Benchmark_Profiling/README.md) 与 [Ascend benchmark/debugging](../../sglang-ascend-npu/10-benchmark-debugging.md)。采集 trace 的同时记录实际后端，避免把同名 Python 入口误当成同一个 kernel。

## 7. 最后一组检查题

1. `all_reduce` 从 200 us 变成 120 us，但 ITL 不变，可能有哪些原因？
2. EP 的 avg token/rank 不变，为什么尾延迟会变差？
3. Mooncake TE、HCCL、ZBAL 能否用一张“谁最快”的表比较？应先固定哪种操作？

返回 [通信专题入口](./README.md)。
