# 06. From SGLang Call Paths to Communication Performance Diagnosis

**English** | [简体中文](../../../zh/ai-infra-basic/Communication/06-sglang-paths-and-performance.md)

## 1. Follow Three Data Paths

### 1.1 TP / Layer-Boundary Collectives

```math
\frac{\mathrm{model}}{\mathrm{LayerCommunicator}}
```

```text
-> communication_op.py
-> GroupCoordinator + selected TP/attention/MoE group
-> NpuCommunicator / PyNccl / custom all-reduce / other path
-> device backend / kernel
```

Entries: [communicator.py](../../../../python/sglang/srt/layers/communicator.py), [communication_op.py](../../../../python/sglang/srt/distributed/communication_op.py), and [parallel_state.py](../../../../python/sglang/srt/distributed/parallel_state.py). The NPU wrapper is [npu_communicator.py](../../../../python/sglang/srt/distributed/device_communicators/npu_communicator.py).

Record the model layer, group members, input shape/dtype, sharding, and whether results are written in-place. Then inspect branches in `GroupCoordinator.all_reduce()`, not only launch arguments. Custom kernels, symmetric memory, and graph modes can change execution.

### 1.2 MoE EP

```text
router/top-k -> token dispatcher -> dispatch handle/layout
            -> local expert compute -> combine -> token order restored
```

Entries: [token_dispatcher/deepep.py](../../../../python/sglang/srt/layers/moe/token_dispatcher/deepep.py) and [base.py](../../../../python/sglang/srt/layers/moe/token_dispatcher/base.py). Identify whether `Buffer` comes from `deep_ep` or `zbal`, then inspect mode, counts, routed traffic, and completion hooks.

### 1.3 PD and Remote KV

```text
prefill/decode queues -> destination allocation + request metadata
                     -> transfer worker / engine
                     -> completion protocol -> decode ready
```

Entries: [prefill.py](../../../../python/sglang/srt/disaggregation/prefill.py), [decode.py](../../../../python/sglang/srt/disaggregation/decode.py), [mooncake/conn.py](../../../../python/sglang/srt/disaggregation/mooncake/conn.py), and [ascend/transfer_engine.py](../../../../python/sglang/srt/disaggregation/ascend/transfer_engine.py). For HiCache remote caching, see the separate [Mooncake Store integration](../../../../python/sglang/srt/mem_cache/storage/mooncake_store/README.md).

## 2. Choose Validation According to the Problem

| Problem | First Baseline | Next Comparison | Conditions to Hold Fixed |
|---|---|---|---|
| CPU coordination / learning collectives | two-process Gloo correctness | more processes and tensor sizes | participants, order, dtype |
| NVIDIA / AMD TP | NCCL / RCCL microbenchmark | framework custom paths, topology, message size | rank mapping, process count, metric definitions |
| Ascend TP | HCCL correctness/performance | compatible ZBAL operators/memory modes | NPU model, CANN, dtype, buffer size |
| MoE EP | dispatch→expert→combine correctness | DeepEP modes, ZBAL adapter, fusion | top-k, routing distribution, valid tokens, capacity |
| PD/KV | end-to-end block contents and request lifetime | TE/NIXL/Ascend paths, chunking, concurrency | cache format, P/D TP, prefix-hit ratio |

“Supported” means correct execution; “faster” requires measurement on the target workload. A project's peak microbenchmark result is not the model service's benefit.

## 3. Define Timing Boundaries First

A useful accounting model is:

```math
t_{\mathrm{observed}}=t_{\mathrm{queue}}+t_{\mathrm{pack}}+\mathrm{t\_wait\_for\_peers}+t_{\mathrm{transfer}}+t_{\mathrm{unpack}}+\mathrm{t\_consumer\_dependency}
```

These intervals can overlap. Treat this as a diagnostic ledger, not a license to double-count overlapping time.

- **Microbenchmark**: validate primitives and links with representative small/medium/large messages; separate cold start, warmed steady state, and sustained tail latency.
- **Model trace**: inspect arrival times across ranks, communication streams, dependency events, and HBM/compute contention.
- **Service benchmark**: fix input/output-length distributions and concurrency; measure TTFT, ITL, throughput, p95/p99, and errors.

Device work is asynchronous. Timing must cover completion using suitable device events or explicit synchronization boundaries. Cross-node absolute timestamps require clock alignment; otherwise compare local intervals and protocol events first.

## 4. Specify Bandwidth Accounting

```math
\text{algorithm bandwidth}= \mathrm{logical} \mathrm{payload} \mathrm{bytes} / \mathrm{elapsed} \mathrm{time}
```

```math
\mathrm{ring} \mathrm{AllReduce} \mathrm{bus} \mathrm{bandwidth} \approx  \mathrm{algorithm} \mathrm{bandwidth} \cdot  2\cdot \left(P-1\right)/P
```

The latter is the [nccl-tests AllReduce convention](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md), not a capture of actual NIC bytes; do not apply it unchanged to MoE or KV transfer. Distinguish per-rank/aggregate, unidirectional/bidirectional, GB/s/GiB/s, and Gb/s. The theoretical conversion of 100 Gb/s is 12.5 GB/s before protocol overhead.

For MoE, record valid tokens, top-k assignments, cross-rank/cross-node traffic, padding, and scale bytes. For KV, record actual bytes after cache hits. Otherwise, reducing padding can be misinterpreted as slower networking.

## 5. Symptoms and Evidence

| Symptom | Check First | Reason |
|---|---|---|
| collective timeout / hang | first error on each rank, group/order, shapes/splits, device binding | one missing participant makes others wait |
| small messages slow, large bandwidth normal | startup, host scheduling, synchronization, graph path | latency terms dominate small messages |
| large messages also slow | NIC/NUMA, actual transport, staging, congestion | the intended data path may not be active |
| one rank slow | preceding compute, routing skew, shared resources, network hotspots | collective duration can include waiting for a peer |
| overlap worsens ITL | core/HBM/NIC contention and extra synchronization | overlapping operations can both slow down |
| PD tail latency high | allocation queues, fragmented blocks, registration, retries, completion counts | payload bandwidth covers only part of the handoff |
| OOM/init failure after enabling ZBAL | packages/hardware matrix, allocator/mixed mode, budgets, setup order | communication is coupled to memory allocation |
| numerical error after communication quantization | scale layout, reduction precision, tolerances | fewer bytes do not guarantee numerical equivalence |

Keep a reproducible baseline and change one interpretable variable at a time. Increasing timeouts can help diagnose slow initialization but cannot fix operation-order bugs; adding barriers can hide the original overlap problem.

## 6. A Reusable Experiment Record

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

For Ascend collection, continue with [Benchmark Profiling](../Benchmark_Profiling/README.md) and [Ascend benchmarking/debugging](../../sglang-ascend-npu/10-benchmark-debugging.md). Record the actual backend with the trace: identically named Python entries need not invoke the same kernel.

## 7. Final Checks

1. AllReduce drops from 200 us to 120 us while ITL stays unchanged. What could explain it?
2. Why can EP tail latency worsen while average tokens per rank remain constant?
3. Can Mooncake TE, HCCL, and ZBAL be ranked in a single “fastest” table? Which operation must be fixed first?

Return to the [communication topic](./README.md).
