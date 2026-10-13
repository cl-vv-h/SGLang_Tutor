# 01. Communication Primitives, Tensor Layouts, and Cost Models

**English** | [简体中文](../../../zh/ai-infra-basic/Communication/01-communication-primitives-and-costs.md)

## 1. Start with Tensors on Two Ranks

A rank identifies a process within a communication group. One process per accelerator is common, but a rank is not a physical device ID. A process may belong to separate TP, EP, and PP groups; interpret each collective within its designated group.

Each rank below holds a vector of length 4. The reduction operator is sum:

```text
rank 0: [1, 2, 3, 4]
rank 1: [10, 20, 30, 40]

AllReduce:
rank 0: [11, 22, 33, 44]
rank 1: [11, 22, 33, 44]

ReduceScatter (equal contiguous shards):
rank 0: [11, 22]
rank 1: [33, 44]

AllGather of those ReduceScatter outputs:
rank 0: [11, 22, 33, 44]
rank 1: [11, 22, 33, 44]
```

AllGather concatenates in rank order without summation. Gathering the original inputs produces length 8. ReduceScatter followed by AllGather mathematically recovers AllReduce only when the reduction operates on the same logical tensor and the shard ordering matches. Different floating-point reduction trees may introduce rounding differences. [NCCL collective semantics](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)

| Primitive | Input → Output | Frequent Use |
|---|---|---|
| Broadcast | root data → all participating ranks | initialization and small tensor replication |
| Reduce | elementwise reduction across ranks → root | metric aggregation |
| AllReduce | elementwise reduction across ranks → every rank | TP partial sums and training gradient synchronization |
| AllGather | rank-local shards → concatenated tensor on every rank | SP/TP layout changes and training parameter collection |
| ReduceScatter | elementwise reduction → one shard per rank | sharded results and training gradients |
| AllToAll / AllToAllV | destination-partitioned inputs → source-partitioned outputs | EP token exchange; V denotes variable lengths |
| Send/Recv | data between a specific source and destination | PP activations and point-to-point protocols |
| Barrier | wait for all participants to arrive | initialization/debugging; not a replacement for data dependencies |

## 2. AllToAllV: Count Records in Each Direction

Let `counts[src][dst]` be the number of records sent from source to destination:

```text
counts = [[1, 2],
          [3, 0]]

rank 0: input_splits=[1,2], output_splits=[1,3]
rank 1: input_splits=[3,0], output_splits=[2,0]
```

Rank 0 has 3 input records and 4 output records; rank 1 has 3 inputs and 2 outputs. Every send direction must match the receiver's length, dtype, and record width for that source. Receive splits are columns of the counts matrix, not copies of the local send splits. A rank with zero tokens must still follow the protocol rather than independently skipping the collective.

MoE records may contain hidden vectors, expert IDs, original token indices, and quantization scales. Token count alone does not determine transferred bytes.

## 3. Why Parallel Strategies Create Communication

| Scenario | Dependency Created by Sharding | Typical Communication |
|---|---|---|
| row-parallel linear | `Y = sum_r X_r W_r` | AllReduce partial results, or ReduceScatter when downstream accepts shards |
| column-parallel linear | `Y_r = X W_r` | AllGather if downstream needs full Y; none may be needed if sharding continues |
| PP inference | next stage needs previous stage activations | Send/Recv |
| EP MoE | experts reside on different ranks | dispatch → expert compute → combine |
| CP attention | local queries need other positions' K/V or intermediate results | ring P2P, gather, or other implementation-specific algorithms |
| DP training | replica gradients must agree | AllReduce / ReduceScatter |
| independent DP inference | separate replicas serve different requests | usually no per-layer gradient synchronization; shared-MoE combinations differ |
| PD disaggregation | decode needs KV produced by prefill | request-level block transfer |

Using multiple devices therefore does not imply an AllReduce at every layer.

## 4. Latency: Small Messages Expose Startup, Large Ones Expose Bandwidth

Begin with an interpretable approximation:

```text
t_transfer ≈ n_steps * alpha + bytes_on_critical_link / B_eff
```

`alpha` is per-step startup/protocol latency; `B_eff` is effective bandwidth. Do not simply add advertised NIC bandwidths. With uniform chunks, balanced communication, and additional reduction costs ignored, ring AllReduce can be estimated as:

```text
P = participating ranks
M = full input bytes per rank
bytes_sent_per_rank = 2 * (P-1) / P * M
t_ring ≈ 2 * (P-1) * alpha + bytes_sent_per_rank / B_eff
```

This counts **bytes sent per rank**, not total network traffic or send-plus-receive traffic. Ring is one algorithm: trees, hierarchical algorithms, and in-network reduction change the model. This equation does not describe every backend implementation.

Example: `P=8`, `M=16 MiB`, `alpha=3 us`, and decimal `B_eff=50 GB/s` give 28 MiB sent per rank and approximately **629 us**. For `M=4 KiB`, the estimate is **42.14 us**, dominated by startup. These are teaching estimates from chosen parameters, not hardware measurements.

## 5. Limits to Overlap

Only independent computation and communication can overlap:

```text
serial:  t_step = t_compute + t_comm
ideal:   t_step >= max(t_compute, t_comm)
```

A GEMM that consumes an AllReduce result has a true dependency on it; two streams do not remove that dependency. Overlap can also contend for SMs/AICores, HBM, or NIC resources, slowing either operation. Judge end-to-end TTFT, ITL, and throughput rather than just overlapping colored bars in a trace.

## 6. Practice and Checks

Run the [semantics demo](./communication_semantics_demo.py). Compare AllGather with AllReduce and verify the ReduceScatter + AllGather equivalence.

1. Why must TP AllReduce elements correspond to the same tokens and output positions?
2. Why does rank 0 receive 4 records in the counts example?
3. Why can doubling `P` make small-message ring AllReduce slower?

Next: [links, RDMA, and memory](./02-links-rdma-and-memory.md). For bandwidth accounting, see [nccl-tests PERFORMANCE](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md).
