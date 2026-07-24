[中文](./15-distributed-hccl-and-communication.md) | [English](./15-distributed-hccl-and-communication_EN.md)

# 15. Distributed HCCL & Communication on NPU

## 1. HCCL Collective Operations in SGLang

| Operation | SGLang Usage | HCCL Call |
|---|---|---|
| All-Reduce | TP attention output, FFN output | `hcclAllReduce` |
| All-Gather | DP attention, embedding gather | `hcclAllGather` |
| Reduce-Scatter | TP gradient (training) | `hcclReduceScatter` |
| All-to-All | MoE EP dispatch/combine | `hcclAllToAll` |
| Broadcast | Weight sync | `hcclBroadcast` |

## 2. Communication Group Initialization

```python
# parallel_state.py: TP group initialization on NPU
if is_npu():
    backend = "hccl"
    torch.distributed.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )
    
    # TP group within node (HCCS)
    tp_group = torch.distributed.new_group(
        ranks=tp_ranks,
        backend="hccl"
    )
```

## 3. Intra-Node vs Inter-Node

| Scope | Interconnect | Bandwidth | Latency |
|---|---|---|---|
| Intra-node (same server) | HCCS | ~200-400 GB/s per NPU | ~1-5 μs |
| Inter-node (cross-server) | RoCE / RDMA | ~100-200 Gb/s | ~5-50 μs |

## 4. TP Communication Pattern

```text
Each TP layer:
  Attention:
    QKV projection (local, column-parallel)
    All-reduce: attention output
    
  FFN/MoE:
    Gate+Up projection (local, column-parallel)
    All-reduce: FFN output (for dense FFN)
    All-to-all: MoE dispatch/combine (for EP)
```

## 5. Communication Overlap

```python
# HCCL operations can be stream-ordered for overlap
with torch.npu.stream(compute_stream):
    local_result = compute(...)
    
with torch.npu.stream(comm_stream):
    hccl_all_reduce(local_result, ...)
    
# Wait for communication before using result
torch.npu.current_stream().wait_event(comm_event)
```

## 6. Debugging HCCL Issues

| Symptom | Likely Cause | Check |
|---|---|---|
| HCCL init hangs | Rank mismatch, network | Verify rank assignment, NCCL/HCCL env vars |
| All-reduce slow | Bad interconnect | Check HCCS topology with `npu-smi info -t topo` |
| All-to-all imbalance | Hot experts | Profile expert load distribution |
| Timeout | Deadlock in comm pattern | Verify matching send/recv counts |
