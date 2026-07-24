[中文](./06-deepep-low-latency-and-layered-a2-path.md) | [English](./06-deepep-low-latency-and-layered-a2-path_EN.md)

# sgl-kernel-npu 06: DeepEP Low-Latency & A2 Layered Path

## Why Low-Latency Path

Small-batch inference (typical in online serving) sends very few tokens per expert per round. Standard all-to-all overhead becomes proportionally large.

```text
Standard path: all-to-all dispatch → expert GEMM → all-to-all combine
  Good for: large batches, many tokens per expert
  Bad for: B=1-8, few tokens spread across many experts

Low-latency path: low_latency_dispatch → expert GEMM → low_latency_combine
  Good for: small batches, minimal communication overhead
  Trade-off: less bandwidth-efficient for large transfers
```

## A2 Layered Path

Combines intra-node and inter-node communication:

```text
A2 Layered:
  Intra-node: HCCS (high bandwidth, low latency)
  Inter-node: RoCE/RDMA (network, higher latency)

Layering strategy:
  1. First layer: dispatch within node (HCCS)
  2. Second layer: cross-node dispatch (RDMA)
  → Minimizes cross-node traffic, maximizes intra-node bandwidth
```

## When to Use Each Path

| Scenario | Path |
|---|---|
| Large batch, multi-node MoE | Standard DeepEP |
| Small batch, single-node | Low-latency |
| Multi-node, heterogeneous interconnect | A2 Layered |
