[中文](./05-deepep-hccl-and-moe-kernel-path.md) | [English](./05-deepep-hccl-and-moe-kernel-path_EN.md)

# sgl-kernel-npu 05: DeepEP, HCCL & MoE Token Path

## DeepEP Architecture

```text
DeepEP decomposes MoE routing into:
  layout → dispatch → local expert compute → combine
```

## Token Flow

```text
1. Router: Top-K selection per token
2. Layout: Count tokens per expert per rank → plan all-to-all
3. Dispatch: All-to-all sends token rows to expert-owning ranks
4. Local Expert Compute: Each rank runs its experts' GEMM
5. Combine: All-to-all returns expert outputs to token-owning ranks
```

## deep_ep::Buffer

```cpp
// Core abstraction for MoE communication
deep_ep::Buffer buffer;
buffer.dispatch(tokens, expert_ids);      // Send tokens
buffer.combine(expert_outputs, tokens);   // Gather results
```

## Key Functions

| Function | Purpose |
|---|---|
| `fused_deep_moe` | End-to-end fused MoE forward |
| `dispatch_ffn_combine` | Dispatch + FFN + combine pipeline |
| `dispatch_count_layout` | Pre-compute token-to-expert mapping |
