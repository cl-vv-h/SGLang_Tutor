[中文](./README.md) | [English](./README_EN.md)

# Mamba / State Space Models

Mamba and State Space Models (SSM): principles, Mamba state management, scheduler strategy, and radix cache state handling in SGLang.

## Files

| File | Content |
|---|---|
| [01-mamba-and-sglang-state.md](./01-mamba-and-sglang-state.md) | How Mamba/SSM integrates with SGLang: MambaStatePool, MambaPool, scheduler strategy |
| [02-mamba-principles.md](./02-mamba-principles.md) | SSM fundamentals: state space equations, discretization, selective scan |
| [03-mamba-radix-cache.md](./03-mamba-radix-cache.md) | Mamba-specific radix cache: state management vs KV Cache, prefix sharing for SSM |

## Key Difference from Transformer

Unlike Transformers that store explicit KV Cache tensors, Mamba models maintain a recurrent "state" — a compressed representation of history. This changes how prefix caching, batching, and state transfer work in serving systems.
