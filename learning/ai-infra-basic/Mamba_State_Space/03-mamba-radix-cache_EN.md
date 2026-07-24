[中文](./03-mamba-radix-cache.md) | [English](./03-mamba-radix-cache_EN.md)

# Mamba Radix Cache: State Management vs KV Cache

## 1. Why Mamba Needs a Different Cache Strategy

Transformer prefix caching works by sharing KV Cache pages: when two requests share the same prompt prefix, they point to the same physical KV pages in the radix tree. This works because KV Cache for a prefix is purely a function of the prefix tokens — it doesn't depend on what comes after.

Mamba state is fundamentally different: it's an accumulated recurrent state. Two requests with the same prefix produce the same state at the prefix boundary, but state can't be "page-shared" the way KV pages can. Instead, Mamba radix cache must:

1. **Save** the state at radix tree nodes (checkpoint the recurrent state)
2. **Restore** the state when a new request matches a cached prefix
3. **Propagate** state correctly when appending new tokens

## 2. Mamba State Checkpointing

At each radix tree node (representing a prefix), SGLang saves:

```text
mamba_state = {layer_idx: state_tensor}
```

Where `state_tensor` has shape `[D_state]` (or `[N_ssm_heads, D_state_per_head]`) for each Mamba layer.

When a new request's prompt matches a cached prefix:
1. The Mamba state is restored from the radix node
2. Subsequent prompt tokens update the state starting from the checkpoint
3. The final state is saved as the request's initial decode state

## 3. Key Differences from KV Radix Cache

| Aspect | KV Radix Cache | Mamba Radix Cache |
|---|---|---|
| Cached object | Multi-layer KV page table | Multi-layer recurrent state tensor |
| Sharing mechanism | Page-level reference counting | State copy/restore at nodes |
| Memory per node | O(S × Nkv × D) | O(L_mamba × D_state) |
| Eviction | Evict unused pages | Evict state checkpoints |

## 4. State Transfer in PD Disaggregation

When prefill and decode are separated (PD disaggregation):
- The prefill worker computes and sends the final Mamba state to the decode worker
- Unlike KV Cache transfer (which may involve large tensor transfers), Mamba state transfer is lightweight: `O(L_mamba × D_state)` bytes

## 5. SGLang Code References

- `python/sglang/srt/mem_cache/radix_cache.py` — RadixCache with Mamba state nodes
- `python/sglang/srt/mem_cache/mamba_pool.py` — MambaStatePool management
- `python/sglang/srt/managers/scheduler.py` — Mamba state lifecycle in scheduling
