# KV Cache Memory

[简体中文](../../../zh/ai-infra-basic/KV_Cache_Memory/README.md) | **English**

This chapter focuses on one of the most precious resources in LLM serving: KV Cache memory. Once you understand KV Cache shapes, allocation, reuse, and eviction, you'll understand much of scheduling, attention backend, prefix cache, and PD disaggregation design.

## What KV Cache Stores

Every Transformer attention layer produces Key and Value. At decode step `t`, you don't need to recompute K/V for the first `t-1` tokens — you read historical K/V from the KV Cache and only compute attention between the current token's Query and the cached K/V.

```text
Per layer:
    K cache: [historical tokens, kv_heads, head_dim]
    V cache: [historical tokens, kv_heads, head_dim]

Per request:
    K/V for prompt tokens
    K/V for generated tokens
```

## Memory Estimation Formula

Rough estimate:

```math
\text{KV bytes}= \mathrm{num\_layers} \cdot  \mathrm{num\_tokens} \cdot  \mathrm{kv\_heads} \cdot  \mathrm{head\_dim} \cdot  2 \qquad\text{K and V * dtype\_bytes}
```

Example: Llama-3-8B with 32 layers, 8 KV heads, 128 head dim, BF16, 4096 tokens:
$`32 \cdot  4096 \cdot  8 \cdot  128 \cdot  2 \cdot  2 \approx  512 \mathrm{MB}`$ per request for KV Cache alone.

## Key Design Points

- **Paged Attention**: KV Cache is allocated in pages (blocks), not contiguous per request — enabling memory sharing and efficient prefix caching.
- **Prefix Cache (RadixAttention)**: Requests sharing the same prompt prefix can share KV Cache pages.
- **Memory Pool**: `req_to_token_pool` and `token_to_kv_pool` manage the mapping from requests to KV slots.
- **HiCache**: Hierarchical cache that offloads cold KV Cache to CPU or SSD, keeping only hot KV Cache on GPU.
