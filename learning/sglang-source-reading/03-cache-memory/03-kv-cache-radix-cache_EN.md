[中文](./03-kv-cache-radix-cache.md) | [English](./03-kv-cache-radix-cache_EN.md)

# KV Cache & Radix Cache

## 1. KV Cache Memory Pool

SGLang's KV Cache is managed through two key pools:

| Pool | Purpose | Shape Mapping |
|---|---|---|
| `req_to_token_pool` | Maps request slot → token slots | `[max_reqs, max_tokens_per_req]` |
| `token_to_kv_pool` | Maps token slot → KV cache memory | `[max_total_tokens, ...]` per layer |

## 2. Paged KV Cache

Instead of contiguous per-request allocation, KV Cache is paged:
- Page size typically 16 or 128 tokens (varies by backend)
- Each request holds a page table mapping logical token positions → physical pages
- Pages can be shared between requests (prefix caching)

## 3. RadixCache (Prefix Cache)

```mermaid
flowchart TD
  Root["Root Node"] --> A["Token: 'Hello'"]
  A --> B["Token: 'World'"]
  A --> C["Token: 'there'"]
  B --> D["Token: '!'"]
  C --> E["Token: ',']
```

A radix tree (compact prefix tree) where:
- Each node represents a token sequence prefix
- Nodes hold references to shared KV Cache pages
- New requests matching a prefix reuse cached KV pages
- When no requests reference a node, its KV pages are freed

## 4. HiCache (Hierarchical Cache)

Extends RadixCache with multi-tier storage:
- **GPU (Hot)**: Frequently accessed KV Cache
- **CPU (Warm)**: Less frequently accessed, swapped out
- **SSD (Cold)**: Long-term storage for infrequent prefixes

HiCache operations:
- `prefetch`: Move data from cold → warm → hot before needed
- `write-through`: Write KV to both GPU and CPU simultaneously
- `eviction`: Remove least-used entries when capacity is full

## 5. KV Cache Lifecycle

```text
Request arrives
  → RadixCache.match(prefix)  # Find matching prefix
  → Allocate new pages for unmatched tokens
  → Prefill: write KV to new pages
  → Decode: append one token's KV per step
  → Request ends: decrement ref counts, free unreferenced pages
```

## 6. Integration with Scheduler

- `get_next_batch_to_run()` checks KV cache availability before forming batches
- `PrefillAdder` enforces KV token budget
- `update_running_batch()` checks decode memory, retracts if insufficient
- `tree_cache` (RadixCache/HiCache) is the Scheduler's primary cache interface
