[中文](./09-router.md) | [English](./09-router_EN.md)

# SGLang Router: SmartRouter, KV-Aware Routing & More

## 1. Multiple Router Concepts in SGLang

The word "router" appears in several SGLang contexts:

| Router Type | Location | Purpose |
|---|---|---|
| **sgl-router** (Rust) | `experimental/sgl-router/` | HTTP-level load balancer; KV-aware routing for PD disaggregation |
| **SmartRouter** | Python `srt/entrypoints/` | Ollama-format request-level routing |
| **Schedule Simulator Router** | Python `srt/managers/` | Random, RoundRobin, Sticky for scheduling simulation |
| **PD Bootstrap Route** | Python `srt/disaggregation/` | Rank connection registration for PD workers |
| **MoE Expert Router** | Python `srt/layers/moe/` | Token-to-expert Top-K routing inside MoE layers |

## 2. sgl-router (Rust) Architecture

```
Client → sgl-router (Rust, port 30000) → SGLang workers (Python)
```

Key features:
- Discovery: Static URL list or K8s EndpointSlice
- Policies: RoundRobin, Random, PowerOfTwo, LoadBased, Sticky, CacheAwareZmq
- PD Separation: Separate prefill/decode worker pools
- Proxy modes: JSON forwarding, SSE streaming bridge
- CircuitBreaker: Fault isolation

### CacheAwareZmq Policy

KV-aware routing: routes requests with similar prefixes to the same worker to maximize RadixCache hit rate:
1. Maintains a HashTree of prefix → worker mapping
2. Receives KV events from workers (cache insert/evict)
3. Routes new requests to workers with matching cached prefixes

## 3. Python-Side Routers

### MoE Expert Router

Inside each MoE layer:
```python
router_logits = hidden_states @ router_weight  # [T, E]
topk_weights, topk_ids = topk_softmax(router_logits)  # [T, K]
# Dispatch tokens to experts, compute, combine
```

### PD Bootstrap `/route`

When PD disaggregation starts:
- Decode workers register with the prefill worker
- `POST /route` establishes the connection
- The prefill worker learns which decode workers are available

## 4. Recommended Reading Order

1. Start with `sgl-router` Rust code — understand HTTP routing layer
2. Read CacheAwareZmqPolicy — understand KV-aware routing algorithm
3. Read SmartRouter — understand Python-side request routing
4. Read MoE Expert Router — understand per-layer token routing
5. Read PD Bootstrap route — understand disaggregation worker registration

## 5. Key Source Files

| File | Content |
|---|---|
| `experimental/sgl-router/src/main.rs` | Server startup, Axum route registration |
| `experimental/sgl-router/src/policy/cache_aware_zmq.rs` | KV-aware routing algorithm |
| `experimental/sgl-router/src/policy/registry.rs` | Worker discovery and registration |
| `python/sglang/srt/layers/moe/router.py` | MoE Top-K router |
| `python/sglang/srt/disaggregation/decode.py` | PD decode worker `/route` |
