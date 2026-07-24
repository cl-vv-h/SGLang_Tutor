[中文](./02-gqa-attention-shapes.md) | [English](./02-gqa-attention-shapes_EN.md)

# Grouped-Query Attention: QKV, RoPE & KV Cache Shape Flow

## 1. Attention Sublayer Input/Output

The Attention sublayer receives normalized hidden states:

```text
X: [T,H]
```

And returns a result of the same shape:

```text
O: [T,H]
```

Internally: QKV projection, head splitting, QK Norm, RoPE, causal Attention, head merging, and output projection.

![GQA Tensor Shape Flow](./assets/gqa-shape-flow.svg)

## 2. QKV Projection

Definitions:

```text
Nq  = number of query heads
Nkv = number of key/value heads
D   = head dimension
Qdim  = Nq * D
KVdim = Nkv * D
```

The three linear projections:

```text
Q_flat = X @ Wq
K_flat = X @ Wk
V_flat = X @ Wv
```

Weights and output shapes:

| Variable | Shape |
|---|---:|
| `X` | `[T,H]` |
| `Wq` | `[H,Qdim]` |
| `Wk` | `[H,KVdim]` |
| `Wv` | `[H,KVdim]` |
| `Q_flat` | `[T,Qdim]` |
| `K_flat`, `V_flat` | `[T,KVdim]` |

High-performance implementations typically concatenate the three weights into a single fused QKV projection:

```text
qkv = qkv_proj(X)
qkv: [T, Qdim + 2*KVdim]

q, k, v = split(qkv, [Qdim, KVdim, KVdim])
```

This fusion reduces kernel launches and intermediate memory accesses without changing the mathematical result.

## 3. Recovering Heads from Flat Dimensions

```text
Q_flat [T,Nq*D]   -> Q [T,Nq,D]
K_flat [T,Nkv*D]  -> K [T,Nkv,D]
V_flat [T,Nkv*D]  -> V [T,Nkv,D]
```

`T` is the packed token count, not sequence length. Each token has `Nq` Query heads but only `Nkv` Key/Value heads.

## 4. MHA, MQA & GQA

| Type | Relationship | How Each Query Head Uses K/V |
|---|---|---|
| MHA | `Nq=Nkv` | Each Query head has independent K/V heads |
| MQA | `Nkv=1` | All Query heads share a single K/V head |
| GQA | `1<Nkv<Nq` | A group of Query heads shares one K/V head |

GQA group size:

```text
G = Nq / Nkv
```

Query head `h` maps to KV head:

```text
kv_head(h) = floor(h / G)
```

Example: `Nq=8, Nkv=2, G=4`:

```text
Q heads 0,1,2,3 -> KV head 0
Q heads 4,5,6,7 -> KV head 1
```

GQA's main inference value is shrinking KV Cache. For identical `D` and context length, its KV Cache size is roughly `Nkv/Nq` of MHA.

## 5. QK Norm

Qwen3-MoE applies RMSNorm to each Q/K head's `D`-dimensional vector:

```text
Q: [T,Nq,D]   -> Q_norm: [T,Nq,D]
K: [T,Nkv,D]  -> K_norm: [T,Nkv,D]
```

Normalization occurs on the last dimension:

```text
Q_norm[t,h,:] = RMSNorm(Q[t,h,:])
K_norm[t,h,:] = RMSNorm(K[t,h,:])
```

Shape unchanged. QK Norm controls vector scale before dot product, preventing different heads or layers from producing excessively large attention logits.

## 6. RoPE (Rotary Position Embedding)

RoPE encodes position `p` as rotation in 2D subspaces. For a pair of components in an even-dimensional vector:

```text
[x_2i']     [ cos(theta_p,i)  -sin(theta_p,i) ] [x_2i  ]
[x_2i+1'] = [ sin(theta_p,i)   cos(theta_p,i) ] [x_2i+1]
```

It acts on Q and K, not on V:

```text
positions: [T]
Q_norm: [T,Nq,D]   -> Q_rope: [T,Nq,D]
K_norm: [T,Nkv,D]  -> K_rope: [T,Nkv,D]
V:      [T,Nkv,D]  -> unchanged
```

All heads of the same token use the same position id, but different frequency dimensions use different rotation angles. The dot product `Q_rope(p) · K_rope(q)` thus carries relative position `p-q` information.

## 7. Causal Attention Math Shapes

Consider a single request, single Query head. Current Query length `Lq`, visible KV length `Lkv`:

```text
Q_h: [Lq,D]
K_h: [Lkv,D]
V_h: [Lkv,D]
```

Scores:

```text
S_h = Q_h @ K_h^T / sqrt(D)
S_h: [Lq,Lkv]
```

After causal masking:

```text
S_h[i,j] = -infinity, if key position j > query position i
```

Probabilities and output:

```text
A_h = softmax(S_h, dim=-1)   [Lq,Lkv]
O_h = A_h @ V_h              [Lq,D]
```

Merging all Query heads:

```text
O_heads: [T,Nq,D]
O_flat:  [T,Nq*D]
```

Production-grade FlashAttention or decode attention kernels don't fully materialize the `[Nq,Lq,Lkv]` score matrix — they compute softmax and weighted sum in blocks. The shape remains the correct logical view for understanding mathematical dependencies.

## 8. Sequence Boundaries in Packed Batches

Suppose two requests' current tokens are packed:

```text
request A: 3 tokens
request B: 2 tokens
Q packed: [5,Nq,D]
```

You cannot treat this as a single sequence of length 5. Attention metadata describes:

```text
request A rows: [0,3)
request B rows: [3,5)
sequence lengths: [3,2]
prefix lengths: [prefix_A,prefix_B]
KV cache locations: per-request page/index mapping
```

The kernel constructs independent visible ranges per request from metadata. A's Query won't read B's K/V, and B won't read A's.

## 9. KV Cache Writing

Each Decoder Layer has independent K/V Cache. The current layer produces:

```text
K_new: [T,Nkv,D]
V_new: [T,Nkv,D]
```

Logically, for request `r`'s token position `p`:

```text
K_cache[layer,r,p,:,:] = K_new[token_row,:,:]
V_cache[layer,r,p,:,:] = V_new[token_row,:,:]
```

Serving systems typically don't allocate contiguous `[B,Lmax,Nkv,D]`. Paged KV Cache maps token positions to physical slots via `req_to_token` mapping.

## 10. Prefill Attention

For an uncached prompt of length `S`:

```text
Q: [S,Nq,D]
K: [S,Nkv,D]
V: [S,Nkv,D]
logical scores: [Nq,S,S]
```

Causal visibility is lower-triangular. Row `i` can only read `[0,i]`.

With prefix cache of length `P` and current extend of `E`:

```text
Q_new: [E,Nq,D]
K_new,V_new: [E,Nkv,D]
K_visible,V_visible: [P+E,Nkv,D]
logical scores: [Nq,E,P+E]
```

Only compute Query for new tokens; historical prefix K/V are read directly from cache.

## 11. Decode Attention

Normal decode has one new token per request:

```text
Lq = 1
Q_new: [1,Nq,D]
K_new,V_new: [1,Nkv,D]
K_visible,V_visible: [Lctx,Nkv,D]
logical scores: [Nq,1,Lctx]
```

With `B` requests packed:

```text
Q_new: [B,Nq,D]
K_new,V_new: [B,Nkv,D]
```

Each request's `Lctx` can differ. The decode kernel reads each request's history via request-to-KV-slot mapping.

## 12. Output Projection

Attention head outputs merged:

```text
O_heads: [T,Nq,D]
O_flat: [T,Nq*D]
```

Output weight:

```text
Wo: [Nq*D,H]
O = O_flat @ Wo
O: [T,H]
```

If `Nq*D=H`, input and output widths match, but `Wo` still performs cross-head mixing.

## 13. Local Shapes Under Tensor Parallelism

Let attention TP size be `Ptp`, assuming `Nq` is divisible:

```text
Nq_local = Nq / Ptp
Qdim_local = Nq_local * D
```

If `Nkv >= Ptp` and divisible:

```text
Nkv_local = Nkv / Ptp
```

If `Nkv < Ptp`, one common strategy replicates KV heads across some TP ranks:

```text
Nkv_local = max(1, Nkv / Ptp)
```

Single-rank typical shapes:

```text
q: [T,Nq_local,D]
k: [T,Nkv_local,D]
v: [T,Nkv_local,D]
attn output local: [T,Nq_local*D]
```

## 14. Reference Attention Implementation Order

Ignoring specific frameworks, the GQA implementation order:

```text
qkv = linear(hidden_states, Wqkv)
q, k, v = split(qkv)
q = reshape(q, [B,S,Nq,D])
k = reshape(k, [B,S,Nkv,D])
v = reshape(v, [B,S,Nkv,D])
q = rms_norm_per_head(q)
k = rms_norm_per_head(k)
q, k = apply_rope(q, k, positions)
k_cache, v_cache = append_cache(k, v)
attn_output = causal_gqa(q, k_cache, v_cache)
output = linear(merge_heads(attn_output), Wo)
```

Key variables:

| Variable | Logical Shape |
|---|---:|
| `hidden_states` | `[T,H]` |
| `qkv` | `[T,q_size+2*kv_size]` |
| `q` | `[T,q_size]`, or recovered as `[T,Nq_local,D]` |
| `k`, `v` | `[T,kv_size]`, or recovered as `[T,Nkv_local,D]` |
| `attn_output` | `[T,q_size]` |
| `output` | `[T,H]` or equivalent local representation before communication |

## 15. KV Cache Capacity Formula

Ignoring alignment, quantization, and allocator metadata, approximate KV Cache bytes per request:

```text
bytes = L * Sctx * 2 * Nkv * D * bytes_per_element
```

Where `2` accounts for K and V. For `B` requests:

```text
total_bytes = sum_r L * Sctx_r * 2 * Nkv * D * bytes_per_element
```

GQA reduces `Nkv`, thus directly reducing persistent KV state across all layers and tokens.
