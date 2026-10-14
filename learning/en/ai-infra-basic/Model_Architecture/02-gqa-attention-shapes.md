# Grouped-Query Attention: QKV, RoPE & KV Cache Shape Flow

## 1. Attention Sublayer Input/Output

The Attention sublayer receives normalized hidden states:

```math
\mathrm{shape}\left(X\right)=\left[T,H\right]
```

And returns a result of the same shape:

```math
\mathrm{shape}\left(O\right)=\left[T,H\right]
```

Internally: QKV projection, head splitting, QK Norm, RoPE, causal Attention, head merging, and output projection.

![GQA Tensor Shape Flow](./assets/gqa-shape-flow.svg)

## 2. QKV Projection

Definitions:

```text
Nq  = number of query heads
```

```math
\mathrm{Nkv} = \mathrm{number} \mathrm{of} \mathrm{key}/\mathrm{value} \mathrm{heads}
```

```text
D   = head dimension
```

```math
\mathrm{Qdim}=\mathrm{Nq}\cdot D
```

```math
\mathrm{KVdim}=\mathrm{Nkv}\cdot D
```

The three linear projections:

```math
Q_{\mathrm{flat}}=X\,\mathrm{Wq}
```

```math
K_{\mathrm{flat}}=X\,\mathrm{Wk}
```

```math
V_{\mathrm{flat}}=X\,\mathrm{Wv}
```

Weights and output shapes:

| Variable | Shape |
|---|---:|
| `X` | $`\left[T,H\right]`$ |
| `Wq` | $`\left[H,\mathrm{Qdim}\right]`$ |
| `Wk` | $`\left[H,\mathrm{KVdim}\right]`$ |
| `Wv` | $`\left[H,\mathrm{KVdim}\right]`$ |
| `Q_flat` | $`\left[T,\mathrm{Qdim}\right]`$ |
| `K_flat`, `V_flat` | $`\left[T,\mathrm{KVdim}\right]`$ |

High-performance implementations typically concatenate the three weights into a single fused QKV projection:

```math
\mathrm{qkv} = \mathrm{qkv\_proj}\left(X\right)
```

```math
\mathrm{shape}\left(\mathrm{qkv}\right)=\left[T,\mathrm{Qdim}+2\cdot \mathrm{KVdim}\right]
```

```math
\text{q, k, v}= \mathrm{split}\left(\mathrm{qkv}, \left[\mathrm{Qdim}, \mathrm{KVdim}, \mathrm{KVdim}\right]\right)
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
| MHA | $`\mathrm{Nq}=\mathrm{Nkv}`$ | Each Query head has independent K/V heads |
| MQA | $`\mathrm{Nkv}=1`$ | All Query heads share a single K/V head |
| GQA | $`1 \lt \mathrm{Nkv} \lt \mathrm{Nq}`$ | A group of Query heads shares one K/V head |

GQA group size:

```math
G=\frac{\mathrm{Nq}}{\mathrm{Nkv}}
```

Query head `h` maps to KV head:

```math
\mathrm{kv\_head}\left(h\right) = \mathrm{floor}\left(h / G\right)
```

Example: $`\mathrm{Nq}=8,\qquad \mathrm{Nkv}=2,\qquad G=4`$:

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

```math
{Q_{\mathrm{norm}}}_{t,h,:}=\mathrm{RMSNorm}\left({Q}_{t,h,:}\right)
```

```math
{K_{\mathrm{norm}}}_{t,h,:}=\mathrm{RMSNorm}\left({K}_{t,h,:}\right)
```

Shape unchanged. QK Norm controls vector scale before dot product, preventing different heads or layers from producing excessively large attention logits.

## 6. RoPE (Rotary Position Embedding)

RoPE encodes position `p` as rotation in 2D subspaces. For a pair of components in an even-dimensional vector:

```math
x'_{2i}=\cos(\theta_{p,i})x_{2i}-\sin(\theta_{p,i})x_{2i+1}
```

```math
x'_{2i+1}=\sin(\theta_{p,i})x_{2i}+\cos(\theta_{p,i})x_{2i+1}
```

It acts on Q and K, not on V:

```math
\mathrm{shape}\left(\mathrm{positions}\right)=\left[T\right]
```

```text
Q_norm: [T,Nq,D]   -> Q_rope: [T,Nq,D]
K_norm: [T,Nkv,D]  -> K_rope: [T,Nkv,D]
V:      [T,Nkv,D]  -> unchanged
```

All heads of the same token use the same position id, but different frequency dimensions use different rotation angles. The dot product `Q_rope(p) · K_rope(q)` thus carries relative position `p-q` information.

## 7. Causal Attention Math Shapes

Consider a single request, single Query head. Current Query length `Lq`, visible KV length `Lkv`:

```math
\mathrm{shape}\left(Q_{h}\right)=\left[\mathrm{Lq},D\right]
```

```math
\mathrm{shape}\left(K_{h}\right)=\left[\mathrm{Lkv},D\right]
```

```math
\mathrm{shape}\left(V_{h}\right)=\left[\mathrm{Lkv},D\right]
```

Scores:

```math
S_{h}=\frac{Q_{h}\,{K_{h}}^{\mathsf T}}{\sqrt{D}}
```

```math
\mathrm{shape}\left(S_{h}\right)=\left[\mathrm{Lq},\mathrm{Lkv}\right]
```

After causal masking:

```text
S_h[i,j] = -infinity, if key position j > query position i
```

Probabilities and output:

```math
A_{h} = \mathrm{softmax}\left(S_{h}, \mathrm{dim}=-1\right) \left[\mathrm{Lq},\mathrm{Lkv}\right]
```

```math
O_{h}=A_{h}\,V_{h}\qquad\mathrm{shape}:\left[\mathrm{Lq},D\right]
```

Merging all Query heads:

```math
\mathrm{shape}\left(O_{\mathrm{heads}}\right)=\left[T,\mathrm{Nq},D\right]
```

```math
\mathrm{shape}\left(O_{\mathrm{flat}}\right)=\left[T,\mathrm{Nq}\cdot D\right]
```

Production-grade FlashAttention or decode attention kernels don't fully materialize the $`\left[\mathrm{Nq},\mathrm{Lq},\mathrm{Lkv}\right]`$ score matrix — they compute softmax and weighted sum in blocks. The shape remains the correct logical view for understanding mathematical dependencies.

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

```math
\mathrm{shape}\left(K_{\mathrm{new}}\right)=\left[T,\mathrm{Nkv},D\right]
```

```math
\mathrm{shape}\left(V_{\mathrm{new}}\right)=\left[T,\mathrm{Nkv},D\right]
```

Logically, for request `r`'s token position `p`:

```math
{K_{\mathrm{cache}}}_{\mathrm{layer},r,p,:,:}={K_{\mathrm{new}}}_{\mathrm{token\_row},:,:}
```

```math
{V_{\mathrm{cache}}}_{\mathrm{layer},r,p,:,:}={V_{\mathrm{new}}}_{\mathrm{token\_row},:,:}
```

Serving systems typically don't allocate contiguous $`\left[B,\mathrm{Lmax},\mathrm{Nkv},D\right]`$. Paged KV Cache maps token positions to physical slots via `req_to_token` mapping.

## 10. Prefill Attention

For an uncached prompt of length `S`:

```math
\mathrm{shape}\left(Q\right)=\left[S,\mathrm{Nq},D\right]
```

```math
\mathrm{shape}\left(K\right)=\left[S,\mathrm{Nkv},D\right]
```

```math
\mathrm{shape}\left(V\right)=\left[S,\mathrm{Nkv},D\right]
```

```text
logical scores: [Nq,S,S]
```

Causal visibility is lower-triangular. Row `i` can only read $`\left[0,i\right]`$.

With prefix cache of length `P` and current extend of `E`:

```math
\mathrm{shape}\left(Q_{\mathrm{new}}\right)=\left[E,\mathrm{Nq},D\right]
```

```text
K_new,V_new: [E,Nkv,D]
K_visible,V_visible: [P+E,Nkv,D]
logical scores: [Nq,E,P+E]
```

Only compute Query for new tokens; historical prefix K/V are read directly from cache.

## 11. Decode Attention

Normal decode has one new token per request:

```math
\mathrm{Lq}=1
```

```math
\mathrm{shape}\left(Q_{\mathrm{new}}\right)=\left[1,\mathrm{Nq},D\right]
```

```text
K_new,V_new: [1,Nkv,D]
K_visible,V_visible: [Lctx,Nkv,D]
logical scores: [Nq,1,Lctx]
```

With `B` requests packed:

```math
\mathrm{shape}\left(Q_{\mathrm{new}}\right)=\left[B,\mathrm{Nq},D\right]
```

```text
K_new,V_new: [B,Nkv,D]
```

Each request's `Lctx` can differ. The decode kernel reads each request's history via request-to-KV-slot mapping.

## 12. Output Projection

Attention head outputs merged:

```math
\mathrm{shape}\left(O_{\mathrm{heads}}\right)=\left[T,\mathrm{Nq},D\right]
```

```math
\mathrm{shape}\left(O_{\mathrm{flat}}\right)=\left[T,\mathrm{Nq}\cdot D\right]
```

Output weight:

```math
\mathrm{shape}\left(\mathrm{Wo}\right)=\left[\mathrm{Nq}\cdot D,H\right]
```

```math
O=O_{\mathrm{flat}}\,\mathrm{Wo}
```

```math
\mathrm{shape}\left(O\right)=\left[T,H\right]
```

If $`\mathrm{Nq}\cdot D=H`$, input and output widths match, but `Wo` still performs cross-head mixing.

## 13. Local Shapes Under Tensor Parallelism

Let attention TP size be `Ptp`, assuming `Nq` is divisible:

```math
\mathrm{Nq\_local}=\frac{\mathrm{Nq}}{\mathrm{Ptp}}
```

```math
\mathrm{Qdim\_local}=\mathrm{Nq\_local}\cdot D
```

If $`\mathrm{Nkv} \ge \mathrm{Ptp}`$ and divisible:

```math
\mathrm{Nkv\_local}=\frac{\mathrm{Nkv}}{\mathrm{Ptp}}
```

If $`\mathrm{Nkv} \lt \mathrm{Ptp}`$, one common strategy replicates KV heads across some TP ranks:

```math
\mathrm{Nkv\_local}=\mathrm{max}\left(1,\frac{\mathrm{Nkv}}{\mathrm{Ptp}}\right)
```

Single-rank typical shapes:

```math
\mathrm{shape}\left(q\right)=\left[T,\mathrm{Nq\_local},D\right]
```

```math
\mathrm{shape}\left(k\right)=\left[T,\mathrm{Nkv\_local},D\right]
```

```math
\mathrm{shape}\left(v\right)=\left[T,\mathrm{Nkv\_local},D\right]
```

```text
attn output local: [T,Nq_local*D]
```

## 14. Reference Attention Implementation Order

Ignoring specific frameworks, the GQA implementation order:

```math
\mathrm{qkv} = \mathrm{linear}\left(\mathrm{hidden\_states}, \mathrm{Wqkv}\right)
```

```math
\text{q, k, v}= \mathrm{split}\left(\mathrm{qkv}\right)
```

```math
q=\mathrm{reshape}\left(q,\left[B,S,\mathrm{Nq},D\right]\right)
```

```math
k=\mathrm{reshape}\left(k,\left[B,S,\mathrm{Nkv},D\right]\right)
```

```math
v=\mathrm{reshape}\left(v,\left[B,S,\mathrm{Nkv},D\right]\right)
```

```math
q = \mathrm{rms\_norm\_per\_head}\left(q\right)
```

```math
k = \mathrm{rms\_norm\_per\_head}\left(k\right)
```

```math
\text{q, k}= \mathrm{apply\_rope}\left(q, k, \mathrm{positions}\right)
```

```math
\text{k\_cache, v\_cache}= \mathrm{append\_cache}\left(k, v\right)
```

```math
\mathrm{attn\_output} = \mathrm{causal\_gqa}\left(q, k_{\mathrm{cache}}, v_{\mathrm{cache}}\right)
```

```math
\mathrm{output} = \mathrm{linear}\left(\mathrm{merge\_heads}\left(\mathrm{attn\_output}\right), \mathrm{Wo}\right)
```

Key variables:

| Variable | Logical Shape |
|---|---:|
| `hidden_states` | $`\left[T,H\right]`$ |
| `qkv` | $`\left[T,q_{\mathrm{size}}+2\cdot \mathrm{kv\_size}\right]`$ |
| `q` | $`\left[T,q_{\mathrm{size}}\right]`$, or recovered as $`\left[T,\mathrm{Nq\_local},D\right]`$ |
| `k`, `v` | $`\left[T,\mathrm{kv\_size}\right]`$, or recovered as $`\left[T,\mathrm{Nkv\_local},D\right]`$ |
| `attn_output` | $`\left[T,q_{\mathrm{size}}\right]`$ |
| `output` | $`\left[T,H\right]`$ or equivalent local representation before communication |

## 15. KV Cache Capacity Formula

Ignoring alignment, quantization, and allocator metadata, approximate KV Cache bytes per request:

```math
\mathrm{bytes}=L\cdot \mathrm{Sctx}\cdot 2\cdot \mathrm{Nkv}\cdot D\cdot \mathrm{bytes\_per\_element}
```

Where `2` accounts for K and V. For `B` requests:

```math
\mathrm{total\_bytes} = \sum_{r} L \cdot  \mathrm{Sctx\_r} \cdot  2 \cdot  \mathrm{Nkv} \cdot  D \cdot  \mathrm{bytes\_per\_element}
```

GQA reduces `Nkv`, thus directly reducing persistent KV state across all layers and tokens.
