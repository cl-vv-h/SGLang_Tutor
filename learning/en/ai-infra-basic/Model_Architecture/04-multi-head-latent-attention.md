# Multi-head Latent Attention: Low-Rank Compression & Decoupled RoPE

> 中文版：[Multi-head Latent Attention](../../../zh/ai-infra-basic/Model_Architecture/04-multi-head-latent-attention.md)

## Reading Map: Build the Whole Mental Model First

MLA is easy to lose inside a forest of projection matrices. Start with only two ideas:

1. **Compress on cache write**: instead of writing `N` complete K/V pairs for every historical token, write one shared content latent `c^{KV}` and one shared RoPE key;
2. **Change coordinates on cache read**: during decode, do not expand every historical latent back to complete K/V. Transform the current query and output projection into latent coordinates and read the compressed cache directly.

```mermaid
flowchart LR
    X["Layer input X<br/>[B,Lq,H]"] --> DQ["Query down projection<br/>[B,Lq,Dcq]"]
    DQ --> UQ["Query up projection and split"]
    UQ --> QC["Content query qC<br/>[B,Lq,N,Dh]"]
    UQ --> QR["Positional query qR<br/>[B,Lq,N,Dr]"]
    X --> DKV["KV down projection"]
    DKV --> CKV["Content latent cKV<br/>[B,Lq,Dc]"]
    DKV --> KR["Shared positional key kR<br/>[B,Lq,Dr]"]
    CKV --> CACHE[("Paged MLA cache<br/>historical cKV and kR")]
    KR --> CACHE
    QC --> ABSORB["Absorb K projection<br/>qC · WUK^T"]
    ABSORB --> SCORE["Content score: latent dot product"]
    QR --> ROPE["Positional dot product after RoPE"]
    CACHE --> SCORE
    CACHE --> ROPE
    SCORE --> ADD["Add, scale, and mask"]
    ROPE --> ADD
    ADD --> SM["Softmax weights p<br/>[B,N,Lq,Lkv]"]
    SM --> LATOUT["Weighted latent first<br/>[B,Lq,N,Dc]"]
    CACHE --> LATOUT
    LATOUT --> OABSORB["Absorbed V/output projection"]
    OABSORB --> Y["Layer output<br/>[B,Lq,H]"]
```

The diagram contains two timelines. `X -> CKV/KR -> Cache` **writes the current token**; `Cache -> Score/LatOut` **reads historical tokens**. In prefill, $`\mathrm{Lq}=\mathrm{Lkv}=S`$; in decode, usually $`\mathrm{Lq}=1`$ and $`\mathrm{Lkv}=S_{\mathrm{past}}+1`$.

### Running Shape Example

The rest of the chapter uses this teaching configuration. It is chosen to make shapes concrete, not to prescribe every model:

```math
B=2,\qquad S=4096,\qquad H=4096
```

```math
N=32,\qquad \mathrm{Dc}=512,\qquad \mathrm{Dcq}=1536
```

```math
\mathrm{Dh}=128,\qquad \mathrm{Dr}=64,\qquad \mathrm{Dv}=128
```

```text
dtype=BF16/FP16 (2 bytes per element)
```

Axes are fixed throughout: `B` batch size, `S/Lkv` history length, `Lq` query length of this call, `N` query heads, and `H` hidden size. $`\left[B,S,N,D\right]`$ in the exposition and $`\left[B,N,S,D\right]`$ in many kernels differ only by layout; their mathematical meaning is identical.

Standard MHA caches per token per layer:

```math
N \times  \left(\mathrm{Dh} + \mathrm{Dv}\right) = 32 \times  \left(128 + 128\right) = 8192 \mathrm{elements}
```

```math
\text{8192 × 2 bytes}= 16384 \mathrm{bytes} = 16 \mathrm{KiB}
```

MLA caches per token per layer:

```math
\mathrm{Dc} + \mathrm{Dr} = 512 + 64 = 576 \mathrm{elements}
```

```math
\text{576 × 2 bytes}= 1152 \mathrm{bytes} = 1.125 \mathrm{KiB}
```

The **logical KV payload** is therefore about $`\text{8192 / 576}= 14.22\times`$ smaller in this example. Real allocations also contain page alignment, block tables, quantization scales, and allocator metadata.

### One Decode Step, End to End

Assume `4095` tokens already exist and token `4096` is appended:

| Stage | Input shape | Operation | Output shape | Persistent? |
|---|---:|---|---:|---|
| Layer input | $`\left[2,1,4096\right]`$ | Previous sublayer/RMSNorm | $`\left[2,1,4096\right]`$ | No |
| KV write | $`\left[2,1,4096\right]`$ | `W_DKV_R` | `Ckv [2,1,512]`, `Krope [2,1,64]` | Yes |
| Q compression | $`\left[2,1,4096\right]`$ | `W_DQ` | `Cq [2,1,1536]` | No |
| Q expansion | $`\left[2,1,1536\right]`$ | `W_UQ` + split | `Qc [2,1,32,128]`, `Qr [2,1,32,64]` | No |
| History view | Block table | gather/page view | `Ckv_hist [2,4096,512]`, `Kr_hist [2,4096,64]` | Already cached |
| Content score | `Qc,Ckv_hist` | absorbed latent dot product | $`\left[2,32,1,4096\right]`$ | No |
| Position score | `RoPE(Qr),RoPE(Kr)` | dot product after head broadcast | $`\left[2,32,1,4096\right]`$ | No |
| Probabilities | both score terms | scale + causal mask + softmax | $`\left[2,32,1,4096\right]`$ | No |
| Latent reduction | probabilities, `Ckv_hist` | weighted sum over history | $`\left[2,1,32,512\right]`$ | No |
| Output | latent reduction | absorbed V/O transform and head merge | $`\left[2,1,4096\right]`$ | No |

This table is the ledger behind every later equation. If an implementation step cannot be found as an explicit operation, a layout transform, matrix absorption, or fused kernel has usually hidden it—the mathematical step has not vanished.

## 1. The Problem MLA Solves

Standard Multi-Head Attention stores complete K and V for every historical token, every layer:

```math
\mathrm{shape}\left(K_{\mathrm{cache}}\right)=\left[B,S,N,\mathrm{Dk}\right]
```

```math
\mathrm{shape}\left(V_{\mathrm{cache}}\right)=\left[B,S,N,\mathrm{Dv}\right]
```

Cache elements per token: $`N\cdot \left(\mathrm{Dk}+\mathrm{Dv}\right)`$. For long-context generation, KV Cache becomes the primary bottleneck for memory capacity and read bandwidth.

MLA's core idea: **Don't directly cache each head's expanded K/V. Instead, cache a shared low-dimensional latent vector; when attention is needed, up-project it to recover per-head content information.**

![MLA Dataflow](./assets/mla-flow.svg)

## 2. Unified Notation

| Symbol | Meaning |
|---|---|
| `X` | Input hidden states, $`\left[B,S,H\right]`$ |
| `N` | Number of attention heads |
| `Dc` | KV latent compression dimension |
| `Dcq` | Query latent compression dimension |
| `Dh` | Content key/query dimension per head |
| `Dr` | RoPE dimension per head |
| `Dv` | Value dimension per head |
| `Ckv` | Compressed KV latent, $`\left[B,S,\mathrm{Dc}\right]`$ |
| `Cq` | Compressed Query latent, $`\left[B,S,\mathrm{Dcq}\right]`$ |

MLA typically satisfies:

```text
Dc << N * (Dh + Dv)
```

Thus caching `Ckv` is more compact than caching all heads' K/V.

## 3. KV Down-Projection: Compressing Hidden State to Latent

For each token's hidden vector `x in R^H`, content latent and position key can be produced by a single joint projection:

```math
\left[c_{\mathrm{kv}},\mathrm{k\_rope\_raw}\right]=x\,\mathrm{W\_DKV\_R}
```

Batch shapes:

```math
\mathrm{shape}\left(X\right)=\left[B,S,H\right]
```

```math
\mathrm{shape}\left(\mathrm{W\_DKV\_R}\right)=\left[H,\mathrm{Dc}+\mathrm{Dr}\right]
```

```math
\mathrm{shape}\left(\mathrm{projected}\right)=\left[B,S,\mathrm{Dc}+\mathrm{Dr}\right]
```

```math
\mathrm{shape}\left(\mathrm{Ckv}\right)=\left[B,S,\mathrm{Dc}\right]
```

```math
\mathrm{shape}\left(\mathrm{K\_rope\_raw}\right)=\left[B,S,\mathrm{Dr}\right]
```

Two independent but equivalent logical projections can also separately produce `Ckv` and `K_rope_raw`. Content compression from `H` to `Dc` is not lossless copy — it's a learned low-rank representation.

## 4. KV Up-Projection: Recovering Content K and V from Latent

Two logically independent up-projections:

```math
K_{\mathrm{content}}=\mathrm{Ckv}\,W_{\mathrm{UK}}
```

```math
V=\mathrm{Ckv}\,W_{\mathrm{UV}}
```

Shapes:

```math
\mathrm{shape}\left(W_{\mathrm{UK}}\right)=\left[\mathrm{Dc},N\cdot \mathrm{Dh}\right]
```

```math
\mathrm{shape}\left(W_{\mathrm{UV}}\right)=\left[\mathrm{Dc},N\cdot \mathrm{Dv}\right]
```

```math
\mathrm{shape}\left(\mathrm{K\_content\_flat}\right)=\left[B,S,N\cdot \mathrm{Dh}\right]
```

```math
\mathrm{shape}\left(V_{\mathrm{flat}}\right)=\left[B,S,N\cdot \mathrm{Dv}\right]
```

```math
\mathrm{shape}\left(K_{\mathrm{content}}\right)=\left[B,S,N,\mathrm{Dh}\right]
```

```math
\mathrm{shape}\left(V\right)=\left[B,S,N,\mathrm{Dv}\right]
```

Each head's K/V is no longer independently projected from `X` directly, but shares the same `Ckv`, then recovers using different up-projection parameters.

## 5. Query Low-Rank Projection

Query can also use low-rank decomposition:

```math
\mathrm{Cq}=X\,W_{\mathrm{DQ}}
```

```math
Q_{\mathrm{full}}=\mathrm{RMSNorm}\left(\mathrm{Cq}\right)\,W_{\mathrm{UQ}}
```

Shapes:

```math
\mathrm{shape}\left(W_{\mathrm{DQ}}\right)=\left[H,\mathrm{Dcq}\right]
```

```math
\mathrm{shape}\left(\mathrm{Cq}\right)=\left[B,S,\mathrm{Dcq}\right]
```

```text
W_UQ: [Dcq,N*(Dh+Dr)]
Q_full: [B,S,N*(Dh+Dr)]
```

Split:

```math
\mathrm{shape}\left(Q_{\mathrm{content}}\right)=\left[B,S,N,\mathrm{Dh}\right]
```

```math
\mathrm{shape}\left(Q_{\mathrm{rope}}\right)=\left[B,S,N,\mathrm{Dr}\right]
```

Query doesn't need caching across generation steps, so Query compression mainly reduces parameter structure or changes projection computation — it doesn't directly determine KV Cache capacity.

## 6. Why RoPE Must Be Decoupled

If RoPE were directly applied to the complete K recovered from $`\mathrm{Ckv}\,W_{\mathrm{UK}}`$, the position-dependent rotation would block matrix absorption: the rotation matrix depends on token position and cannot be simply merged into fixed `W_UK`.

MLA therefore splits each Q/K head into two parts:

```math
Q_{\mathrm{head}}=\mathrm{concat}\left(Q_{\mathrm{content}},Q_{\mathrm{rope}}\right)
```

```math
K_{\mathrm{head}}=\mathrm{concat}\left(K_{\mathrm{content}},K_{\mathrm{rope}}\right)
```

Where:
- `Q_content` and `K_content` have no RoPE — they handle semantic content matching
- `Q_rope` and `K_rope` have RoPE — they handle relative position
- `K_rope` can be shared across heads, then broadcast to `N` heads

Shapes:

```math
\mathrm{shape}\left(Q_{\mathrm{content}}\right)=\left[B,S,N,\mathrm{Dh}\right]
```

```math
\mathrm{shape}\left(Q_{\mathrm{rope}}\right)=\left[B,S,N,\mathrm{Dr}\right]
```

```math
\mathrm{shape}\left(K_{\mathrm{content}}\right)=\left[B,S,N,\mathrm{Dh}\right]
```

```text
K_rope:    [B,S,1,Dr] -> broadcast [B,S,N,Dr]
```

Final per-head Query/Key dimension:

```math
\mathrm{Dqk}=\mathrm{Dh}+\mathrm{Dr}
```

## 7. MLA Attention Scores

For head `n`, query position `i`, key position `j`:

```math
{\mathrm{score}}_{n,i,j}={Q_{\mathrm{content}}}_{n,i}\cdot {K_{\mathrm{content}}}_{n,j}+{Q_{\mathrm{rope}}}_{n,i}\cdot {K_{\mathrm{rope}}}_{j}
```

Scaled: $`\mathrm{score}=\frac{\mathrm{score}}{\sqrt{\mathrm{Dh}+\mathrm{Dr}}}`$. After causal mask and softmax:

```math
O\left[n,i,:\right] = \sum_{j} \alpha\left[n,i,j\right] \cdot  V\left[n,j,:\right]
```

```math
\mathrm{shape}\left(O\right)=\left[B,S,N,\mathrm{Dv}\right]
```

Merge heads then restore $`\left[B,S,H\right]`$ via output projection.

## 8. Naive Implementation: Recover K/V First, Then Attention

Most direct:

```math
\mathrm{Ckv}=X\,W_{\mathrm{DKV}}\qquad\mathrm{shape}:\left[B,S,\mathrm{Dc}\right]
```

```math
\mathrm{Kc}=\mathrm{reshape}\left(\mathrm{Ckv}\,W_{\mathrm{UK}}\right)\qquad\mathrm{shape}:\left[B,S,N,\mathrm{Dh}\right]
```

```math
V=\mathrm{reshape}\left(\mathrm{Ckv}\,W_{\mathrm{UV}}\right)\qquad\mathrm{shape}:\left[B,S,N,\mathrm{Dv}\right]
```

```math
Q = \mathrm{build\_query}\left(X\right) \left[B,S,N,\mathrm{Dh}+\mathrm{Dr}\right]
```

```math
K = \mathrm{concat}\left(\mathrm{Kc}, \mathrm{broadcast}\left(K_{\mathrm{rope}}\right)\right) \left[B,S,N,\mathrm{Dh}+\mathrm{Dr}\right]
```

```math
O=\mathrm{softmax}\left(Q\,{K}^{\mathsf T}+\mathrm{mask}\right)\,V
```

Clear mathematically, but losing the bandwidth advantage of compressed cache during decode if full K/V is recovered every round.

## 9. Matrix Absorption: Don't Explicitly Recover Content K

For a single head, content score:

```math
q_{c}\cdot k_{c}=q_{c}\cdot c_{\mathrm{kv}}\,W_{\mathrm{UK}}=q_{c}\,{W_{\mathrm{UK}}}^{\mathsf T}\cdot c_{\mathrm{kv}}
```

Define latent query:

```math
q_{\mathrm{latent}}=q_{c}\,{W_{\mathrm{UK}}}^{\mathsf T}
```

```math
\mathrm{shape}\left(q_{\mathrm{latent}}\right)=\left[B,\mathrm{Lq},N,\mathrm{Dc}\right]
```

Content scores can directly dot-product with cached `Ckv [B,Lkv,Dc]`:

```math
\mathrm{shape}\left(\mathrm{content\_score}\right)=\left[B,N,\mathrm{Lq},\mathrm{Lkv}\right]
```

No need to materialize `K_content [B,Lkv,N,Dh]` for historical tokens.

## 10. Matrix Absorption: Don't Explicitly Recover Historical V

Naive value aggregation:

```math
o = \sum_{j} \alpha_{j} \cdot  v_{j} = \sum_{j} \alpha_{j} \cdot  \left(c_{j} \, W_{\mathrm{UV}}\right) = \left(\sum_{j} \alpha_{j} \cdot  c_{j}\right) \, W_{\mathrm{UV}}
```

So we can first aggregate in latent space:

```math
o_{\mathrm{latent}}=\alpha\,\mathrm{Ckv}\qquad\mathrm{shape}:\left[B,\mathrm{Lq},N,\mathrm{Dc}\right]
```

```math
o_{\mathrm{head}}=o_{\mathrm{latent}}\,W_{\mathrm{UV}}\qquad\mathrm{shape}:\left[B,\mathrm{Lq},N,\mathrm{Dv}\right]
```

Historical Value heads need no full expansion.

## 11. What Compressed KV Cache Stores

After decoupled RoPE, each historical token needs:

```math
\mathrm{shape}\left(\mathrm{Ckv}\right)=\left[\mathrm{Dc}\right]
```

```math
\mathrm{shape}\left(K_{\mathrm{rope}}\right)=\left[\mathrm{Dr}\right]
```

Single token, single layer cache elements:

```math
\text{MLA cache elements}= \mathrm{Dc} + \mathrm{Dr}
```

Standard MHA: $`N\cdot \left(\mathrm{Dk}+\mathrm{Dv}\right)`$. GQA: $`\mathrm{Nkv}\cdot \left(\mathrm{Dk}+\mathrm{Dv}\right)`$. MLA compression ratio ≈ $`\frac{\mathrm{Dc}+\mathrm{Dr}}{N\cdot \left(\mathrm{Dk}+\mathrm{Dv}\right)}`$.

## 12. Decode Dataflow

Current round, one new token per sequence:

```math
\mathrm{shape}\left(X_{\mathrm{new}}\right)=\left[B,1,H\right]
```

```math
\mathrm{shape}\left(\mathrm{Ckv\_new}\right)=\left[B,1,\mathrm{Dc}\right]
```

```math
\mathrm{shape}\left(\mathrm{K\_rope\_new}\right)=\left[B,1,\mathrm{Dr}\right]
```

Cache grows:

```math
\mathrm{shape}\left(\mathrm{Ckv\_cache}\right)=\left[B,\mathrm{Sctx},\mathrm{Dc}\right]
```

```math
\mathrm{shape}\left(\mathrm{K\_rope\_cache}\right)=\left[B,\mathrm{Sctx},\mathrm{Dr}\right]
```

Absorbed content path: `Q_content -> latent query [B,1,N,Dc] -> dot with Ckv_cache -> content scores [B,N,1,Sctx]`. Position path: `Q_rope × K_rope_cache -> position scores [B,N,1,Sctx]`. Sum, softmax, aggregate in latent space, map to head outputs.

## 13. Prefill Dataflow

Prefill computes `S` tokens in parallel. Higher compute density. Can choose:
1. Expand K/V then use standard matrix Attention
2. Compute directly in latent space
3. Hybrid based on shape and hardware

All must satisfy identical causal mask and mathematical mapping.

## 14. Parameter Meaning of Low-Rank Structure

KV projection decomposed as:

```text
X [H] -> Ckv [Dc] -> K/V heads [N*(Dh+Dv)]
```

Equivalent weight matrix has factored form $`\mathrm{W\_KV\_effective}=W_{\mathrm{DKV}}\,W_{\mathrm{UKV}}`$, with rank upper bound ≤ `Dc`. This imposes structural low-rank constraint.

## 15. MLA, MHA, GQA Structural Comparison

| Structure | Query Heads | Cached State | Inter-Head Sharing |
|---|---:|---|---|
| MHA | `N` | Full `N` K/V groups | None |
| MQA | `N` | 1 K/V group | All Q heads share same K/V |
| GQA | `N` | `Nkv` K/V groups | Each Q-head group shares K/V head |
| MLA | `N` | Shared latent + position key | Each head interprets same latent via up-projection |

GQA compresses cache by reducing KV head count; MLA compresses cache via low-rank latent representation. Sharing mechanisms differ.

## 16. Key Correctness Conditions for MLA

1. `W_DKV` and `W_UK/W_UV` must be jointly trained; latent is not post-hoc lossless compression.
2. RoPE content and position subspaces must be split as designed, otherwise matrix absorption doesn't hold.
3. Causal mask still acts on token position dimension; latent compression doesn't change autoregressive visibility.
4. Cache must simultaneously store content latent and position key that cannot be absorbed by fixed matrices.
5. Absorbed and expanded implementations should produce equivalent results within numerical error.

## 17. What MLA Solves—and What It Does Not

MLA primarily compresses the **feature dimension** of each historical KV entry. It does not remove the sequence dimension:

```math
\text{cache entries}: O\left(L\right)
```

```math
\text{dense prefill attention}: O\left(L^{2}\right)
```

```math
\text{dense decode attention per token}: O\left(L\right)
```

This distinction matters when reading newer architectures. DSA sparsifies which MLA entries enter the main core; CSA first compresses the sequence dimension and then selects compressed entries; KDA replaces token-addressable history with a fixed recurrent state.

## 18. Expanded and Absorbed Execution Modes

Production backends can expose the same MLA checkpoint through different kernel shapes:

- **expanded mode** reconstructs head-wise K/V and uses an MHA-like kernel;
- **absorbed mode** folds up-projection matrices into query/output paths and consumes the latent cache in an MQA-like kernel;
- **hybrid dispatch** chooses a path based on prefill/decode phase, sequence length, hardware, and supported dtypes.

The modes are mathematically related but have different intermediate memory, bandwidth, and kernel requirements. DeepSeek Sparse Attention uses the absorbed/MQA-form latent representation for its selected entries. An implementation must make the cache layout and backend mode explicit rather than inferring them from the model name.

## 19. SGLang Source Map

In the source snapshot used by this tutorial:

- MLA model wiring and absorbed weights: [`deepseek_v2.py`](../../../../python/sglang/srt/models/deepseek_v2.py);
- generic KV/latent memory pools: [`memory_pool.py`](../../../../python/sglang/srt/mem_cache/memory_pool.py);
- NPU-specific MLA module: [`deepseek_v2_attention_mla_npu.py`](../../../../python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py);
- FlashMLA Python entry point: [`flash_mla.py`](../../../../sgl-kernel/python/sgl_kernel/flash_mla.py);
- HIP path: [`hip_flash_mla.py`](../../../../python/sglang/srt/layers/attention/hip_flash_mla.py).

When debugging shape or performance issues, trace `model projection -> cache write -> backend dispatch -> kernel layout` and check prefill and decode independently.

## 20. Serving and Ascend Checklist

1. Record whether each phase uses expanded or absorbed execution.
2. Include both content latent and decoupled-RoPE key in cache-byte calculations.
3. Confirm paged-cache stride, block table, dtype, quantization scale, and alignment.
4. On Ascend, verify the selected path against the pinned SGLang, `torch_npu`, CANN, and custom-kernel versions.
5. Profile projections, RoPE, cache write, attention core, and output projection separately before fusing.
6. Use the same latent layout when composing MLA with DSA; do not gather expanded head-wise K/V by accident.

## 21. Continue Reading

- [Efficient-attention landscape](./06-efficient-attention-landscape.md)
- [DeepSeek Sparse Attention](./07-deepseek-sparse-attention.md)
- [Compressed Sparse Attention and HCA](./08-compressed-sparse-attention.md)
- [Kimi Delta Attention](./09-kimi-delta-attention.md)

## 22. References

- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
