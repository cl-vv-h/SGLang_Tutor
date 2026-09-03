# Efficient Attention Landscape: Compression, Sparsity, Recurrence, and Kernels

## 1. Why an Attention Map Is Necessary

Names such as GQA, MLA, DSA, CSA, KDA, and FlashAttention are often placed in one list, but they optimize different parts of the system:

- **State compression** asks what history is cached.
- **Sequence sparsity** asks which historical positions are read.
- **Recurrent compression** asks whether token history can be folded into fixed-size state.
- **Kernel optimization** asks how to execute the same mathematical result with less HBM traffic.

If these axes are mixed together, it is easy to make incorrect claims such as “MLA makes prefill linear” or “FlashAttention is sparse attention.”

## 2. Start from Dense Causal Attention

For sequence length `L`, query/key head dimension `Dk`, and value dimension `Dv`:

```text
score[i,j] = q_i dot k_j / sqrt(Dk),  j <= i
output[i] = sum_j softmax(score[i,:])[j] * v_j
```

Ignoring projections and head constants:

| Phase | Core attention work | Historical state read by one decode step |
|---|---:|---:|
| Full-sequence prefill/training | `O(L^2 * D)` | not applicable |
| One-token decode | `O(L * D)` | `O(L * Dcache)` |

Every efficient design changes one or more terms in this baseline.

## 3. Four Independent Optimization Axes

### 3.1 Compress the Head Axis: MQA and GQA

MHA stores one K/V pair per query head. MQA stores one shared K/V pair. GQA keeps several K/V groups:

```text
MHA cache per token: Nq  * (Dk + Dv)
GQA cache per token: Nkv * (Dk + Dv), Nkv < Nq
MQA cache per token:       Dk + Dv
```

They reduce cache width, but a query still visits every visible position. Dense prefill remains quadratic in `L`.

### 3.2 Compress the Feature Axis: MLA

MLA stores a learned latent vector and a positional component instead of expanded per-head K/V:

```text
MLA cache per token: Dc + Dr
```

Matrix absorption lets decode consume this latent cache directly. MLA primarily reduces cache capacity and bandwidth; it does not reduce the number of historical positions visited.

### 3.3 Reduce the Position Set: SWA, NSA, DSA, CSA

These methods limit or transform the visible history:

- **Sliding Window Attention (SWA)** uses a fixed local window.
- **Native Sparse Attention (NSA)** combines compressed global tokens, selected fine-grained blocks, and a local window.
- **DeepSeek Sparse Attention (DSA)** trains a lightweight indexer to select token-level MLA entries.
- **Compressed Sparse Attention (CSA)** first compresses multiple source tokens into one entry, then selects top-k compressed entries.

This axis can reduce the main attention work from visiting `L` entries per query to visiting a bounded or slowly growing subset.

### 3.4 Replace Explicit History with Recurrent State: GDN and KDA

GDN and KDA do not perform softmax retrieval over a set of cached historical tokens. They fold history into a state matrix:

```text
state per request and layer: [num_heads, value_dim, key_dim]
```

The state size does not grow with context length. This provides linear sequence processing and constant-size runtime state, but exact token-addressable retrieval is replaced by compressed associative memory.

## 4. Architecture Versus Kernel

FlashAttention and FlashDecoding normally preserve the mathematical result of dense attention:

```text
same visible tokens + same softmax result
different tiling, online softmax, and HBM traffic
```

PagedAttention primarily changes cache allocation and address translation. It does not define a new trained attention architecture either.

Architecture and kernel choices can be composed:

```text
GQA + FlashAttention
MLA + paged latent cache + FlashMLA
DSA + learned indexer + sparse attention kernel
KDA + chunkwise prefill kernel + fused recurrent decode kernel
```

## 5. Unified Comparison

Let `k` be the selected-entry count, `w` the local window, `m` a compression ratio, and `Sstate` the recurrent-state size. Constants and projection work are omitted.

| Method | Historical representation | Positions/entries visited per query | Prefill core trend | Decode state growth |
|---|---|---:|---:|---:|
| MHA | full K/V per head | `L` | `O(L^2)` | `O(L)` |
| GQA/MQA | shared/grouped K/V | `L` | `O(L^2)` | `O(L)` with smaller slope |
| MLA | latent KV + positional key | `L` | `O(L^2)` | `O(L)` with smaller slope |
| SWA | recent K/V | `w` | `O(L*w)` | `O(w)` if old entries are evicted |
| NSA | compressed + selected blocks + window | design-dependent | approximately linear for fixed budgets | usually grows through compressed/selected cache |
| DSA | MLA entries + indexer keys | main path `k`; indexer scans history | main path `O(L*k)` | `O(L)` cache; bounded main-attention read |
| CSA | compressed entries + indexer keys + local raw tail | main path `k+w`; indexer scans about `L/m` | main path `O(L*(k+w))` | about `O(L/m)` plus bounded local state |
| HCA | heavily compressed entries + local raw tail | about `L/m' + w` | `O(L^2/m')` | about `O(L/m')` plus bounded local state |
| GDN/KDA | recurrent matrix + short-conv state | fixed state | `O(L*Sstate)` | `O(Sstate)` |

The DSA/CSA indexer still scans a growing history. “Top-k is constant” therefore does not mean the complete layer is strictly `O(1)` per decode token.

## 6. The Three Kinds of Compression Are Not Interchangeable

### 6.1 Head/Feature Compression

GQA and MLA retain one entry for every token but reduce each entry's width. They are good at KV capacity and decode-bandwidth reduction while keeping dense token visibility.

### 6.2 Sequence Compression

CSA and HCA merge several source tokens into a compressed entry. The cache has fewer time-axis entries, but each entry is a learned summary rather than an exact token record.

### 6.3 Recurrent Compression

GDN and KDA repeatedly overwrite a fixed-size state. There is no list of old entries to gather later. Prefix sharing, rollback, speculative verification, and state transfer consequently need state-aware semantics rather than ordinary KV-page semantics.

## 7. Sparse Attention Design Families

### 7.1 Fixed Pattern

SWA, local-global patterns, and dilated patterns determine connectivity from position alone. They are predictable and kernel-friendly, but cannot retrieve arbitrary distant content unless a global path is added.

### 7.2 Query-Dependent Retrieval

DSA learns a cheap score for each query-history pair and selects top-k entries. The selection pattern adapts to content, but indexer cost, top-k reduction, irregular gather, and cache addressing become first-class system work.

### 7.3 Hierarchical Compression and Selection

NSA and CSA first reduce the search space through compression and then retain a fine-grained or selected path. A local window repairs near-field details that compression boundaries could hide.

NSA and DSA are separate research mechanisms. In this SGLang snapshot, the backend string `nsa` is a deprecated alias for `dsa`; that compatibility alias does **not** make the two papers identical.

## 8. Prefill and Decode Must Be Analyzed Separately

### Prefill

Many queries are available together. Dense attention can use large matrix tiles efficiently. Sparse designs save FLOPs, but selection, ragged layouts, causal masks, backward support, and load balance decide whether theoretical savings become wall-clock gains.

### Decode

Each request normally contributes one query. Reading historical state often dominates arithmetic. Cache width, selected-entry count, paging locality, dequantization, and kernel launch overhead matter more than a prefill-only FLOP estimate.

For KDA/GDN, decode is a recurrent state update instead of a historical-cache scan. Its main concern becomes state-matrix bandwidth and small-batch kernel efficiency.

## 9. Serving-System Questions to Ask

For any new attention mechanism, answer these before choosing a backend:

1. What persistent state is stored per layer and per request?
2. Does state grow with tokens, blocks, or not at all?
3. Are prefill and decode mathematically identical but differently scheduled?
4. Does the kernel need dense, paged, ragged, top-k, or recurrent metadata?
5. Can a prefix cache share the state without recomputation?
6. How are speculative branches copied, verified, committed, or rolled back?
7. Which dimensions and sequence lengths are static under CUDA/NPU Graph capture?
8. Is quantization part of the trained architecture or only a serving optimization?
9. Does tensor/context parallelism partition heads, sequence entries, or recurrent state?
10. Is the reported speedup end-to-end, or only for the core attention kernel?

## 10. Ascend NPU Optimization View

Do not translate a CUDA kernel name directly into an NPU implementation plan. Decompose the dataflow:

| Stage | NPU-focused question |
|---|---|
| Projection | Can Q/K/V, latent, gate, and indexer projections use large Cube-friendly matmuls? |
| Compression | Can projection, gated pooling, normalization, RoPE, quantization, and cache store be fused? |
| Indexing | Can score computation and top-k avoid materializing a huge score tensor in HBM? |
| Sparse core | Are selected indices already in the physical paged-cache address space expected by the kernel? |
| Recurrent core | Can gate activation, state decay, delta update, and output read stay in on-chip memory? |
| Graph replay | Which lengths, page tables, selected indices, and state-slot ids must be refreshed in place? |

Always validate with a profiler. Lower algorithmic FLOPs do not guarantee lower latency if gathers are uncoalesced, top-k is launch-bound, or vector/Cube stages serialize.

## 11. Reading Route

1. Read [MHA/MQA/GQA shapes](./02-gqa-attention-shapes.md).
2. Read [MLA](./04-multi-head-latent-attention.md) for feature-axis cache compression.
3. Read [DSA](./07-deepseek-sparse-attention.md) for learned token selection over MLA entries.
4. Read [CSA/HCA](./08-compressed-sparse-attention.md) for sequence compression plus sparse/dense retrieval.
5. Read [KDA](./09-kimi-delta-attention.md) and the [GDN topic](../Gated_Delta_Network/) for recurrent linear attention.
6. Read [Attention Kernel](../Attention_Kernel/) for exact-attention execution optimizations.

## 12. References

- [Multi-Query Attention](https://arxiv.org/abs/1911.02150)
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
- [DeepSeek-V2 Technical Report (MLA)](https://arxiv.org/abs/2405.04434)
- [Native Sparse Attention](https://arxiv.org/abs/2502.11089)
- [DeepSeek-V3.2 (DSA)](https://arxiv.org/abs/2512.02556)
- [DeepSeek-V4 (CSA/HCA)](https://arxiv.org/abs/2606.19348)
- [Kimi Linear (KDA)](https://arxiv.org/abs/2510.26692)
- [FlashAttention](https://arxiv.org/abs/2205.14135)
