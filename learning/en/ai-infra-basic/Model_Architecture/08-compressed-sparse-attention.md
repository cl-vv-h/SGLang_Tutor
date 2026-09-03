# Compressed Sparse Attention (CSA) and Hierarchical Compressed Attention (HCA)

Compressed Sparse Attention (CSA) is the long-context attention design introduced with DeepSeek-V4. Its key move is easy to state:

> Compress the sequence first, then retrieve a small number of compressed entries for the expensive attention core.

Hierarchical Compressed Attention (HCA) provides a much more aggressively compressed dense path. A local sliding-window branch preserves recent token detail. Across the model's layer schedule, these mechanisms cover three time scales:

- **SWA**: exact, uncompressed recent context;
- **CSA**: moderately compressed, content-selected context;
- **HCA**: heavily compressed, always-visible global context.

This chapter focuses on the execution and serving consequences of that hierarchy. Here, **CSA** specifically means DeepSeek-V4's Compressed Sparse Attention, not another use of the acronym.

## 1. Why Compress Before Selecting?

DeepSeek Sparse Attention (DSA) selects token-level latent KV entries. Its main attention core is sparse, but its lightweight indexer still scans the token history.

CSA changes the candidate space from `L` token entries to roughly `L / m` compressed entries, where `m` is the compression stride. If the core retrieves `k` compressed entries, then:

| Component | DSA | CSA |
|---|---:|---:|
| Indexer candidates per query | `L` | about `L / m` |
| Main-core entries per query | `k` | `k` compressed entries |
| Main long-range cache granularity | token | compressed chunk |

Compression is not free: each compressed entry summarizes multiple source positions, so the model must learn what information survives. The local and hierarchical paths compensate for different losses.

## 2. Notation and Shape Ledger

Let:

- `X in R[L, d_model]`: hidden states;
- `m`: CSA compression stride;
- `m'`: HCA compression stride, normally much larger than `m`;
- `Nc = ceil(L / m)`: number of CSA entries;
- `Nh = ceil(L / m')`: number of HCA entries;
- `C in R[Nc, d_c]`: CSA compressed KV entries;
- `K_I in R[Nc, d_i]`: compressed indexer keys;
- `k`: number of CSA entries selected per query;
- `w`: uncompressed sliding-window length.

Use different symbols for `K_I` and `k`: the former is a key tensor, while the latter is the top-k budget.

## 3. Overlapping Sequence Compression

CSA does not simply average every `m` tokens. It builds two projected content streams and two learned gating streams:

```text
C_a = X W_a^C       Z_a = X W_a^Z
C_b = X W_b^C       Z_b = X W_b^Z
```

For compressed slot `i`, the compressor consumes two adjacent source chunks:

```text
previous chunk: X[(i-1)m : i*m]
current chunk:  X[i*m : (i+1)m]
```

The `a` and `b` paths are arranged so the slot receives up to `2m` source positions. A softmax over their gate logits produces learned, feature-wise mixing weights. Conceptually:

```text
C_i = sum over j in the 2m source positions of softmax(Z_i)[j] * projected_content[j]
```

The exact implementation batches and reshapes these operations, but three properties define the contract:

1. output stride is `m`, so cache length is approximately `L / m`;
2. adjacent outputs overlap in source coverage;
3. partial chunks require persistent compressor state during incremental decoding.

The overlap softens chunk boundaries: evidence near the edge of one chunk can contribute through the neighboring path.

## 4. Compressed Indexer

CSA runs retrieval over compressed entries, not raw token positions. Its indexer follows the same high-level pattern as DSA:

```text
c_q = x_t W_DQ
q_I = c_q W_UQ^I
score(t, s) = sum_j w_j^I * ReLU(q^I_{t,j} dot k^I_s)
I_t = TopK_s(score(t, s), k)
```

The difference is the domain of `s`: it ranges over `Nc` compressed entries. Indexer keys therefore need their own compressed cache synchronized with the main compressed KV cache.

The selected IDs are **compressed-entry IDs**. They must not be interpreted as token IDs or ordinary paged-KV slots.

## 5. Sparse Main Core

The selected compressed entries feed a multi-query attention-style core. DeepSeek-V4 shares the compressed representation as key and value and uses grouped output projection to recover head-specific expressivity.

A simplified query path is:

```text
Q_t = x_t W^Q
selected = C[I_t]
A_t = softmax(Q_t selected^T + mask)
O_t = GroupedOutput(A_t selected)
```

The real implementation includes normalization, positional handling, head grouping, and fused kernels. Keep the conceptual boundaries clear:

- the **indexer** ranks candidates;
- top-k returns compressed logical IDs;
- the **main core** recomputes attention logits and softmax on those entries.

## 6. HCA: A Dense Global Safety Net

HCA compresses the sequence with a larger stride `m'` and performs dense attention over the resulting short sequence:

```text
X[0:L] -> non-overlapping heavy compression -> H[0:Nh]
query -> dense attention over H
```

Unlike CSA, HCA does not need a retrieval indexer. Its sequence is short enough to remain fully visible. HCA is useful for broad global signals, while CSA preserves more detail for selected distant regions.

## 7. The Local SWA Branch

Compression inevitably discards token-level detail. Sliding-window attention (SWA) keeps the most recent `w` tokens uncompressed:

```text
context(t) = [t-w+1, ..., t]
```

This branch handles exact local syntax, recent tool output, and near-field dependencies without asking a compressor to preserve every small distinction.

Each compressed-attention layer pairs its local path with the long-range branch configured for that layer: CSA or HCA. The model interleaves layer types; it does not necessarily run both CSA and HCA in every block. These are complementary trained paths, not interchangeable inference options.

## 8. Positional Information and Attention Sinks

Sequence compression makes positions subtler than in ordinary MHA. DeepSeek-V4 uses partial rotary position encoding (RoPE): only a designated slice of the representation carries rotary position. The output path applies the corresponding inverse rotation where required by the architecture.

This is an implementation contract, not a cosmetic transformation. Applying RoPE to all channels or omitting output-side handling changes the function.

The architecture also includes an attention sink so queries retain a stable fallback destination. Sparse and compressed kernels must preserve the sink's masking and normalization semantics.

## 9. Example Configuration, Not a Universal Constant

The published DeepSeek-V4 Flash configuration uses the following illustrative values:

| Parameter | Example value |
|---|---:|
| CSA compression stride `m` | 4 |
| CSA top-k `k` | 512 |
| HCA compression stride `m'` | 128 |
| Local window `w` | 128 |

DeepSeek-V4 Pro increases the CSA selection budget to 1024. These are checkpoint architecture fields. Do not silently tune them as if they were only serving parameters.

## 10. Complexity and Memory

For a decode query with history length `L`, the approximate work of each configured layer type is:

```text
CSA layer: O(L / m) indexer + O(k) main core + O(w) local path
HCA layer: O(L / m') dense core + O(w) local path
```

Across the model, the cache contains several representations; a layer owns the subset required by its configured type:

```text
CSA main cache        ~ O(L / m)
CSA indexer-key cache ~ O(L / m)
HCA cache             ~ O(L / m')
local uncompressed KV ~ O(w) active entries per sequence
compressor tail state ~ O(m + m')
```

Bytes, not only entry counts, determine the real benefit. Record dtype, feature width, scale/metadata overhead, alignment, and page fragmentation for each pool.

## 11. Prefill and Decode Are Different Programs

### Prefill

Prefill can compress chunks in parallel and batch indexer queries. Its risks are temporary tensors, top-k workspace, causal correctness inside the last partial chunk, and irregular gathers into the sparse core.

### Decode

Decode appends one token at a time. Most steps only update an incomplete compression chunk. A completed chunk emits a new compressed cache entry. The runtime must therefore carry the partial content and gate state between steps.

Graph capture must use bounded buffers for:

- compressed cache addresses;
- indexer top-k outputs;
- partial compressor state;
- local-window metadata;
- per-request sequence and compressed lengths.

Reallocation or shape-dependent host work can erase graph-mode gains.

## 12. Heterogeneous Cache Management

CSA, HCA, and SWA advance at different rates. Treating them as one ordinary KV pool creates correctness and fragmentation problems.

A robust request state records:

```text
token_length
csa_complete_length and csa_tail_state
hca_complete_length and hca_tail_state
local_window mapping
csa indexer/main-cache alignment
```

Page and block sizes should respect compressor strides. If physical allocation units do not align naturally, define explicit carry-over rules rather than rounding away incomplete data.

Prefix caching must snapshot every representation at a mutually consistent boundary. Copying only completed compressed entries while losing the compressor tail changes future outputs.

## 13. SGLang Source Map

In the source snapshot used by this tutorial, the relevant paths are:

- model wiring: [`deepseek_v4.py`](../../../../python/sglang/srt/models/deepseek_v4.py);
- architecture fields: [`deepseek_v4.py`](../../../../python/sglang/srt/configs/deepseek_v4.py);
- compression: [`compressor.py`](../../../../python/sglang/srt/layers/attention/dsv4/compressor.py);
- compressed retrieval: [`indexer.py`](../../../../python/sglang/srt/layers/attention/dsv4/indexer.py);
- attention dispatch: [`deepseek_v4_backend.py`](../../../../python/sglang/srt/layers/attention/deepseek_v4_backend.py);
- cache pools: [`deepseek_v4_memory_pool.py`](../../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py);
- partial-chunk state: [`deepseek_v4_compress_state.py`](../../../../python/sglang/srt/mem_cache/deepseek_v4_compress_state.py).

Read them in that order: configuration establishes the shapes, the model establishes dataflow, and the backend/cache files establish the serving contract.

## 14. Ascend NPU Optimization View

The checked source's specialized DeepSeek-V4 path is centered on CUDA/HIP/Triton. The existence of generic PyTorch classes does not prove that an end-to-end Ascend path is production-ready. Verify the pinned SGLang, `torch_npu`, CANN, and kernel versions before deployment.

When bringing the architecture to Ascend, profile these regions separately:

1. projection plus overlapping compression;
2. compressed indexer and top-k;
3. logical-to-physical compressed-cache translation;
4. sparse gather and main attention;
5. HCA dense attention;
6. local SWA and final branch combination.

Good fusion candidates include projection+norm, compressor gate+softmax+reduce, index score+mask, and gather+attention. Preserve tail-state and causal semantics before optimizing launch count.

## 15. DSA, CSA, and HCA Side by Side

| Property | DSA | CSA | HCA |
|---|---|---|---|
| Stored long-range unit | token-level latent entry | moderately compressed chunk | heavily compressed chunk |
| Retrieval | top-k | top-k | none; dense over short sequence |
| Candidate count | `L` | `L / m` | `L / m'` |
| Main query work | `k` | `k` | `L / m'` |
| Primary role | selected fine detail | selected mid-resolution history | coarse global coverage |

## 16. Common Misconceptions

1. **“CSA is just KV quantization.”** It learns sequence compression; quantization changes numeric representation.
2. **“One compressed entry corresponds to exactly one non-overlapping chunk.”** CSA uses overlapping source coverage.
3. **“Top-k returns token indices.”** It returns compressed-entry indices.
4. **“HCA is another sparse branch.”** HCA is dense attention over a heavily compressed sequence.
5. **“Completed compressed caches are enough for prefix reuse.”** Partial compressor state is also part of request state.
6. **“The example ratios can be freely changed at runtime.”** They are coupled to checkpoint weights and architecture.

## 17. Correctness and Performance Checklist

1. Compressor boundary and padding behavior match the checkpoint implementation.
2. CSA main entries and indexer keys are emitted on the same logical step.
3. Compressed top-k IDs map through the correct request-specific cache table.
4. Partial chunks survive decode, prefix caching, migration, and speculative rollback.
5. SWA boundaries are causal and request-isolated.
6. Partial RoPE, inverse rotation, and sink semantics match the reference.
7. Memory accounting includes all cache pools, scales, metadata, and fragmentation.
8. Profiling separates compression, retrieval, sparse core, HCA, SWA, and graph overhead.

## 18. References

- [DeepSeek-V4: Advancing Open-Source Intelligence](https://arxiv.org/abs/2606.19348)
- [DSA tutorial in this repository](./07-deepseek-sparse-attention.md)
- [Efficient-attention landscape](./06-efficient-attention-landscape.md)
