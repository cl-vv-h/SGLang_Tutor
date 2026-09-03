# DeepSeek Sparse Attention: Learned Indexing over MLA Cache

## 1. The Problem DSA Solves

MLA makes each historical cache entry much narrower, but dense MLA still lets every query attend to every previous entry:

```text
MLA: smaller entry x all historical positions
```

At very long context, the number of entries read and the core attention computation still grow with context length. DeepSeek Sparse Attention (DSA) adds a learned retrieval stage:

```text
cheap indexer scans history
  -> select top-k token entries
  -> expensive MLA core reads only selected entries
```

DSA therefore targets the **position axis**, while MLA primarily targets the **feature/cache-width axis**.

## 2. DSA Is Built under MLA

DeepSeek-V3.2 instantiates DSA on the MQA execution mode of MLA. Each historical token still produces one shared MLA latent KV entry, and all query heads reuse that entry.

```text
historical token s
  -> MLA latent entry c_s
  -> indexer key kI_s

query token t
  -> main MLA query heads
  -> lightweight indexer queries and head weights
```

The indexer and main attention serve different purposes and use different projections. Indexer scores are only used to choose positions; the main MLA logits and softmax still determine the output over those positions.

## 3. Unified Notation

| Symbol | Meaning |
|---|---|
| `L` | visible sequence length |
| `k` | number of entries selected per query, `k << L` |
| `HI` | number of indexer query heads |
| `DI` | indexer head dimension |
| `h_t` | hidden state of query token `t`, `[H]` |
| `qI_t,j` | indexer query of head `j`, `[DI]` |
| `kI_s` | shared indexer key of historical token `s`, `[DI]` |
| `wI_t,j` | scalar weight of indexer head `j` |
| `I_t,s` | scalar index score between query `t` and historical token `s` |
| `c_s` | MLA latent KV entry of token `s` |

## 4. Lightning Indexer

The indexer computes a scalar retrieval score:

```text
I_t,s = sum_(j=1..HI) wI_t,j * ReLU(qI_t,j dot kI_s)
```

Shape view for a packed batch with `Tq` query tokens and `Lkv` candidate entries:

```text
QI:       [Tq, HI, DI]
KI_cache: [Lkv, DI]
head_w:   [Tq, HI]

dot:      [Tq, Lkv, HI]
ReLU + weighted sum over HI
scores:   [Tq, Lkv]
```

ReLU is part of the trained scoring function, not an interchangeable implementation detail. The indexer uses fewer/smaller heads and can use low precision, making a full scan much cheaper than running the main attention on all entries.

## 5. Fine-Grained Top-k Selection

For query token `t`:

```text
S_t = TopK(I_t, :, k)
selected_cache_t = {c_s | s in S_t}
```

The main output is then:

```text
u_t = MLA_Attention(h_t, selected_cache_t)
```

Important consequences:

1. Selection is **query-dependent**; two query tokens may retrieve different positions.
2. The indexer collapses its heads into one scalar score per historical position.
3. All main query heads consume the selected shared MLA entries in MQA mode.
4. Top-k changes the support of softmax; it is not equivalent to computing dense softmax and dropping small weights afterward.

## 6. Complexity: Main Attention and Indexer Must Be Counted Separately

Ignoring constants:

| Component | Full-sequence prefill | One-token decode |
|---|---:|---:|
| Dense MLA core | `O(L^2 * Dmain)` | `O(L * Dmain)` |
| DSA indexer | `O(L^2 * HI * DI)` | `O(L * HI * DI)` |
| DSA sparse main core | `O(L * k * Dmain)` | `O(k * Dmain)` |

The indexer is still quadratic across a full sequence and linear in history for one decode token. DSA is efficient because the indexer constant is deliberately much smaller than the main MLA constant and because the expensive core uses only `k` entries.

The official DeepSeek-V3.2 training configuration selects 2048 entries per query. Treat that as a model configuration, not a universal DSA constant.

## 7. What Is Cached

DSA does not replace MLA cache with only top-k entries. Future queries may select any old position, so persistent history includes:

```text
main latent cache: one MLA entry c_s per token
indexer cache:     one indexer key kI_s per token
```

Top-k indices are per-forward metadata. They determine which entries the current query reads, but they are not the historical state itself.

Compared with dense MLA, DSA adds indexer-key storage while bounding the number of main entries read per query. Cache capacity still grows with context length.

## 8. Why DSA Needs Training

Arbitrary post-hoc top-k selection can remove information the model expects. DeepSeek-V3.2 uses two stages:

### 8.1 Dense Indexer Warm-up

- Keep the main attention dense.
- Freeze the main model and train the indexer.
- Build a target distribution by aggregating main-attention scores across heads and normalizing it.
- Minimize KL divergence between that target and the indexer distribution.

### 8.2 Sparse Continued Training

- Enable top-k selection.
- Train the main model to adapt to sparse visibility.
- Continue training the indexer with its auxiliary alignment loss on selected positions.
- Detach the indexer's input from the main computational graph so the language-model loss and indexer loss have separated optimization paths.

DSA is therefore a natively trained architecture, not merely a serving-time KV eviction heuristic.

## 9. Prefill Dataflow

For packed query tokens:

```text
hidden states
  -> build/store new MLA latent entries
  -> build/store new indexer keys
  -> indexer scores against causal history
  -> top-k per query, pad invalid slots
  -> convert logical token indices to cache addresses
  -> sparse MLA attention
  -> output projection
```

Practical concerns include causal masking, variable sequence lengths, top-k padding, selected-token load imbalance, and whether the sparse kernel can consume paged cache without materializing a dense gather.

When a visible sequence is no longer than `k`, selection can degenerate to all valid positions. A short-sequence dense or sequential-index path may be faster than launching the full sparse pipeline.

## 10. Decode Dataflow

For one new token per request:

```text
1. Project current hidden state into main MLA query and indexer query/weights.
2. Append current main latent entry and indexer key to their caches.
3. Scan valid indexer keys and produce one score per historical token.
4. Select k logical positions; pad unused slots with -1 when history < k.
5. Translate logical positions through the page table.
6. Run sparse MLA core on selected physical entries.
```

This separates a low-dimensional sequential scan from a high-dimensional bounded attention read.

## 11. Paged Cache and Graph-Replay Contracts

A production implementation cannot pass paper-level token ids directly to a paged kernel:

```text
logical top-k token index
  -> request page table
  -> physical cache index
  -> sparse attention metadata
```

For graph capture, output buffers often have a fixed `[graph_batch, k]` shape. Runtime sequence lengths, page tables, indexer cache lengths, and selected indices must be refreshed without changing captured tensor addresses. `-1` sentinels mark invalid top-k slots.

## 12. SGLang Source Map

| Source | Responsibility |
|---|---|
| [`python/sglang/srt/models/deepseek_v2.py`](../../../../python/sglang/srt/models/deepseek_v2.py) | Adds `Indexer` to MLA layers and threads top-k indices through the model |
| [`python/sglang/srt/layers/attention/dsa/dsa_indexer.py`](../../../../python/sglang/srt/layers/attention/dsa/dsa_indexer.py) | Indexer projections, key cache, score paths, and top-k production |
| [`python/sglang/srt/layers/attention/dsa/dsa_topk_backend.py`](../../../../python/sglang/srt/layers/attention/dsa/dsa_topk_backend.py) | Top-k implementation dispatch and fused transforms |
| [`python/sglang/srt/layers/attention/dsa/transform_index.py`](../../../../python/sglang/srt/layers/attention/dsa/transform_index.py) | Converts selected logical indices through paged-cache tables |
| [`python/sglang/srt/layers/attention/dsa_backend.py`](../../../../python/sglang/srt/layers/attention/dsa_backend.py) | Prefill/decode metadata and sparse MLA backend dispatch |
| [`python/sglang/srt/layers/attention/attention_registry.py`](../../../../python/sglang/srt/layers/attention/attention_registry.py) | Registers `dsa`; keeps `nsa` only as a deprecated backend alias |

The implementation has separate choices for index scoring, top-k, and sparse core attention. Tuning only one of them may simply move the bottleneck to another stage.

## 13. Ascend NPU Path in This Snapshot

The current source snapshot contains an explicit Ascend path:

- `dsa_indexer.py` uses `torch_npu.npu_rotary_mul` for indexer RoPE and `torch_npu.npu_lightning_indexer` for index selection.
- [`python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`](../../../../python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py) separates `forward_dsa_prepare_npu` from `forward_dsa_core_npu`.
- `AttnForwardMethod.DSA_NPU` gives DSA its own model-forward route instead of treating it as ordinary dense MLA.

For NPU performance analysis, measure at least four ranges independently:

```text
indexer projection/rope
index score + top-k
logical-to-physical index transform
sparse main attention + output projection
```

Also inspect synchronization between Vector-heavy top-k/address work and Cube-heavy projection/attention work. A fast core attention operator cannot hide an indexer or metadata pipeline that serializes before it.

## 14. DSA Versus NSA

| Dimension | DSA | Native Sparse Attention (NSA) |
|---|---|---|
| Main unit selected | fine-grained token-level MLA entries | fine-grained token blocks |
| Global coarse path | lightweight indexer scans token keys | explicit compressed-token attention branch |
| Local path | selected tokens; MLA foundation | dedicated sliding-window branch |
| Combination | one selected set feeds the main core | compression, selection, and window outputs are gated together |

Both are natively trainable sparse attention designs, but their dataflows and cache contracts are different.

## 15. Common Misconceptions

1. **“DSA makes all attention computation `O(L*k)`.”** The expensive core does; the lightweight indexer still scans all candidate history.
2. **“Only selected entries need to be cached.”** Future queries may select previously unselected entries, so full latent and indexer history must remain addressable.
3. **“Indexer score is the attention score.”** It is a retrieval score; the selected entries enter a separate MLA core and softmax.
4. **“DSA can be added to any dense checkpoint without training.”** The published architecture warms up the indexer and continues sparse training.
5. **“SGLang's `nsa` alias means DSA equals NSA.”** It is a deprecated configuration alias, not an algorithmic equivalence.

## 16. Correctness and Performance Checklist

1. Top-k candidates obey causal and per-request sequence boundaries.
2. Invalid slots use a sentinel that every downstream transform and kernel respects.
3. Logical indices are translated through the correct request page table.
4. Main latent cache and indexer key cache advance consistently.
5. Short histories use all valid entries rather than duplicated or future entries.
6. Quantized indexer ranking is compared against a higher-precision reference using top-k recall, not only logit error.
7. Sparse outputs are validated against the model's trained reference path, not assumed equal to dense MLA.
8. Profile indexer, top-k, transform, sparse core, and graph metadata separately.

## 17. References

- [DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models](https://arxiv.org/abs/2512.02556)
- [Official DeepSeek-V3.2-Exp repository](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp)
- [Native Sparse Attention](https://arxiv.org/abs/2502.11089)
- [MLA tutorial in this repository](./04-multi-head-latent-attention.md)
