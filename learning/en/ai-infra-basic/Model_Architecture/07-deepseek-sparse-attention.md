# DeepSeek Sparse Attention: Learned Indexing over MLA Cache

> 中文版：[DeepSeek Sparse Attention](../../../zh/ai-infra-basic/Model_Architecture/07-deepseek-sparse-attention.md)

## Reading Map: Two Scores, Not One

A DSA forward pass contains two completely different kinds of score:

- the cheap **indexer score** only answers “which positions should main attention inspect?”;
- the **main attention score** enters softmax and determines how selected entries produce model output.

```mermaid
flowchart LR
    X["Current hidden state<br/>[Tq,H]"] --> IQ["Indexer Q and weights<br/>QI [Tq,HI,DI]<br/>w [Tq,HI]"]
    X --> MQ["Main MLA query<br/>[Tq,N,...]"]
    IK[("Indexer-key cache<br/>[Lkv,DI]")] --> IS["Low-dimensional full scan<br/>[Tq,Lkv]"]
    IQ --> IS
    IS --> TOPK["Top-k logical positions<br/>idx [Tq,k]"]
    TOPK --> MAP["Page-table translation<br/>physical_idx [Tq,k]"]
    MAP --> GATHER["Sparse main-cache read<br/>selected Ckv [Tq,k,Dc]<br/>selected Kr [Tq,k,Dr]"]
    MC[("MLA main cache")] --> GATHER
    MQ --> MS["Main MLA logits<br/>[Tq,N,k]"]
    GATHER --> MS
    MS --> SM["Softmax over only k positions<br/>P [Tq,N,k]"]
    SM --> OUT["Weighted latent + output projection<br/>[Tq,H]"]
    GATHER --> OUT
```

DSA is therefore neither “top-k over MLA weights” nor “delete old KV.” It is a **learned candidate-set generator followed by fresh main attention over that set**.

### Running Small-Shape Example

To expose every axis, use:

```math
B=1,\qquad \mathrm{Tq}=2,\qquad \mathrm{Lkv}=8
```

```math
\mathrm{HI}=2,\qquad \mathrm{DI}=2,\qquad k=3
```

```math
N=4,\qquad \mathrm{Dc}=4,\qquad \mathrm{Dr}=2
```

| Tensor | Shape | Axis meaning |
|---|---:|---|
| `QI` | $`\left[2,2,2\right]`$ | query token × indexer head × index feature |
| `KI_cache` | $`\left[8,2\right]`$ | historical token × index feature; key shared by indexer heads |
| `head_w` | $`\left[2,2\right]`$ | query token × indexer head |
| `index_score` | $`\left[2,8\right]`$ | one scalar for every query/history pair |
| `topk_idx` | $`\left[2,3\right]`$ | three logical token IDs per query |
| `Ckv_selected` | $`\left[2,3,4\right]`$ | query × selected position × latent feature |
| `Krope_selected` | $`\left[2,3,2\right]`$ | query × selected position × positional feature |
| `main_logits` | $`\left[2,4,3\right]`$ | query × main head × selected position |
| `main_prob` | $`\left[2,4,3\right]`$ | softmax normalizes only the last $`k=3`$ positions |
| `latent_out` | $`\left[2,4,4\right]`$ | query × main head × latent feature |
| `output` | $`\left[2,H\right]`$ | merged, output-projected residual branch |

`topk_idx` has no main-head axis. After reduction across `HI`, each query obtains one shared position set; all $`N=4`$ main query heads compute their own logits over that same set.

### Numerical Walk from Logical Positions to Output

Suppose the first query produces:

```math
\mathrm{shape}\left(\mathrm{position}\right)=\left[0,1,2,3,4,5,6,7\right]
```

```text
score:    [0.2,  1.7, -0.1,  0.8,  2.0,  0.4,  1.1,  0.3]
```

$`\mathrm{TopK}\left(k=3\right)`$ returns logical positions $`\left[4,1,6\right]`$. Assume the block table maps these to physical cache slots $`\left[20,9,14\right]`$: **ranking produces logical token positions; the kernel reads physical slots**. For main head 0, the actual MLA logits might be:

```text
selected logical positions: [4,   1,   6]
main logits:                [0.3, 2.1,-0.4]
softmax:                    [0.133,0.802,0.066]
```

For corresponding latents

```math
c_{4}=\left[1,0,1,0\right],\qquad c_{1}=\left[0,2,0,0\right],\qquad c_{6}=\left[0,0,1,3\right]
```

the head first computes in latent space:

```math
z=0.133\cdot c_{4}+0.802\cdot c_{1}+0.066\cdot c_{6}=\left[0.133,1.604,0.199,0.198\right]\qquad\text{[Dc=4]}
```

Only then does the absorbed value/output transform run. The indexer scores `1.7, 2.0, 1.1` never enter main softmax; they only determine the set `{4,1,6}`.

### Three Paths: Prefill, Decode, and Training

```mermaid
flowchart TD
    TRAIN["Training warm-up"] --> TD["Keep main attention dense"]
    TD --> KL["Supervise indexer from main-attention distribution<br/>KL alignment"]
    KL --> TS["Sparse continued training<br/>top-k enters main path"]

    PREFILL["Inference prefill<br/>Tq=S"] --> PM["Causal index scores [S,S]"]
    PM --> PK["Different top-k set per position"]
    PK --> PC["Sparse main attention [S,N,k]"]

    DECODE["Inference decode<br/>Tq=1"] --> DS["Scan KI_cache [Lkv,DI]"]
    DS --> DK["Top-k + page translation"]
    DK --> DC["Read k main-cache entries"]
```

Training explains how the retriever learns; prefill/decode explain how the trained model executes. Combining all three into one pseudocode path hides the distinct contracts for the dense teacher, causal mask, paged addresses, and graph replay.

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

![DSA tensor dataflow](./assets/dsa-dataflow.svg)

## 3. Unified Notation

| Symbol | Meaning |
|---|---|
| `L` | visible sequence length |
| `k` | number of entries selected per query, `k << L` |
| `HI` | number of indexer query heads |
| `DI` | indexer head dimension |
| `h_t` | hidden state of query token `t`, $`\left[H\right]`$ |
| `qI_t,j` | indexer query of head `j`, $`\left[\mathrm{DI}\right]`$ |
| `kI_s` | shared indexer key of historical token `s`, $`\left[\mathrm{DI}\right]`$ |
| `wI_t,j` | scalar weight of indexer head `j` |
| `I_t,s` | scalar index score between query `t` and historical token `s` |
| `c_s` | MLA latent KV entry of token `s` |

## 4. Lightning Indexer

The indexer computes a scalar retrieval score:

```math
I_{t},s = \sum_{j=1..\mathrm{HI}} \mathrm{wI\_t},j \cdot  \mathrm{ReLU}\left(\mathrm{qI\_t},j \cdot \mathrm{kI\_s}\right)
```

Shape view for a packed batch with `Tq` query tokens and `Lkv` candidate entries:

```math
\mathrm{shape}\left(\mathrm{QI}\right)=\left[\mathrm{Tq},\mathrm{HI},\mathrm{DI}\right]
```

```math
\mathrm{shape}\left(\mathrm{KI\_cache}\right)=\left[\mathrm{Lkv},\mathrm{DI}\right]
```

```math
\mathrm{shape}\left(\mathrm{head\_w}\right)=\left[\mathrm{Tq},\mathrm{HI}\right]
```

```math
\mathrm{shape}\left(\mathrm{dot}\right)=\left[\mathrm{Tq},\mathrm{Lkv},\mathrm{HI}\right]
```

```text
ReLU + weighted sum over HI
```

```math
\mathrm{shape}\left(\mathrm{scores}\right)=\left[\mathrm{Tq},\mathrm{Lkv}\right]
```

### 4.1 Projection-by-Projection Shape Changes

The indexer receives the model hidden state and the low-rank MLA query activation. A representative projection path in the SGLang source is:

```text
hidden x:   [Tq,Hmodel]
```

```math
\mathrm{shape}\left(q_{\mathrm{lora}}\right)=\left[\mathrm{Tq},\mathrm{Dq}\right]
```

```math
\mathrm{QI\_flat}=q_{\mathrm{lora}}\,{\mathrm{WqI}}^{\mathsf T}
```

```math
\mathrm{shape}\left(\mathrm{WqI}\right)=\left[\mathrm{HI}\cdot \mathrm{DI},\mathrm{Dq}\right]
```

```math
\mathrm{shape}\left(\mathrm{QI\_flat}\right)=\left[\mathrm{Tq},\mathrm{HI}\cdot \mathrm{DI}\right]
```

```math
\mathrm{shape}\left(\mathrm{QI}\right)=\left[\mathrm{Tq},\mathrm{HI},\mathrm{DI}\right]
```

```math
\mathrm{KI}=\mathrm{LayerNorm}\left(x\,{\mathrm{WkI}}^{\mathsf T}\right)
```

```math
\mathrm{shape}\left(\mathrm{WkI}\right)=\left[\mathrm{DI},\mathrm{Hmodel}\right]
```

```text
KI:         [Tq,DI]             # one shared key per token
```

```math
\mathrm{head\_w}=x\,{\mathrm{WwI}}^{\mathsf T}
```

```math
\mathrm{shape}\left(\mathrm{WwI}\right)=\left[\mathrm{HI},\mathrm{Hmodel}\right]
```

```math
\mathrm{shape}\left(\mathrm{head\_w}\right)=\left[\mathrm{Tq},\mathrm{HI}\right]
```

The indexer query has `HI` heads, but its key is shared. Broadcasting the key across query heads gives:

```text
QI[:,None,:,:]:        [Tq,1,HI,DI]
KI_cache[None,:,:, :]: [1,Lkv,1,DI]
dot result:            [Tq,Lkv,HI]
```

After `ReLU`, multiply by `head_w[:,None,:] [Tq,1,HI]` and reduce over `HI`:

```math
\mathrm{shape}\left(\mathrm{weighted}\right)=\left[\mathrm{Tq},\mathrm{Lkv},\mathrm{HI}\right]
```

```math
\mathrm{shape}\left(\mathrm{scores}\right)=\left[\mathrm{Tq},\mathrm{Lkv}\right]
```

This head reduction is why top-k returns one position set per query token rather than a different set per main attention head.

### 4.2 Indexer RoPE Split

The indexer head can contain positional and non-positional subspaces:

```math
\mathrm{shape}\left(\mathrm{QI}\right)=\left[\mathrm{Tq},\mathrm{HI},\mathrm{DI}\right]
```

```text
-> QI_rope [Tq,HI,Dr]
-> QI_nope [Tq,HI,DI-Dr]
```

```math
\mathrm{shape}\left(\mathrm{KI}\right)=\left[\mathrm{Tq},\mathrm{DI}\right]
```

```text
-> KI_rope [Tq,Dr]
-> KI_nope [Tq,DI-Dr]
```

Apply RoPE only to the designated `Dr` channels and concatenate back to `DI`. In the NPU path visible in this repository, a representative shape is $`\mathrm{HI}=64`$, $`\mathrm{DI}=128`$, $`\mathrm{Dr}=64`$: the projection produces $`\left[\mathrm{Tq},64,128\right]`$, splits into two 64-channel parts, rotates the positional part, and restores $`\left[\mathrm{Tq},64,128\right]`$.

### 4.3 A Small Numerical Ranking Example

Let $`\mathrm{HI}=2`$, $`\mathrm{DI}=2`$, and four candidate keys:

```math
\mathrm{qI\_0} = \left[1, 0\right] \mathrm{head} \mathrm{weight} w_{0} = 0.75
```

```math
\mathrm{qI\_1} = \left[0, 1\right] \mathrm{head} \mathrm{weight} w_{1} = 0.25
```

```math
\mathrm{kI\_0}=\left[2,1\right]
```

```math
\mathrm{kI\_1}=\left[-1,4\right]
```

```math
\mathrm{kI\_2}=\left[1,-2\right]
```

```math
\mathrm{kI\_3}=\left[0,3\right]
```

Per-head dot products are:

```text
candidate 0: [2, 1]  -> ReLU -> [2,1]
candidate 1: [-1,4]  -> ReLU -> [0,4]
candidate 2: [1,-2]  -> ReLU -> [1,0]
candidate 3: [0, 3]  -> ReLU -> [0,3]
```

Weighted scalar scores become:

```math
I_{0}=0.75\cdot 2+0.25\cdot 1=1.75
```

```math
I_{1}=0.75\cdot 0+0.25\cdot 4=1.0
```

```math
I_{2}=0.75\cdot 1+0.25\cdot 0=0.75
```

```math
I_{3}=0.75\cdot 0+0.25\cdot 3=0.75
```

For $`k=2`$, candidates 0 and 1 are selected. Notice that negative evidence is removed by ReLU before the head-weighted reduction; moving ReLU after the reduction would define a different function.

ReLU is part of the trained scoring function, not an interchangeable implementation detail. The indexer uses fewer/smaller heads and can use low precision, making a full scan much cheaper than running the main attention on all entries.

## 5. Fine-Grained Top-k Selection

For query token `t`:

```math
S_{t} = \mathrm{TopK}\left(I_{t}, :, k\right)
```

```math
\mathrm{selected\_cache}_t=\left\{c_s\mid s\in S_t\right\}
```

The main output is then:

```math
u_{t} = \mathrm{MLA\_Attention}\left(h_{t}, \mathrm{selected\_cache\_t}\right)
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
| Dense MLA core | $`O\left({L}^{2}\cdot \mathrm{Dmain}\right)`$ | $`O\left(L\cdot \mathrm{Dmain}\right)`$ |
| DSA indexer | $`O\left({L}^{2}\cdot \mathrm{HI}\cdot \mathrm{DI}\right)`$ | $`O\left(L\cdot \mathrm{HI}\cdot \mathrm{DI}\right)`$ |
| DSA sparse main core | $`O\left(L\cdot k\cdot \mathrm{Dmain}\right)`$ | $`O\left(k\cdot \mathrm{Dmain}\right)`$ |

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

![DSA prefill and decode programs](./assets/dsa-prefill-decode.svg)

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

For graph capture, output buffers often have a fixed $`\left[\mathrm{graph\_batch},k\right]`$ shape. Runtime sequence lengths, page tables, indexer cache lengths, and selected indices must be refreshed without changing captured tensor addresses. `-1` sentinels mark invalid top-k slots.

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

1. **“DSA makes all attention computation $`O\left(L\cdot k\right)`$.”** The expensive core does; the lightweight indexer still scans all candidate history.
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

## 17. Full Tensor Ledger for One Forward

Let packed prefill contain `R` requests, `Tq` new query tokens in total, and `Tkv` visible cached/new KV entries across requests. `Lmax` is the largest per-request history.

| Stage | Tensor | Logical shape | Persistent? |
|---|---|---:|---|
| Input | hidden states | $`\left[\mathrm{Tq},\mathrm{Hmodel}\right]`$ | no |
| MLA query down projection | `q_lora` | $`\left[\mathrm{Tq},\mathrm{Dq}\right]`$ | no |
| Main query heads | `Qmain` | $`\left[\mathrm{Tq},\mathrm{Nq},\mathrm{Dmain}\right]`$ | no |
| New MLA latent | `Cnew` | $`\left[\mathrm{Tq},\mathrm{Dc}+\mathrm{Dr}\right]`$ | yes, append |
| Indexer query | `QI` | $`\left[\mathrm{Tq},\mathrm{HI},\mathrm{DI}\right]`$ | no |
| Indexer head weights | `head_w` | $`\left[\mathrm{Tq},\mathrm{HI}\right]`$ | no |
| New index key | `KInew` | $`\left[\mathrm{Tq},\mathrm{DI}\right]`$ | yes, append |
| Indexer history | `KIcache` | $`\left[\mathrm{Tkv},\mathrm{DI}\right]`$ plus request metadata | yes |
| Conceptual scores | `scores` | $`\left[\mathrm{Tq},\mathrm{Lmax}\right]`$ | workspace or fused away |
| Selected logical IDs | `topk_ids` | $`\left[\mathrm{Tq},k\right]`$ | forward metadata |
| Selected physical IDs | `page_ids` | $`\left[\mathrm{Tq},k\right]`$ | forward metadata |
| Selected latent entries | `Csel` | $`\left[\mathrm{Tq},k,\mathrm{Dc}+\mathrm{Dr}\right]`$ | often read directly, not materialized |
| Main logits | `Alogit` | $`\left[\mathrm{Tq},\mathrm{Nq},k\right]`$ | workspace or tiled |
| Main probabilities | `P` | $`\left[\mathrm{Tq},\mathrm{Nq},k\right]`$ | preferably online/tiled |
| Head output | `Ohead` | $`\left[\mathrm{Tq},\mathrm{Nq},\mathrm{Dv}\right]`$ | no |
| Layer output | `O` | $`\left[\mathrm{Tq},\mathrm{Hmodel}\right]`$ | no |

“Logical shape” does not imply that a dense tensor is allocated. A fused indexer can stream key tiles and keep only running top-k candidates; a sparse attention kernel can load latent entries from physical page IDs without constructing `Csel`.

## 18. Reference Pseudocode with Shape Assertions

The following pseudocode prioritizes semantics over performance:

```python
def dsa_reference(x, q_lora, latent_cache, index_cache, valid_mask, k):
    # x:             [Tq, Hmodel]
    # q_lora:        [Tq, Dq]
    # latent_cache:  [Tq, Lkv, Dlatent]  # conceptual per-query view
    # index_cache:   [Tq, Lkv, DI]       # conceptual per-query view
    # valid_mask:    [Tq, Lkv]

    Tq, Lkv, DI = index_cache.shape
    q_i = linear_q_index(q_lora).view(Tq, HI, DI)       # [Tq,HI,DI]
    w_i = linear_head_weight(x)                         # [Tq,HI]

    dots = einsum("thd,tld->tlh", q_i, index_cache)    # [Tq,Lkv,HI]
    score = (relu(dots) * w_i[:, None, :]).sum(-1)     # [Tq,Lkv]
    score = where(valid_mask, score, -inf)

    actual_k = min(k, Lkv)
    ids = topk(score, actual_k, dim=-1).indices         # [Tq,actual_k]
    chosen_is_valid = gather(valid_mask, dim=-1, index=ids)
    ids = where(chosen_is_valid, ids, -1)               # invalidate short rows
    ids = pad_to_k_with_minus_one(ids, k)               # [Tq,k]

    selected = batched_gather(latent_cache, ids)        # [Tq,k,Dlatent]
    out = sparse_mla(q_lora, selected, ids >= 0)        # [Tq,Nq,Dv]
    return out, ids
```

Production code replaces the conceptual per-query cache views with packed/paged pools, but its result must obey these request and causal boundaries.

## 19. Why Top-k Changes the Softmax Mathematics

Dense attention normalizes over all valid positions `V_t`:

```math
p_{\mathrm{dense}}\left(s\right) = \mathrm{exp}\left(a_{s}\right) / \sum_{u \in V_{t}} \mathrm{exp}\left(a_{u}\right)
```

DSA first obtains a selected support `S_t` from a different score function, then the main core normalizes only over that support:

```math
p_{\mathrm{dsa}}\left(s\right) = \mathrm{exp}\left(a_{s}\right) / \sum_{u \in S_{t}} \mathrm{exp}\left(a_{u}\right), s \in S_{t}
```

```math
p_{\mathrm{dsa}}\left(s\right) = 0, s \mathrm{not} \in S_{t}
```

Even if the main logits `a_s` are unchanged, removing candidates changes the denominator. This explains both why sparse continued training is needed and why comparing a sparse result to dense MLA with an extremely tight elementwise tolerance is not a valid correctness target. Kernel correctness should match the trained sparse reference; model quality should be evaluated at the task level.

## 20. Cache Bytes and Decode Traffic Example

Suppose one layer uses:

```math
\mathrm{Dc} + \mathrm{Dr} = 576 \mathrm{latent} \mathrm{elements} \mathrm{per} \mathrm{token}, \mathrm{BF16}
```

```math
\mathrm{DI} = 128 \mathrm{index} \mathrm{elements} \mathrm{per} \mathrm{token}, \mathrm{INT8} \mathrm{plus} \mathrm{block} \mathrm{scales}
```

```math
L = 128K \mathrm{tokens}
```

```math
k=2048
```

Ignoring alignment and scales:

```math
\text{latent capacity}= 131072 \cdot  576 \cdot  2 \mathrm{bytes} \approx  144 \mathrm{MiB}
```

```math
\text{index capacity}= 131072 \cdot  128 \cdot  1 \mathrm{byte} \approx  16 \mathrm{MiB}
```

The full capacity still grows with `L`. But the main core's per-query latent read is bounded near:

```math
2048 \cdot  576 \cdot  2 \mathrm{bytes} \approx  2.25 \mathrm{MiB}
```

The indexer scans roughly 16 MiB of quantized keys for that query unless its implementation reuses cache/on-chip locality or distributes the scan. This example makes the optimization target explicit: DSA trades a cheap, narrow sequential scan for a much smaller expensive latent-attention read.

## 21. Debugging from Symptoms

| Symptom | First tensors/invariants to inspect |
|---|---|
| Works for one request, fails for mixed lengths | `query_start_loc`, sequence lengths, causal/request mask |
| Works below `k`, fails above `k` | all-valid shortcut versus real top-k path |
| Correct eager mode, wrong graph replay | refreshed lengths/page tables/top-k buffers and stable addresses |
| Large quality loss after index quantization | top-k recall, tie handling, scale layout, RoPE ordering |
| Indexer is fast but end-to-end is slow | index transform, gather locality, sparse core, stream synchronization |
| Prefix cache gives nondeterministic output | main latent and indexer caches committed at different boundaries |
| NPU result differs only at long context | sentinel handling, page translation width, top-k ordering, accumulator precision |

## 22. Exercises

1. With $`\mathrm{Tq}=32`$, $`\mathrm{HI}=64`$, $`\mathrm{DI}=128`$, $`\mathrm{Lkv}=8192`$, write every intermediate shape in the unfused indexer.
2. Compute the ratio of main-core logits for dense MLA versus DSA when $`L=128K`$ and $`k=2048`$.
3. Explain why evicting every non-selected token after one query is incorrect.
4. Design a streaming top-k implementation that never materializes $`\left[\mathrm{Tq},\mathrm{Lkv}\right]`$ scores.
5. For a paged cache with page size 64, derive the physical address of logical token 130 from its request block table.

## 23. References

- [DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models](https://arxiv.org/abs/2512.02556)
- [Official DeepSeek-V3.2-Exp repository](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp)
- [Native Sparse Attention](https://arxiv.org/abs/2502.11089)
- [MLA tutorial in this repository](./04-multi-head-latent-attention.md)
