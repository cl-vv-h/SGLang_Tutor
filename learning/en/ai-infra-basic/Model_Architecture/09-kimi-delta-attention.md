# Kimi Delta Attention (KDA): Fine-Grained Gating for Linear Attention

Kimi Delta Attention (KDA) is the recurrent linear-attention component of Kimi Linear. It extends Gated DeltaNet (GDN) with a **per-key-channel forget gate**.

That one change is operationally important:

- dense and sparse softmax attention store history as a sequence cache;
- GDN and KDA fold history into a fixed-size recurrent state;
- KDA can forget different key-space features at different rates.

This chapter derives the recurrence, maps it to SGLang, and explains why prefill and decode need different kernels.

## 1. From Attention Over Tokens to a Recurrent State

Ordinary causal attention stores past keys and values:

```text
o_t = softmax(q_t K_{<=t}^T) V_{<=t}
```

Its cache grows with sequence length. Linear attention instead summarizes the past in a matrix. Using an implementation-friendly orientation:

```text
M_t in R[d_v, d_k]
o_t = M_t q_t
```

The state size depends on head dimensions, not on the number of processed tokens.

## 2. Why a Delta Rule?

A naive additive state keeps writing outer products:

```text
M_t = M_{t-1} + v_t outer k_t
```

This can overwrite poorly because a new key may collide with content already stored at a similar direction. The delta rule first asks what the state currently predicts for `k_t`, then writes only the residual:

```text
v_hat_t = M_{t-1} k_t
r_t = v_t - v_hat_t
M_t = M_{t-1} + beta_t * (r_t outer k_t)
```

`beta_t` controls the write strength.

## 3. GDN Versus KDA

GDN adds decay before the delta update, but its decay gate is a scalar for each token and head. KDA replaces it with a vector over key channels:

| Gate | GDN | KDA |
|---|---|---|
| Forget factor per token/head | scalar `alpha_t` | vector `alpha_t in R[d_k]` |
| State columns | decay together | decay independently |
| Recurrent state | fixed size | fixed size |

Fine-grained gating lets one feature remain stable while another is rapidly refreshed.

## 4. Shape Ledger

For one token and one head, let:

- `q_t, k_t in R[d_k]`;
- `v_t in R[d_v]`;
- `a_t in R[d_k]`: raw forget-gate activation;
- `alpha_t in (0, 1]^{d_k}`: per-channel retention;
- `beta_t in (0, 1)`: scalar delta write strength;
- `M_t in R[d_v, d_k]`: recurrent state;
- `o_t in R[d_v]`: output before normalization and output gating.

For a batch of tokens, the implementation normally lays tensors out as `[T, H, D]`; recurrent state is per request, layer, and KV head.

## 5. The KDA Recurrence

SGLang's storage-friendly state orientation yields the following steps:

```text
g_t       = -exp(A_log) * softplus(a_t + dt_bias)
alpha_t   = exp(g_t)
M_decay   = M_{t-1} * alpha_t[None, :]
v_hat_t   = M_decay k_t
r_t       = v_t - v_hat_t
M_t       = M_decay + beta_t * (r_t outer k_t)
o_t       = M_t (q_t / sqrt(d_k))
```

Because `g_t <= 0`, `alpha_t` is in `(0, 1]`. Each column of `M` receives a different retention factor.

The KDA report writes the transpose orientation `S_t in R[d_k, d_v]`:

```text
S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1}
      + beta_t k_t v_t^T
o_t = S_t^T q_t
```

The two forms are equivalent after transposing. Always establish the state orientation before comparing code and equations.

## 6. Projection, Short Convolution, and Output Gate

KDA is more than the recurrence alone. A typical layer performs:

```text
hidden states
  -> Q/K/V projections
  -> short causal convolution on projected streams
  -> Q/K normalization
  -> forget gate alpha and write gate beta
  -> recurrent KDA core
  -> normalization
  -> learned output gate
  -> output projection
```

The short convolution supplies precise local order information before recurrent compression. Consequently, serving state includes both the KDA matrix and the convolution tail.

## 7. Memory and Compute Complexity

For each layer and request, the persistent state is approximately:

```text
KDA state: H_kv * d_v * d_k elements
conv state: projected_channels * (kernel_size - 1) elements
```

It is independent of sequence length. Per-token recurrence work is `O(H_kv * d_v * d_k)` rather than a scan over all prior tokens.

This does not mean KDA is automatically fast. The state matrix can be large, and performance depends on keeping it on chip long enough, fusing elementwise gates with matrix/vector operations, and avoiding tiny launches.

## 8. Prefill: Chunkwise Parallel Training and Inference

Applying the recurrence one token at a time during a long prefill underutilizes accelerators. KDA therefore uses chunkwise algorithms that process blocks of tokens in parallel while carrying a boundary state between chunks.

A prefill kernel must preserve the exact causal recurrence:

```text
state_in -> chunk(tokens i ... i+C-1) -> outputs + state_out
```

Important tuning variables include chunk size, head dimension, accumulator precision, state traffic, and the cost of materializing gate products. Validate chunkwise output and final state against a token-by-token reference.

## 9. Decode: Fused Recurrent Update

Decode has one or a few new tokens per request. The efficient path keeps the recurrence fused:

```text
decay state -> predict -> residual -> rank-1 update -> read output
```

Splitting these into separate kernels repeatedly reads and writes the full state matrix. A fused kernel reduces high-bandwidth-memory traffic and launch overhead.

Continuous batching adds a state-routing problem: each decode row must load and update the state belonging to the correct request. Request compaction and slot reuse must update this mapping atomically.

## 10. Kimi Linear Is a Hybrid Architecture

Kimi Linear does not replace every attention layer with KDA. The published architecture uses a `3:1` pattern of KDA layers to Multi-Head Latent Attention (MLA) layers.

The two paths serve different purposes:

- KDA provides constant-size long-history state and linear-time scanning;
- occasional MLA layers retain an exact content-addressable path over cached sequence entries.

Therefore, end-to-end memory still grows with context length at MLA layers, but much more slowly than if every layer used softmax attention.

## 11. Serving-State Semantics

Paged KV cache intuition is insufficient for recurrent layers. A KDA request owns mutable state:

```text
per-layer matrix state M
per-layer convolution tail
sequence position and optional cache dtype metadata
```

This changes several serving features:

- **prefix caching** must snapshot or share a state produced at an exact token boundary;
- **speculative decoding** must checkpoint or recompute state when draft tokens are rejected;
- **request migration** transfers recurrent and convolution states, not KV pages alone;
- **beam search** duplicates state on branching and prevents accidental aliasing;
- **CUDA/NPU graph replay** requires stable state-buffer addresses and bounded batch shapes.

In-place updates improve performance but make rollback semantics explicit and unavoidable.

## 12. Tensor Parallelism

KDA heads can be partitioned across tensor-parallel ranks. Each rank owns its local Q/K/V projections and recurrent states. The output projection then follows the model's row/column-parallel contract.

The critical rules are:

1. state head count and gate head count use the same partition;
2. no request state is accidentally shared between ranks;
3. checkpoint head layout matches runtime sharding;
4. output reductions occur exactly once.

Because the recurrent update is head-local, its core normally needs no token-by-token all-reduce.

## 13. SGLang Source Map

In the source snapshot used by this tutorial:

- architecture configuration: [`kimi_linear.py`](../../../../python/sglang/srt/configs/kimi_linear.py);
- layer/model wiring: [`kimi_linear.py`](../../../../python/sglang/srt/models/kimi_linear.py);
- prefill/decode dispatch: [`kda_backend.py`](../../../../python/sglang/srt/layers/attention/linear/kda_backend.py);
- Triton recurrence: [`kda_triton.py`](../../../../python/sglang/srt/layers/attention/linear/kernels/kda_triton.py);
- FLA wrapper: [`kda.py`](../../../../python/sglang/srt/layers/attention/fla/kda.py);
- fused recurrent fallback: [`fused_recurrent.py`](../../../../python/sglang/srt/layers/attention/fla/fused_recurrent.py);
- optional CuTe DSL kernel: [`cutedsl_kda.py`](../../../../python/sglang/jit_kernel/cutedsl_kda.py);
- recurrent-state sizing: [`mamba_utils.py`](../../../../python/sglang/srt/configs/mamba_utils.py).

The model file is the best place to verify gate shapes, short-convolution state, KDA/MLA layer selection, and tensor-parallel ownership.

## 14. Ascend NPU Optimization View

The SGLang snapshot has an NPU-specific causal-convolution call, while the KDA core is routed through the available linear-attention kernel path. Backend support evolves, so validate the exact SGLang, `torch_npu`, CANN, and `sgl_kernel_npu` combination instead of inferring support from a class name.

For Ascend profiling, separate:

1. Q/K/V and gate projections;
2. causal convolution update;
3. normalization and gate activation;
4. KDA prefill or recurrent core;
5. output normalization, gate, and projection;
6. state gather/scatter under continuous batching.

High-value fusion targets are gate activation+decay, decay+delta update+readout, and convolution update+state writeback. Track state bytes read/written per generated token; kernel duration alone can hide a bandwidth-bound design.

## 15. KDA Versus Sparse and Compressed Attention

| Property | DSA | CSA | KDA |
|---|---|---|---|
| Historical representation | token-level latent cache | compressed sequence cache | fixed recurrent matrix |
| Access | content-selected top-k | compressed top-k | state read/update |
| Persistent size vs length | linear | reduced linear | constant |
| Exact retrieval of a past entry | selected latent entry | selected compressed entry | no explicit entry |
| Central systems problem | irregular gather | compression + heterogeneous caches | mutable state routing |

These mechanisms solve different bottlenecks and can coexist in hybrid model families.

## 16. Common Misconceptions

1. **“KDA is softmax attention with a smaller KV cache.”** It is a recurrent linear-attention update with no token-addressable KV history in KDA layers.
2. **“GDN and KDA use the same gate.”** GDN uses a head-wise scalar decay; KDA uses a per-key-channel vector.
3. **“Constant memory means constant compute for an entire prompt.”** Work is constant per token, so total prefill work remains linear in prompt length.
4. **“The state matrix is enough to resume a request.”** The short-convolution tail and position metadata are also required.
5. **“Speculative rejection can just decrement sequence length.”** In-place recurrent state must be rolled back or recomputed.
6. **“All layers in Kimi Linear are KDA.”** The architecture interleaves KDA with MLA.

## 17. Correctness and Performance Checklist

1. `alpha` is per key channel and is applied along the correct state axis.
2. Code and derivation use consistent `M[d_v, d_k]` or `S[d_k, d_v]` orientation.
3. Chunkwise prefill matches recurrent reference outputs and final state.
4. Decode state routing remains correct after request admission, eviction, and compaction.
5. Prefix reuse and speculative rollback include convolution state.
6. Accumulator precision is validated over very long sequences.
7. TP sharding matches head and gate layouts from the checkpoint.
8. Profiling reports state bandwidth, fusion boundaries, and graph replay coverage.

## 18. References

- [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)
- [Official MoonshotAI/Kimi-Linear repository](https://github.com/MoonshotAI/Kimi-Linear)
- [Gated DeltaNet tutorial in this repository](../Gated_Delta_Network/README.md)
- [MLA tutorial in this repository](./04-multi-head-latent-attention.md)
