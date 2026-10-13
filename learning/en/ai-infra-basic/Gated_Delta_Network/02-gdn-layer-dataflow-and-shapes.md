# 02. GDN Layer Dataflow and Shapes

**English** | [简体中文](../../../zh/ai-infra-basic/Gated_Delta_Network/02-gdn-layer-dataflow-and-shapes.md)

## 1. Notation

This lecture uses the following symbols to describe a GDN layer. They correspond to fields used by Qwen3-Next and the SGLang implementation.

| Symbol | Code Field | Meaning |
|---|---|---|
| `T` | `seq_len` or packed token count | total number of real tokens in the current forward |
| `B` | `batch_size` | number of active requests during decode |
| `H_model` | `hidden_size` | model hidden width |
| `H` | `num_k_heads` | number of GDN key/query heads |
| `HV` | `num_v_heads` | number of GDN value heads |
| `K` | `head_k_dim` | dimension of each q/k head |
| `V` | `head_v_dim` | dimension of each value head |
| `C` | `conv_kernel_size` | causal conv1d window size |
| `R` | `HV / H` | number of value heads mapped to each key head |

Unless stated otherwise, shapes are logical shapes before tensor parallel (TP) sharding. This grouping requires `HV` to be an integer multiple of `H`; tensors on a device use that TP rank's local head counts.

A common Qwen3-Next-style configuration is:

```text
H_model = 2048
H = 16
HV = 32
K = 128
V = 128
C = 4
R = 2
```

Therefore:

```text
key_dim   = H * K  = 16 * 128 = 2048
value_dim = HV * V = 32 * 128 = 4096
```

## 2. Layer Overview

![GDN layer dataflow](./assets/gdn-layer-dataflow.svg)

A GDN layer starts from backbone hidden states:

```text
hidden_states: [T, H_model]
```

Then the computation splits into several paths:

```text
hidden_states
  -> in_proj_qkvz -> q, k, v, z
  -> in_proj_ba   -> b, a

q/k/v -> causal conv1d -> GDN recurrent/chunk kernel -> core_attn_out
z     -> output gate / RMSNormGated
RMSNormGated(core_attn_out, z) -> out_proj -> layer output
```

## 3. Input Projection: `in_proj_qkvz`

In SGLang, `Qwen3GatedDeltaNet` creates:

```text
in_proj_qkvz:
    input_size  = hidden_size
    output part = [key_dim, key_dim, value_dim, value_dim]
```

So:

```text
projected_states_qkvz: [T, 2*key_dim + 2*value_dim]
```

Substituting the common configuration:

```text
2*key_dim + 2*value_dim
= 2*2048 + 2*4096
= 12288
```

The four projected segments mean:

| Segment | Logical Shape | Meaning |
|---|---:|---|
| `q` | `[T,H,K]` | query |
| `k` | `[T,H,K]` | key |
| `v` | `[T,HV,V]` | value |
| `z` | `[T,HV,V]` | gate input for output gate/norm |

`z` does not enter the recurrent state update. It modulates the GDN output at the output stage.

### 3.1 What Is `z`, and How Does It Differ from `g/beta`?

`z` is an **output-gate activation** projected from the current layer input `x_t`. Logically, `z_t = W_z x_t`; the implementation fuses `W_z` with the q/k/v projections in `in_proj_qkvz`. `W_z` is a trained parameter, whereas `z_t` is a token-dependent intermediate tensor, not another KV cache or the recurrent state.

For the default Qwen3-Next SiLU/Swish output gate described here, the code sets `norm_before_gate=True`. For each token and value head, RMSNorm operates along `V`, followed by elementwise multiplication:

```text
o = core_attn_out[t,h,:]                         # [V]
z = z[t,h,:]                                    # [V]
rms = sqrt(mean(o * o) + eps)                    # scalar
normed[t,h,:] = (weight * o / rms) * silu(z)      # [V]
silu(z) = z * sigmoid(z)
```

Here `weight` is a trainable norm weight of length `V`, and `*` means elementwise multiplication. Thus `z` has the same shape as the output: `[T,HV,V]`. A SiLU gate is not a probability in `[0,1]`: it can exceed 1 or be negative, so it is neither a sigmoid gate nor a binary switch. If a model explicitly configures another `output_gate_type`, follow that configuration.

| Control Signal | Shape per Token | What It Controls | Directly Updates This Layer's Recurrent State? |
|---|---|---|---|
| `exp(g)` | `[HV]` | how much old state survives | yes |
| `beta` | `[HV]` | how much prediction residual is written | yes |
| `silu(z)` | `[HV,V]` | how much of each retrieved value channel reaches subsequent computation | no |

This separates retaining information from exposing it for the current token. For example, `z=0` zeros that output channel without clearing the corresponding memory in this layer's state; later tokens can still retrieve it. The operation is not `core_attn_out + z`: it is **multiplicative gating after normalization**, followed by `out_proj`.

## 4. Input Projection: `in_proj_ba`

`in_proj_ba` produces two sets of gate activations:

```text
in_proj_ba:
    input_size  = hidden_size
    output part = [HV, HV]
```

Therefore:

```text
projected_states_ba: [T, 2*HV]
b: [T,HV]
a: [T,HV]
```

| Activation | Later Computation | Meaning |
|---|---|---|
| `a` | `g = -exp(A_log) * softplus(a + dt_bias)` | controls state forgetting |
| `b` | `beta = sigmoid(b)` | controls residual write strength |

`a` and `b` are token activations, not parameters. `in_proj_ba.weight` is the parameter.

## 5. Packed Checkpoint and Split/Reshape

For efficient loading and execution, checkpoint weights for GDN projection can be packed. SGLang's `load_weights()` maps:

```text
in_proj_qkv -> first three segments of in_proj_qkvz
in_proj_z   -> fourth segment of in_proj_qkvz
in_proj_b   -> first segment of in_proj_ba
in_proj_a   -> second segment of in_proj_ba
```

After projection, split, and reshape, the layer gets:

```text
mixed_qkv: [T, 2*key_dim + value_dim]
z:         [T, HV, V]
b:         [T, HV]
a:         [T, HV]
```

With the common configuration:

```text
mixed_qkv: [T, 8192]
z:         [T, 32, 128]
b/a:       [T, 32]
```

`mixed_qkv` is the flat concatenation of q/k/v:

```text
mixed_qkv = concat(flatten(q), flatten(k), flatten(v))
```

## 6. Causal Conv1d

GDN applies a short causal convolution on the q/k/v path:

```text
conv_dim = 2*key_dim + value_dim
conv_kernel_size = C
```

For packed q/k/v:

```text
mixed_qkv_before_conv: [T, conv_dim]
mixed_qkv_after_conv:  [T, conv_dim]
```

During decode, each request usually provides only one new token, so convolution needs a window state:

```text
conv_states: [num_slots, conv_dim, C-1] or an equivalent backend-specific layout
```

During prefill, causal conv can be applied over the whole sequence. During decode, `causal_conv1d_update` updates each request's conv state incrementally.

### 6.1 What Does It Compute, and Why Is It Causal?

This is a **depthwise short convolution** along the token axis: each q/k/v channel has its own length-`C` kernel, with no cross-channel mixing in this operation. The preceding linear projection already mixes hidden channels. Let `u` denote `mixed_qkv` before convolution and `c` denote one channel, with convolution weights indexed by lag:

```text
y[t,c] = silu(sum(w[c,j] * u[t-j,c] for j in range(C)))

C = 4:
y[t,c] = silu(w[c,0]*u[t,c]   + w[c,1]*u[t-1,c]
            + w[c,2]*u[t-2,c] + w[c,3]*u[t-3,c])
```

This describes the default SiLU path without bias used here; the kernel determines the stored weight order. Only the current position and the previous `C-1` positions are accessed, never future tokens, preserving autoregressive causality. Missing history at the start of a new request is zero; a continuing request reads its own conv state. Packed batches must respect request boundaries: the end of one request cannot become the history of the next.

Convolution and SiLU preserve `[T,conv_dim]` while changing the features at each position. q/k also undergo L2 normalization afterward, potentially fused into the GDN kernel; this is distinct from output RMSNorm. `z/a/b` do not pass through this short convolution.

### 6.2 Why Add It: Local Order Features and Long-Term Associative Memory

Without convolution, this layer's input projection operates independently at each token. With convolution, the q/k/v used for state reads and writes explicitly depend on several recent positions of this layer's input. The two operations provide complementary capabilities:

- **Local positional modeling**: separate weights for different lags can distinguish the preceding token from one three positions back. For example, when the window covers “not good,” the features at “good” can include the preceding negation. This illustrates representational capability, not a claim that a particular learned kernel implements this rule.
- **Long-term content association**: GDN stores and updates key→value associations in a fixed-size matrix. Short convolution first builds queries, write addresses, and content with local context, reducing the burden on that matrix memory to model local patterns on its own.
- **Modest additional cost**: for fixed `C`, convolution costs `O(T * conv_dim * C)`, and decode retains only the previous `C-1` inputs per channel, independent of context length. This is additional work; modeling quality is the main reason to include it.

This introduces an inductive bias for locality and relative order. Even without short convolution, GDN's sequential recurrence is order-sensitive, and `hidden_states` may already contain context from preceding layers. It is incorrect to claim that removing convolution removes all awareness of token order. Convolution is also distinct from RoPE and is not mathematically required for the delta rule to work.

### 6.3 Scientific Evidence: Component Ablations Beyond the Mechanism

[Gated Delta Networks, §3.3 and Appendix A.2, Table S.1](https://arxiv.org/html/2412.06464v1#A1.SS2), describes the architecture and tests component removal. These models have 400M parameters and were trained for 15B tokens on the same FineWeb-Edu subset:

| Setting | Avg-PPL ↓ | Avg-Acc ↑ |
|---|---:|---:|
| Full Gated DeltaNet (head dim 128) | 27.35 | 47.26 |
| Without short convolution | 28.95 | 46.16 |
| Without output gate | 29.12 | 45.46 |

This supports a quality benefit from short convolution and the output gate in that experimental setting. It does not prove a universal benefit across all scales and datasets, and it is not a Qwen3-Next ablation. The preceding section explains plausible mechanisms from the computation; these ablations do not independently establish every proposed explanation.

## 7. Split into q/k/v After Convolution

After convolution, `mixed_qkv` is split again:

```text
query: [1,T,H,K]
key:   [1,T,H,K]
value: [1,T,HV,V]
```

Why is the first dimension `1`? In online serving, requests with different lengths are often packed into one token dimension:

```text
query_start_loc: [N+1]
```

This array records the start/end offsets of each request inside the packed token sequence. In this representation, the batch dimension is often written as `1`; the real request boundaries are described by `cu_seqlens` or `query_start_loc`.

## 8. Computing `g` and `beta`

For prefill/extend, SGLang calls fused gating:

```text
g, beta = fused_gdn_gating(A_log, a, b, dt_bias)
```

Logical shapes:

```text
a:      [T,HV]
b:      [T,HV]
A_log:  [HV]
dt_bias:[HV]

g:      [1,T,HV]
beta:   [1,T,HV]
```

Formula:

```text
g[t,h] = -exp(A_log[h]) * softplus(a[t,h] + dt_bias[h])
beta[t,h] = sigmoid(b[t,h])
```

The packed decode fast path fuses this computation into the recurrent kernel so it does not need to separately materialize `q/k/v/g/beta`.

## 9. Recurrent State Shape

In serving, GDN state is stored in the same kind of runtime state pool as Mamba state. Logically:

```text
ssm_states: [num_slots, HV, V, K]
```

| Dimension | Meaning |
|---|---|
| `num_slots` | number of request state slots, managed by the request pool |
| `HV` | number of value heads |
| `V` | value head dimension |
| `K` | key/query head dimension |

For one request and one layer, state size is:

```text
HV * V * K
```

With the common configuration:

```text
32 * 128 * 128 = 524288 elements
```

If stored in BF16:

```text
524288 * 2 bytes = 1 MiB per linear-attention layer per request slot
```

This is a calculation for a specified dtype. In this repository, `mamba2_state_dtype()` defaults the recurrent state to FP32, giving **2 MiB**; configuration or an environment variable can override it. These numbers cover recurrent state only, excluding conv state, extra cache slots, and temporary buffers. This is not tiny, but it does not grow with the number of historical tokens.

### 9.1 Why Not Multiply by the q/k Head Count `H` Again?

Each value head maps to **one** q/k head, rather than pairing with all `H` q/k heads. For value head `h_v`, its q/k head is:

```text
R = HV // H
h_k = h_v // R

H = 16, HV = 32, R = 2:
q/k head 0  -> value heads 0, 1   -> S[0], S[1]
q/k head 1  -> value heads 2, 3   -> S[2], S[3]
...
q/k head 15 -> value heads 30, 31 -> S[30], S[31]
```

Value heads in a group share q/k vectors but have separate `v`, `g`, `beta`, and state matrices. For example, both `S[0]` and `S[1]` use head 0's k for writes and q for reads, but their states need not match. This grouping is commonly called grouped-value attention; do not directly reuse the head relationships of softmax GQA, where multiple query heads share K/V heads.

Adding head indices to the single-head equations from Chapter 01 makes the shapes explicit. Here q/k are already normalized, and the read scale is shown separately:

```text
q = query[t,h_k,:]                         # [K]
k = key[t,h_k,:]                           # [K]
v = value[t,h_v,:]                         # [V]
S_decay = exp(g[t,h_v]) * S_prev[h_v]       # [V,K]
r = v - S_decay @ k                        # [V]
S_new[h_v] = S_decay + beta[t,h_v] * r[:,None] * k[None,:]
o[t,h_v,:] = S_new[h_v] @ (q * scale)       # [V]
```

One state matrix maps the `K`-dimensional key/query space to the `V`-dimensional value space, giving `[V,K]`. There are `HV` such matrices, giving `[HV,V,K]` overall. `q` only reads the state, so no separate query state is needed. The q/k feature dimension is represented by `K`, while the q/k head count is represented through grouping:

```text
[HV,V,K] == reshape([H,R,V,K])
HV * V * K = H * R * V * K
```

Thus `H` is not missing: it is already counted through `HV = H * R`. Multiplying by `H` again would allocate state for nonexistent all-to-all q/k-head × value-head pairs. Even when `H=HV`, there are only `H` state matrices, not `H*H`.

### 9.2 Checking the Kernel and Memory Under TP

The repository's [packed recurrent decode kernel](../../../../python/sglang/srt/layers/attention/fla/fused_recurrent.py) directly uses:

```python
i_h = i_hv // (HV // H)
```

It addresses q/k using `i_h` and v/state using `i_hv`; the state offset for a head is `i_hv * V * K`. This implements the grouping above without requiring q/k to be physically replicated first.

With conventional attention TP sharding and divisible head counts, TP degree `P` gives each rank a state of `[num_slots,HV/P,V,K]`. Per-request, per-layer bytes on each device are `(HV/P)*V*K*state_element_bytes`; the sum across ranks remains `HV*V*K*state_element_bytes`. Head padding or special backend layouts require inspecting the actual allocation.

Also, **convolution state still includes `H` directly**: its logical element count is `(2*H*K + HV*V)*(C-1)`. It retains a short window of q/k/v projections, whereas recurrent state retains key→value matrix memory. These are different caches.

## 10. GDN Kernel Output

The GDN recurrent/chunk kernel returns:

```text
core_attn_out: [1,T,HV,V]
```

Then it is reshaped:

```text
core_attn_out: [T, HV, V]
z:             [T, HV, V]
```

The two tensors enter gated normalization:

```text
normed = RMSNormGated(core_attn_out, z)
normed: [T, HV, V]
```

Then flatten:

```text
normed_flat: [T, HV*V] = [T, value_dim]
```

Finally the output projection maps it back to model hidden size:

```text
out_proj: [value_dim -> hidden_size]
output:   [T, H_model]
```

This output returns to the decoder layer's residual path and then to the MLP/MoE path.

## 11. End-to-End Decode Flow

During decode, each active request usually contributes one new token:

```text
hidden_states: [B,H_model]
```

Flow:

```text
1. in_proj_qkvz:
       [B,H_model] -> [B,2*key_dim+2*value_dim]

2. in_proj_ba:
       [B,H_model] -> [B,2*HV]

3. split:
       mixed_qkv: [B,2*key_dim+value_dim]
       z:         [B,HV,V]
       a,b:       [B,HV]

4. causal_conv1d_update:
       mixed_qkv + conv_states -> mixed_qkv_after_conv [B,conv_dim]
       conv_states is updated in place

5. packed recurrent kernel:
       read ssm_states[cache_indices]
       compute q/k/v/g/beta
       update ssm_states[cache_indices]
       output core_attn_out [1,B,HV,V]

6. RMSNormGated(core_attn_out, z):
       [B,HV,V]

7. out_proj:
       [B,HV*V] -> [B,H_model]
```

## 12. End-to-End Prefill/Extend Flow

Prefill/extend processes multiple tokens at once:

```text
hidden_states: [T,H_model]
query_start_loc: [N+1]
```

Flow:

```text
1. Project into mixed_qkv/z/a/b
2. Run causal conv1d on packed q/k/v
3. Split into q [1,T,H,K], k [1,T,H,K], v [1,T,HV,V]
4. Compute g [1,T,HV] and beta [1,T,HV]
5. Run chunk_gated_delta_rule over the sequence in parallel
6. Write each request's final recurrent state back
7. Produce output [1,T,HV,V]
8. Run gated norm + out projection -> [T,H_model]
```

Prefill should not run one sequential kernel per token like decode, because long prompts would be slow. Instead, it uses a chunk-parallel algorithm: split the sequence into chunks, solve triangular dependencies inside each chunk, and merge states across chunks.

## 13. Special Case: Target Verify

In speculative decoding target verify, one request may carry multiple draft tokens. GDN must temporarily advance the state along candidate paths, but it cannot commit all draft tokens into the official state immediately.

SGLang's strategy is:

```text
official ssm_states:
    state after already committed tokens

intermediate_states_buffer:
    temporary states for draft tree / verify process

disable_state_update=True:
    do not directly commit final state during verify
```

After accept/reject is resolved, the correct state along the accepted path is committed. This mirrors the idea of temporary KV management in speculative decoding, except the object being managed is recurrent state rather than K/V pages.

## 14. Shape Summary

| Stage | Tensor | Shape |
|---|---|---:|
| input | `hidden_states` | `[T,H_model]` |
| qkvz projection | `projected_states_qkvz` | `[T,2*H*K+2*HV*V]` |
| ba projection | `projected_states_ba` | `[T,2*HV]` |
| after split | `mixed_qkv` | `[T,2*H*K+HV*V]` |
| after split | `z` | `[T,HV,V]` |
| after split | `a,b` | `[T,HV]` |
| after conv | `mixed_qkv` | `[T,2*H*K+HV*V]` |
| recurrent input | `q,k` | `[1,T,H,K]` |
| recurrent input | `v` | `[1,T,HV,V]` |
| gating | `g,beta` | `[1,T,HV]` |
| state pool | `ssm_states` | `[num_slots,HV,V,K]` |
| recurrent output | `core_attn_out` | `[1,T,HV,V]` |
| after gate/norm | `normed` | `[T,HV,V]` |
| flatten | `normed_flat` | `[T,HV*V]` |
| output projection | `output` | `[T,H_model]` |

## 15. Summary

1. A GDN layer projects hidden states into six activations: `q/k/v/z/a/b`.
2. `q/k/v` pass through a short causal convolution before entering the recurrent/chunk kernel.
3. `a/b` produce `g/beta`, controlling state forgetting and residual write strength.
4. `z` is not written into the state; it modulates `core_attn_out` at the output stage.
5. GDN state is `[num_slots,HV,V,K]`, stored by request slot, and does not grow with token count.

## 16. Source and Paper References

- [Qwen3GatedDeltaNet](../../../../python/sglang/srt/models/qwen3_next.py): projections, convolution weights, `norm_before_gate=True`, and the output path.
- [Gated RMSNorm](../../../../python/sglang/srt/layers/attention/fla/layernorm_gated.py): the ordering of RMSNorm and the SiLU/Swish output gate.
- [GDN backend](../../../../python/sglang/srt/layers/attention/linear/gdn_backend.py): causal convolution per request and prefill/decode dispatch.
- [Fused recurrent kernel](../../../../python/sglang/srt/layers/attention/fla/fused_recurrent.py): q/k head mapping, L2 normalization, and state reads/writes.
- [Mamba state configuration](../../../../python/sglang/srt/configs/mamba_utils.py): state dtype and TP shapes for convolution and recurrent state.
- [Gated Delta Networks paper](https://arxiv.org/html/2412.06464v1): §3.3 for the architecture; Appendix A.2, Table S.1 for component ablations. Paper results motivate the design; exact shapes and execution paths follow this repository's implementation.
