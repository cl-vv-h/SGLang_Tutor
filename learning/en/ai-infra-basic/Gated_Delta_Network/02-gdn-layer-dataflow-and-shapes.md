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
| `R` | $`\frac{\mathrm{HV}}{H}`$ | number of value heads mapped to each key head |

Unless stated otherwise, shapes are logical shapes before tensor parallel (TP) sharding. This grouping requires `HV` to be an integer multiple of `H`; tensors on a device use that TP rank's local head counts.

A common Qwen3-Next-style configuration is:

```math
H_{\mathrm{model}}=2048
```

```math
H=16
```

```math
\mathrm{HV}=32
```

```math
K=128
```

```math
V=128
```

```math
C=4
```

```math
R=2
```

Therefore:

```math
\mathrm{key\_dim}=H\cdot K=16\cdot 128=2048
```

```math
\mathrm{value\_dim}=\mathrm{HV}\cdot V=32\cdot 128=4096
```

## 2. Layer Overview

![GDN layer dataflow](./assets/gdn-layer-dataflow.svg)

A GDN layer starts from backbone hidden states:

```math
\mathrm{shape}\left(\mathrm{hidden\_states}\right)=\left[T,H_{\mathrm{model}}\right]
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

in_proj_qkvz:

```math
\mathrm{input\_size}=\mathrm{hidden\_size}
```

```math
\text{output part}= \left[\mathrm{key\_dim}, \mathrm{key\_dim}, \mathrm{value\_dim}, \mathrm{value\_dim}\right]
```

So:

```math
\mathrm{shape}\left(\mathrm{projected\_states\_qkvz}\right)=\left[T,2\cdot \mathrm{key\_dim}+2\cdot \mathrm{value\_dim}\right]
```

Substituting the common configuration:

```math
2\cdot \mathrm{key\_dim}+2\cdot \mathrm{value\_dim}=2\cdot 2048+2\cdot 4096=12288
```

The four projected segments mean:

| Segment | Logical Shape | Meaning |
|---|---:|---|
| `q` | $`\left[T,H,K\right]`$ | query |
| `k` | $`\left[T,H,K\right]`$ | key |
| `v` | $`\left[T,\mathrm{HV},V\right]`$ | value |
| `z` | $`\left[T,\mathrm{HV},V\right]`$ | gate input for output gate/norm |

`z` does not enter the recurrent state update. It modulates the GDN output at the output stage.

### 3.1 What Is `z`, and How Does It Differ from `g/beta`?

`z` is an **output-gate activation** projected from the current layer input `x_t`. Logically, `z_t = W_z x_t`; the implementation fuses `W_z` with the q/k/v projections in `in_proj_qkvz`. `W_z` is a trained parameter, whereas `z_t` is a token-dependent intermediate tensor, not another KV cache or the recurrent state.

For the default Qwen3-Next SiLU/Swish output gate described here, the code sets `norm_before_gate=True`. For each token and value head, RMSNorm operates along `V`, followed by elementwise multiplication:

```math
o={\mathrm{core\_attn\_out}}_{t,h,:}\qquad\text{[V]}
```

```math
z={z}_{t,h,:}\qquad\text{[V]}
```

```math
\mathrm{rms}=\sqrt{\mathrm{mean}\left(o\cdot o\right)+\epsilon}\qquad\text{scalar}
```

```math
{\mathrm{normed}}_{t,h,:}=\frac{\mathrm{weight}\odot o}{\mathrm{rms}}\odot \mathrm{silu}\left(z\right)\qquad\text{[V]}
```

```math
\mathrm{silu}\left(z\right)=z\odot \mathrm{sigmoid}\left(z\right)
```

Here `weight` is a trainable norm weight of length `V`, and `*` means elementwise multiplication. Thus `z` has the same shape as the output: $`\left[T,\mathrm{HV},V\right]`$. A SiLU gate is not a probability in $`\left[0,1\right]`$: it can exceed 1 or be negative, so it is neither a sigmoid gate nor a binary switch. If a model explicitly configures another `output_gate_type`, follow that configuration.

| Control Signal | Shape per Token | What It Controls | Directly Updates This Layer's Recurrent State? |
|---|---|---|---|
| $`\mathrm{exp}\left(g\right)`$ | $`\left[\mathrm{HV}\right]`$ | how much old state survives | yes |
| `beta` | $`\left[\mathrm{HV}\right]`$ | how much prediction residual is written | yes |
| `silu(z)` | $`\left[\mathrm{HV},V\right]`$ | how much of each retrieved value channel reaches subsequent computation | no |

This separates retaining information from exposing it for the current token. For example, $`z=0`$ zeros that output channel without clearing the corresponding memory in this layer's state; later tokens can still retrieve it. The operation is not `core_attn_out + z`: it is **multiplicative gating after normalization**, followed by `out_proj`.

## 4. Input Projection: `in_proj_ba`

`in_proj_ba` produces two sets of gate activations:

in_proj_ba:

```math
\mathrm{input\_size}=\mathrm{hidden\_size}
```

```math
\text{output part}= \left[\mathrm{HV}, \mathrm{HV}\right]
```

Therefore:

```math
\mathrm{shape}\left(\mathrm{projected\_states\_ba}\right)=\left[T,2\cdot \mathrm{HV}\right]
```

```math
\mathrm{shape}\left(b\right)=\left[T,\mathrm{HV}\right]
```

```math
\mathrm{shape}\left(a\right)=\left[T,\mathrm{HV}\right]
```

| Activation | Later Computation | Meaning |
|---|---|---|
| `a` | $`g=-\mathrm{exp}\left(A_{\mathrm{log}}\right)\cdot \mathrm{softplus}\left(a+\mathrm{dt\_bias}\right)`$ | controls state forgetting |
| `b` | $`\beta=\mathrm{sigmoid}\left(b\right)`$ | controls residual write strength |

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

```math
\mathrm{shape}\left(\mathrm{mixed\_qkv}\right)=\left[T,2\cdot \mathrm{key\_dim}+\mathrm{value\_dim}\right]
```

```math
\mathrm{shape}\left(z\right)=\left[T,\mathrm{HV},V\right]
```

```math
\mathrm{shape}\left(b\right)=\left[T,\mathrm{HV}\right]
```

```math
\mathrm{shape}\left(a\right)=\left[T,\mathrm{HV}\right]
```

With the common configuration:

```math
\mathrm{shape}\left(\mathrm{mixed\_qkv}\right)=\left[T,8192\right]
```

```math
\mathrm{shape}\left(z\right)=\left[T,32,128\right]
```

```text
b/a:       [T, 32]
```

`mixed_qkv` is the flat concatenation of q/k/v:

```math
\mathrm{mixed\_qkv}=\mathrm{concat}\left(\mathrm{flatten}\left(q\right),\mathrm{flatten}\left(k\right),\mathrm{flatten}\left(v\right)\right)
```

## 6. Causal Conv1d

GDN applies a short causal convolution on the q/k/v path:

```math
\mathrm{conv\_dim}=2\cdot \mathrm{key\_dim}+\mathrm{value\_dim}
```

```math
\mathrm{conv\_kernel\_size}=C
```

For packed q/k/v:

```math
\mathrm{shape}\left(\mathrm{mixed\_qkv\_before\_conv}\right)=\left[T,\mathrm{conv\_dim}\right]
```

```math
\mathrm{shape}\left(\mathrm{mixed\_qkv\_after\_conv}\right)=\left[T,\mathrm{conv\_dim}\right]
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

Convolution and SiLU preserve $`\left[T,\mathrm{conv\_dim}\right]`$ while changing the features at each position. q/k also undergo L2 normalization afterward, potentially fused into the GDN kernel; this is distinct from output RMSNorm. `z/a/b` do not pass through this short convolution.

### 6.2 Why Add It: Local Order Features and Long-Term Associative Memory

Without convolution, this layer's input projection operates independently at each token. With convolution, the q/k/v used for state reads and writes explicitly depend on several recent positions of this layer's input. The two operations provide complementary capabilities:

- **Local positional modeling**: separate weights for different lags can distinguish the preceding token from one three positions back. For example, when the window covers “not good,” the features at “good” can include the preceding negation. This illustrates representational capability, not a claim that a particular learned kernel implements this rule.
- **Long-term content association**: GDN stores and updates key→value associations in a fixed-size matrix. Short convolution first builds queries, write addresses, and content with local context, reducing the burden on that matrix memory to model local patterns on its own.
- **Modest additional cost**: for fixed `C`, convolution costs $`O\left(T\cdot \mathrm{conv\_dim}\cdot C\right)`$, and decode retains only the previous `C-1` inputs per channel, independent of context length. This is additional work; modeling quality is the main reason to include it.

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

```math
\mathrm{shape}\left(\mathrm{query}\right)=\left[1,T,H,K\right]
```

```math
\mathrm{shape}\left(\mathrm{key}\right)=\left[1,T,H,K\right]
```

```math
\mathrm{shape}\left(\mathrm{value}\right)=\left[1,T,\mathrm{HV},V\right]
```

Why is the first dimension `1`? In online serving, requests with different lengths are often packed into one token dimension:

```math
\mathrm{shape}\left(\mathrm{query\_start\_loc}\right)=\left[N+1\right]
```

This array records the start/end offsets of each request inside the packed token sequence. In this representation, the batch dimension is often written as `1`; the real request boundaries are described by `cu_seqlens` or `query_start_loc`.

## 8. Computing `g` and `beta`

For prefill/extend, SGLang calls fused gating:

```math
\text{g, beta}= \mathrm{fused\_gdn\_gating}\left(A_{\mathrm{log}}, a, b, \mathrm{dt\_bias}\right)
```

Logical shapes:

```math
\mathrm{shape}\left(a\right)=\left[T,\mathrm{HV}\right]
```

```math
\mathrm{shape}\left(b\right)=\left[T,\mathrm{HV}\right]
```

```math
\mathrm{shape}\left(A_{\mathrm{log}}\right)=\left[\mathrm{HV}\right]
```

```math
\mathrm{shape}\left(\mathrm{dt\_bias}\right)=\left[\mathrm{HV}\right]
```

```math
\mathrm{shape}\left(g\right)=\left[1,T,\mathrm{HV}\right]
```

```math
\mathrm{shape}\left(\beta\right)=\left[1,T,\mathrm{HV}\right]
```

Formula:

```math
{g}_{t,h}=-\mathrm{exp}\left({A_{\mathrm{log}}}_{h}\right)\cdot \mathrm{softplus}\left({a}_{t,h}+{\mathrm{dt\_bias}}_{h}\right)
```

```math
{\beta}_{t,h}=\mathrm{sigmoid}\left({b}_{t,h}\right)
```

The packed decode fast path fuses this computation into the recurrent kernel so it does not need to separately materialize `q/k/v/g/beta`.

## 9. Recurrent State Shape

In serving, GDN state is stored in the same kind of runtime state pool as Mamba state. Logically:

```math
\mathrm{shape}\left(\mathrm{ssm\_states}\right)=\left[\mathrm{num\_slots},\mathrm{HV},V,K\right]
```

| Dimension | Meaning |
|---|---|
| `num_slots` | number of request state slots, managed by the request pool |
| `HV` | number of value heads |
| `V` | value head dimension |
| `K` | key/query head dimension |

For one request and one layer, state size is:

```math
\mathrm{HV}\cdot V\cdot K
```

With the common configuration:

```math
32 \cdot  128 \cdot  128 = 524288 \mathrm{elements}
```

If stored in BF16:

```math
524288 \cdot  2 \mathrm{bytes} = 1 \mathrm{MiB} \mathrm{per} \mathrm{linear}-\mathrm{attention} \mathrm{layer} \mathrm{per} \mathrm{request} \mathrm{slot}
```

This is a calculation for a specified dtype. In this repository, `mamba2_state_dtype()` defaults the recurrent state to FP32, giving **2 MiB**; configuration or an environment variable can override it. These numbers cover recurrent state only, excluding conv state, extra cache slots, and temporary buffers. This is not tiny, but it does not grow with the number of historical tokens.

### 9.1 Why Not Multiply by the q/k Head Count `H` Again?

Each value head maps to **one** q/k head, rather than pairing with all `H` q/k heads. For value head `h_v`, its q/k head is:

```math
R=\left\lfloor\frac{\mathrm{HV}}{H}\right\rfloor
```

```math
h_{k}=\left\lfloor\frac{h_{v}}{R}\right\rfloor
```

```text
H = 16, HV = 32, R = 2:
q/k head 0  -> value heads 0, 1   -> S[0], S[1]
q/k head 1  -> value heads 2, 3   -> S[2], S[3]
...
q/k head 15 -> value heads 30, 31 -> S[30], S[31]
```

Value heads in a group share q/k vectors but have separate `v`, `g`, `beta`, and state matrices. For example, both `S[0]` and `S[1]` use head 0's k for writes and q for reads, but their states need not match. This grouping is commonly called grouped-value attention; do not directly reuse the head relationships of softmax GQA, where multiple query heads share K/V heads.

Adding head indices to the single-head equations from Chapter 01 makes the shapes explicit. Here q/k are already normalized, and the read scale is shown separately:

```math
q={\mathrm{query}}_{t,h_{k},:}\qquad\text{[K]}
```

```math
k={\mathrm{key}}_{t,h_{k},:}\qquad\text{[K]}
```

```math
v={\mathrm{value}}_{t,h_{v},:}\qquad\text{[V]}
```

```math
S_{\mathrm{decay}}=\mathrm{exp}\left({g}_{t,h_{v}}\right)\cdot {S_{\mathrm{prev}}}_{h_{v}}\qquad\text{[V,K]}
```

```math
r=v-S_{\mathrm{decay}}\,k\qquad\text{[V]}
```

```text
S_new[h_v] = S_decay + beta[t,h_v] * r[:,None] * k[None,:]
```

```math
{o}_{t,h_{v},:}={S_{\mathrm{new}}}_{h_{v}}\,q\cdot \mathrm{scale}\qquad\text{[V]}
```

One state matrix maps the `K`-dimensional key/query space to the `V`-dimensional value space, giving $`\left[V,K\right]`$. There are `HV` such matrices, giving $`\left[\mathrm{HV},V,K\right]`$ overall. `q` only reads the state, so no separate query state is needed. The q/k feature dimension is represented by `K`, while the q/k head count is represented through grouping:

```text
[HV,V,K] == reshape([H,R,V,K])
```

```math
\mathrm{HV}\cdot V\cdot K=H\cdot R\cdot V\cdot K
```

Thus `H` is not missing: it is already counted through $`\mathrm{HV}=H\cdot R`$. Multiplying by `H` again would allocate state for nonexistent all-to-all q/k-head × value-head pairs. Even when $`H=\mathrm{HV}`$, there are only `H` state matrices, not `H*H`.

### 9.2 Checking the Kernel and Memory Under TP

The repository's [packed recurrent decode kernel](../../../../python/sglang/srt/layers/attention/fla/fused_recurrent.py) directly uses:

```python
i_h = i_hv // (HV // H)
```

It addresses q/k using `i_h` and v/state using `i_hv`; the state offset for a head is $`i_{\mathrm{hv}}\cdot V\cdot K`$. This implements the grouping above without requiring q/k to be physically replicated first.

With conventional attention TP sharding and divisible head counts, TP degree `P` gives each rank a state of $`\left[\mathrm{num\_slots},\frac{\mathrm{HV}}{P},V,K\right]`$. Per-request, per-layer bytes on each device are `(HV/P)*V*K*state_element_bytes`; the sum across ranks remains `HV*V*K*state_element_bytes`. Head padding or special backend layouts require inspecting the actual allocation.

Also, **convolution state still includes `H` directly**: its logical element count is $`\left(2\cdot H\cdot K+\mathrm{HV}\cdot V\right)\cdot \left(C-1\right)`$. It retains a short window of q/k/v projections, whereas recurrent state retains key→value matrix memory. These are different caches.

## 10. GDN Kernel Output

The GDN recurrent/chunk kernel returns:

```math
\mathrm{shape}\left(\mathrm{core\_attn\_out}\right)=\left[1,T,\mathrm{HV},V\right]
```

Then it is reshaped:

```math
\mathrm{shape}\left(\mathrm{core\_attn\_out}\right)=\left[T,\mathrm{HV},V\right]
```

```math
\mathrm{shape}\left(z\right)=\left[T,\mathrm{HV},V\right]
```

The two tensors enter gated normalization:

```math
\mathrm{normed}=\mathrm{RMSNormGated}\left(\mathrm{core\_attn\_out},z\right)
```

```math
\mathrm{shape}\left(\mathrm{normed}\right)=\left[T,\mathrm{HV},V\right]
```

Then flatten:

```math
\mathrm{normed\_flat}: \left[T, \mathrm{HV}\cdot V\right] = \left[T, \mathrm{value\_dim}\right]
```

Finally the output projection maps it back to model hidden size:

```text
out_proj: [value_dim -> hidden_size]
```

```math
\mathrm{shape}\left(\mathrm{output}\right)=\left[T,H_{\mathrm{model}}\right]
```

This output returns to the decoder layer's residual path and then to the MLP/MoE path.

## 11. End-to-End Decode Flow

During decode, each active request usually contributes one new token:

```math
\mathrm{shape}\left(\mathrm{hidden\_states}\right)=\left[B,H_{\mathrm{model}}\right]
```

Flow:

```text
1. in_proj_qkvz:
[B,H_model] -> [B,2*key_dim+2*value_dim]
2. in_proj_ba:
[B,H_model] -> [B,2*HV]
3. split:
```

```math
\mathrm{shape}\left(\mathrm{mixed\_qkv}\right)=\left[B,2\cdot \mathrm{key\_dim}+\mathrm{value\_dim}\right]
```

```math
\mathrm{shape}\left(z\right)=\left[B,\mathrm{HV},V\right]
```

```text
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
```

```math
\left[B,\mathrm{HV},V\right]
```

```text
7. out_proj:
[B,HV*V] -> [B,H_model]
```

## 12. End-to-End Prefill/Extend Flow

Prefill/extend processes multiple tokens at once:

```math
\mathrm{shape}\left(\mathrm{hidden\_states}\right)=\left[T,H_{\mathrm{model}}\right]
```

```math
\mathrm{shape}\left(\mathrm{query\_start\_loc}\right)=\left[N+1\right]
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
| input | `hidden_states` | $`\left[T,H_{\mathrm{model}}\right]`$ |
| qkvz projection | `projected_states_qkvz` | $`\left[T,2\cdot H\cdot K+2\cdot \mathrm{HV}\cdot V\right]`$ |
| ba projection | `projected_states_ba` | $`\left[T,2\cdot \mathrm{HV}\right]`$ |
| after split | `mixed_qkv` | $`\left[T,2\cdot H\cdot K+\mathrm{HV}\cdot V\right]`$ |
| after split | `z` | $`\left[T,\mathrm{HV},V\right]`$ |
| after split | `a,b` | $`\left[T,\mathrm{HV}\right]`$ |
| after conv | `mixed_qkv` | $`\left[T,2\cdot H\cdot K+\mathrm{HV}\cdot V\right]`$ |
| recurrent input | `q,k` | $`\left[1,T,H,K\right]`$ |
| recurrent input | `v` | $`\left[1,T,\mathrm{HV},V\right]`$ |
| gating | `g,beta` | $`\left[1,T,\mathrm{HV}\right]`$ |
| state pool | `ssm_states` | $`\left[\mathrm{num\_slots},\mathrm{HV},V,K\right]`$ |
| recurrent output | `core_attn_out` | $`\left[1,T,\mathrm{HV},V\right]`$ |
| after gate/norm | `normed` | $`\left[T,\mathrm{HV},V\right]`$ |
| flatten | `normed_flat` | $`\left[T,\mathrm{HV}\cdot V\right]`$ |
| output projection | `output` | $`\left[T,H_{\mathrm{model}}\right]`$ |

## 15. Summary

1. A GDN layer projects hidden states into six activations: `q/k/v/z/a/b`.
2. `q/k/v` pass through a short causal convolution before entering the recurrent/chunk kernel.
3. `a/b` produce `g/beta`, controlling state forgetting and residual write strength.
4. `z` is not written into the state; it modulates `core_attn_out` at the output stage.
5. GDN state is $`\left[\mathrm{num\_slots},\mathrm{HV},V,K\right]`$, stored by request slot, and does not grow with token count.

## 16. Source and Paper References

- [Qwen3GatedDeltaNet](../../../../python/sglang/srt/models/qwen3_next.py): projections, convolution weights, `norm_before_gate=True`, and the output path.
- [Gated RMSNorm](../../../../python/sglang/srt/layers/attention/fla/layernorm_gated.py): the ordering of RMSNorm and the SiLU/Swish output gate.
- [GDN backend](../../../../python/sglang/srt/layers/attention/linear/gdn_backend.py): causal convolution per request and prefill/decode dispatch.
- [Fused recurrent kernel](../../../../python/sglang/srt/layers/attention/fla/fused_recurrent.py): q/k head mapping, L2 normalization, and state reads/writes.
- [Mamba state configuration](../../../../python/sglang/srt/configs/mamba_utils.py): state dtype and TP shapes for convolution and recurrent state.
- [Gated Delta Networks paper](https://arxiv.org/html/2412.06464v1): §3.3 for the architecture; Appendix A.2, Table S.1 for component ablations. Paper results motivate the design; exact shapes and execution paths follow this repository's implementation.
