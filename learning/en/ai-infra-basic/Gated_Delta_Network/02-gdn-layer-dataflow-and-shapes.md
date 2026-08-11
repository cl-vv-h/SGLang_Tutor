# 02. GDN Layer Dataflow and Shapes

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
core_attn_out + z -> out_proj -> layer output
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

If stored in BF16, this is roughly:

```text
524288 * 2 bytes = 1 MB per linear-attention layer per request slot
```

This is not tiny, but the key point is that it does not grow with the number of historical tokens.

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
