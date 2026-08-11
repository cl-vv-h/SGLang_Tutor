# 01. GDN Math: Gated Delta Rule and the State Matrix

## 1. The Problem GDN Tries to Solve

Standard causal attention stores historical K/V tensors for every layer:

```text
K_cache: [context_len, num_kv_heads, head_dim]
V_cache: [context_len, num_kv_heads, head_dim]
```

The longer the context, the larger the KV Cache. GDN takes a different route: instead of storing K/V for every historical token, it maintains a per-head state matrix.

```text
S_t[h] : [V, K]
```

where:

| Symbol | Meaning |
|---|---|
| `t` | current token position |
| `h` | value head id |
| `K` | key/query head dimension |
| `V` | value head dimension |
| `S_t[h]` | recurrent state of value head `h` after token `t` has been processed |

For every new token, GDN does three things:

```text
1. Decay the old state through a gate
2. Use the key to predict the current value from the state
3. Write the prediction error back into the state, then read the output with the query
```

![GDN recurrent state update](./assets/gdn-recurrent-state.svg)

## 2. Core Equations for One Token and One Head

First consider one value head and omit batch/head indices. For the current token:

```text
q_t: [K]
k_t: [K]
v_t: [V]
S_(t-1): [V,K]
g_t: scalar, <= 0
beta_t: scalar, in (0,1)
```

The recurrent GDN update can be written as:

```text
Decay the old state:
    S_decay = exp(g_t) * S_(t-1)

Predict the current value:
    v_pred = S_decay @ k_t

Compute the residual:
    r_t = v_t - v_pred

Control write strength:
    u_t = beta_t * r_t

Write into the state:
    S_t = S_decay + u_t outer k_t

Read the output:
    o_t = S_t @ (q_t * scale)
```

Variable meanings:

| Variable | Shape | Meaning |
|---|---:|---|
| `q_t` | `[K]` | query vector of the current token; used to read from the state |
| `k_t` | `[K]` | key vector of the current token; selects the direction to write into |
| `v_t` | `[V]` | value information of the current token |
| `S_(t-1)` | `[V,K]` | state before processing the current token |
| `g_t` | scalar | forget gate; the more negative it is, the stronger the old-state decay |
| `beta_t` | scalar | write gate; the closer to 1, the more strongly the residual is written |
| `v_pred` | `[V]` | value predicted by the old state along direction `k_t` |
| `r_t` | `[V]` | difference between current value and the old state's prediction |
| `u_t` | `[V]` | residual write vector after modulation by `beta_t` |
| `o_t` | `[V]` | output of the current head |

The outer product has shape:

```text
u_t outer k_t : [V,1] @ [1,K] -> [V,K]
```

so it can be added back to `S_decay [V,K]`.

## 3. Why It Is Called a Delta Rule

The delta-rule intuition is: do not write `v_t` into the state directly; write only the part that the old state failed to explain.

```text
r_t = v_t - S_decay @ k_t
```

Read it as:

```text
If the old state can already predict v_t from k_t,
the residual is small, so this token writes very little.

If the old state predicts poorly,
the residual is large, so the state is corrected along direction k_t.
```

This differs from ordinary linear attention accumulation. A simple linear attention state update looks like:

```text
S_t = S_(t-1) + v_t outer k_t
```

GDN instead uses:

```text
S_t = exp(g_t) * S_(t-1)
      + beta_t * (v_t - exp(g_t) * S_(t-1) @ k_t) outer k_t
```

It adds two important mechanisms:

| Mechanism | Role |
|---|---|
| `exp(g_t)` | adaptively forgets old state |
| `v_t - S_decay @ k_t` | writes only prediction error instead of repeatedly writing already represented information |

## 4. Where `g_t` Comes From

In SGLang's GDN implementation, `g_t` is computed from `A_log`, `a_t`, and `dt_bias`:

```text
g_t = -exp(A_log) * softplus(a_t + dt_bias)
```

| Variable | Shape | Trainable | Meaning |
|---|---:|---:|---|
| `A_log` | `[HV]` | yes | one log-space decay parameter per value head |
| `a_t` | `[HV]` | no | token-level gate activation produced by the `in_proj_a` projection |
| `dt_bias` | `[HV]` | yes | one time-step/decay bias per value head |
| `softplus(a_t + dt_bias)` | `[HV]` | no | positive time-step or decay strength |
| `g_t` | `[HV]` | no | non-positive gate that controls old-state decay |

Why this parameterization?

```text
exp(A_log) > 0
softplus(a_t + dt_bias) > 0
g_t <= 0
0 < exp(g_t) <= 1
```

This guarantees the state decay factor is never larger than 1, preventing unconstrained amplification of old state.

The softplus function is:

```text
softplus(x) = log(1 + exp(x))
```

When `x` is large, `softplus(x) ~= x`; when `x` is very negative, it still remains positive.

## 5. Where `beta_t` Comes From

`beta_t` is produced from another projection activation `b_t`:

```text
beta_t = sigmoid(b_t)
```

| Variable | Shape | Trainable | Meaning |
|---|---:|---:|---|
| `b_t` | `[HV]` | no | current token's write-gate activation produced by the `in_proj_b` projection |
| `beta_t` | `[HV]` | no | write strength, in `(0,1)` |

`b_t` itself is not a parameter. The projection matrix that produces `b_t` is trainable.

## 6. Multi-Head Mapping and GQA-Style Sharing

In common GDN configurations, the number of key heads and value heads can differ. Let:

```text
H  = num_key_heads
HV = num_value_heads
R  = HV / H
```

If `HV > H`, multiple value heads share the same key head:

```text
key_head_id = value_head_id // R
```

SGLang's recurrent kernel uses the same mapping:

```text
i_h = i_hv // (HV // H)
```

Each value head has its own state `S[h_v]`, but the `q/k` vectors it reads with may come from the corresponding key-head group.

## 7. Why Q/K L2 Normalization Is Used

SGLang GDN kernels often enable:

```text
use_qk_l2norm_in_kernel=True
```

which means:

```text
q_t = q_t / sqrt(sum(q_t^2) + eps)
k_t = k_t / sqrt(sum(k_t^2) + eps)
```

then:

```text
q_t = q_t * (1 / sqrt(K))
```

This helps in several ways:

| Purpose | Explanation |
|---|---|
| Stable state reads/writes | `S @ k` and `S @ q` do not explode because q/k norms are too large |
| More controllable beta | residual writes are governed mainly by direction and gate strength |
| Attention-like scale | `1/sqrt(K)` follows the same intuition as the scale factor in dot-product attention |

## 8. GDN vs Standard Attention

Standard attention:

```text
score_tj = q_t dot k_j
alpha_tj = softmax(score_tj)
o_t = sum_j alpha_tj * v_j
```

GDN:

```text
S_t = recurrent_update(S_(t-1), k_t, v_t, g_t, beta_t)
o_t = S_t @ q_t
```

| Dimension | Standard Attention | GDN |
|---|---|---|
| History storage | stores K/V for every historical token | stores a fixed-size state matrix |
| How history is read | softmax weighting over all historical tokens | query reads from the state matrix |
| State size | grows linearly with context length | independent of context length |
| Strength | exact token-to-token retrieval | compressed memory and linear recurrence |
| Serving focus | KV Cache management and attention kernels | recurrent state, convolution state, chunk/recurrent kernels |

## 9. A Minimal Numeric Example

Assume:

```text
K = 2
V = 3
S_(t-1): [3,2]
k_t: [2]
v_t: [3]
q_t: [2]
```

The computation is:

```text
S_decay = exp(g_t) * S_(t-1)       # [3,2]
v_pred = S_decay @ k_t             # [3]
r_t = v_t - v_pred                 # [3]
u_t = beta_t * r_t                 # [3]
u_t outer k_t                      # [3,2]
S_t = S_decay + u_t outer k_t      # [3,2]
o_t = S_t @ q_t                    # [3]
```

This head outputs `[3]`. With multiple heads, all value-head outputs are concatenated:

```text
O_t: [HV, V]
flatten(O_t): [HV * V]
```

Then the result enters output gate/norm and the output projection.

## 10. Summary

1. GDN's core state is `S_t [V,K]`; it is per-request runtime state, not a model parameter.
2. `g_t` controls forgetting and `beta_t` controls writing. Both are computed from token activations and trainable parameters.
3. The delta rule writes `v_t - S_decay @ k_t`, the residual not already explained by the old state.
4. GDN memory does not grow with context length, but it compresses history into a state matrix and therefore trades off exact KV retrieval for compact recurrent memory.
5. Multiple value heads can share a key head; SGLang kernels implement this through `i_h = i_hv // (HV // H)`.
