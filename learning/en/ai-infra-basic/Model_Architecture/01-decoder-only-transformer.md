# Decoder-only Transformer: From Token to Next Token

## 1. The Problem the Model Solves

A decoder-only language model receives a sequence of tokens:

```text
x_0, x_1, ..., x_(S-1)
```

And computes the conditional distribution of the next token at each position:

```text
P(x_t | x_0, x_1, ..., x_(t-1))
```

During training, all positions in the sequence can be computed in parallel, but each position can only attend to itself and earlier tokens. During inference, prefill processes the prompt in one pass, while decode appends one or a few tokens per step.

Modern generative LLMs universally use Decoder-only Transformers. The attention sublayer handles token-to-token information exchange, while the FFN or Sparse MoE sublayer handles per-token nonlinear feature transformation.

## 2. Overall Dataflow

![Decoder-only Transformer Overall Dataflow](./assets/decoder-only-overview.svg)

The overall computation can be written as:

```text
input_ids
  -> token embedding
  -> Decoder Layer 0
  -> Decoder Layer 1
  -> ...
  -> Decoder Layer L-1
  -> final RMSNorm
  -> LM Head
  -> logits
  -> sampling
  -> next token ids
```

Logical shapes at each stage:

| Stage | Input | Output | Notes |
|---|---:|---:|---|
| Token IDs | $`\left[B,S\right]`$ or packed $`\left[T\right]`$ | Same as input | Integer vocabulary indices |
| Embedding | $`\left[T\right]`$ | $`\left[T,H\right]`$ | Lookup embedding row by token id |
| Decoder Layers | $`\left[T,H\right]`$ | $`\left[T,H\right]`$ | Repeated `L` times, main path width unchanged |
| Final RMSNorm | $`\left[T,H\right]`$ | $`\left[T,H\right]`$ | Independent normalization per token's hidden vector |
| LM Head | $`\left[M,H\right]`$ | $`\left[M,V\right]`$ | `M` = number of token rows actually needing logits |
| Sampling | $`\left[M,V\right]`$ | $`\left[B\right]`$ or multiple tokens | Temperature, Top-K, Top-P sampling |

Training typically sets $`M=T`$. Online autoregressive inference usually only needs logits at each request's last valid position, so $`M=B`$ is common — there's no need to generate full-vocabulary logits for every row of the prompt.

## 3. Equal-Length Layout vs Packed Token Layout

Textbooks usually write hidden states as $`\left[B,S,H\right]`$. When sequences in a batch have different lengths, you can pad to a uniform length or pack only valid tokens:

```math
\text{request 0}: S_{0} = 4
```

```math
\text{request 1}: S_{1} = 2
```

```math
\text{request 2}: S_{2} = 5
```

If all padded to $`S_{\mathrm{max}}=5`$, shape is $`\left[3,5,H\right]`$ with 4 padding rows. Packed representation keeps only valid tokens:

```math
T=4+2+5=11
```

```math
\mathrm{hidden\_states}.\mathrm{shape} = \left[11,H\right]
```

The packed representation must additionally store each sequence's start offset and length. Attention uses these boundaries to restrict visibility; linear layers, normalization, and MoE can directly process contiguous $`\left[T,H\right]`$.

Both representations have the same mathematical meaning:

```math
\mathrm{shape}(X_{\mathrm{dense}})=[B,S,H],\quad\mathrm{shape}(X_{\mathrm{packed}})=[T,H],\quad T=\sum_iS_i
```

## 4. Token Embedding

Embedding weights:

```math
\mathrm{shape}\left(W_{\mathrm{embed}}\right)=\left[V,H\right]
```

```math
\mathrm{shape}\left(\mathrm{input\_ids}\right)=\left[T\right]
```

The embedding lookup selects one row per id:

```math
{X_{0}}_{t,:}={W_{\mathrm{embed}}}_{{\mathrm{input\_ids}}_{t},:}
```

```math
\mathrm{shape}\left(X_{0}\right)=\left[T,H\right]
```

This is not ordinary matrix multiplication — it's sparse row indexing. Each token transforms from an integer into a continuous vector of length `H`.

Position is not directly added as a $`\left[T,H\right]`$ absolute position vector. Qwen3-MoE applies positional information to Q and K within Attention via RoPE.

## 5. One Decoder Layer

![Pre-Norm Decoder Layer](./assets/decoder-block.svg)

Most modern decoder-only models use a pre-norm residual structure. Let the layer input be `x_l [T,H]`:

```math
n_{l}=\mathrm{RMSNorm}\left(x_{l}\right)\qquad\mathrm{shape}:\left[T,H\right]
```

```math
a_{l} = \mathrm{CausalGQA}\left(n_{l}\right) \left[T,H\right]
```

```math
r_{l}=x_{l}+a_{l}\qquad\mathrm{shape}:\left[T,H\right]
```

```math
m_{l}=\mathrm{RMSNorm}\left(r_{l}\right)\qquad\mathrm{shape}:\left[T,H\right]
```

```math
f_{l}=\mathrm{SparseMoE}\left(m_{l}\right)\qquad\mathrm{shape}:\left[T,H\right]
```

```math
{x}_{l+1}=r_{l}+f_{l}\qquad\mathrm{shape}:\left[T,H\right]
```

The main path is always $`\left[T,H\right]`$, making residual addition valid. Attention and MoE can change dimensions internally but must restore $`\left[T,H\right]`$ before returning.

### 5.1 RMSNorm

For a single token vector `x in R^H`:

```math
\mathrm{rms}\left(x\right) = \mathrm{sqrt}\left(\mathrm{mean}\left(x_{i}^{2}\right) + \epsilon\right)
```

```math
y_{i} = \gamma_{i} \cdot  x_{i} / \mathrm{rms}\left(x\right)
```

Where `gamma [H]` are learnable weights. RMSNorm does not mix across tokens or across hidden-dimension positions — it only computes scale over each token's `H` elements.

```math
\mathrm{shape}\left(\mathrm{input}\right)=\left[T,H\right]
```

```math
\mathrm{shape}\left(\gamma\right)=\left[H\right]
```

```math
\mathrm{shape}\left(\mathrm{output}\right)=\left[T,H\right]
```

### 5.2 Residual Connection

Residual connections add sublayer output back to the main path:

```math
r = x + \mathrm{sublayer}\left(\mathrm{norm}\left(x\right)\right)
```

This requires the two terms to have exactly the same shape. Its purpose is to preserve the original representation and provide short gradient paths for deep networks. In inference, it also determines how kernel fusion and temporary buffers are organized.

## 6. Attention vs MoE: Responsibility Boundaries

### Attention: Cross-Token Mixing

Token `t`'s Attention output depends on visible historical tokens:

```math
a_{t} = \sum_{j} \alpha_{t,j} \cdot  v_{j}, j \le  t
```

Thus Attention exchanges information along the sequence dimension. It requires positions, causal masks, and historical KV Cache.

### MoE: Per-Token Transformation

The MoE router selects experts based on each token's hidden vector. Each expert's internal MLP does not read other tokens:

```math
f_{t} = \sum_{e \in \mathrm{TopK}\left(t\right)} p_{t,e} \cdot  \mathrm{Expert\_e}\left(m_{t}\right)
```

MoE performs nonlinear transformations along the feature dimension; token-to-token coupling primarily comes from routing's parallel scheduling and communication, not mathematical token-level attention.

## 7. Final RMSNorm & LM Head

After `L` layers:

```math
\mathrm{shape}\left(X_{L}\right)=\left[T,H\right]
```

```math
X_{\mathrm{final}} = \mathrm{RMSNorm}\left(X_{L}\right): \left[T,H\right]
```

LM Head weights:

```math
\mathrm{shape}\left(W_{\mathrm{lm}}\right)=\left[V,H\right]
```

Selecting `M` rows of hidden states:

```math
\mathrm{shape}\left(H_{\mathrm{selected}}\right)=\left[M,H\right]
```

```math
\mathrm{logits}=H_{\mathrm{selected}}\,{W_{\mathrm{lm}}}^{\mathsf T}
```

```math
\mathrm{shape}\left(\mathrm{logits}\right)=\left[M,V\right]
```

Each `logits[m,v]` is the unnormalized score of the `m`-th position-to-predict for vocabulary token `v`. Probability:

```math
P\left(v\right)={\mathrm{softmax}\left(\frac{\mathrm{logits}}{\mathrm{temperature}}\right)}_{v}
```

`argmax` is greedy decoding; Top-K and Top-P renormalize over a candidate subset then sample.

## 8. Prefill Dataflow

Suppose two requests with new input lengths 3 and 2:

```math
B=2
```

```math
\text{extend lengths}= \left[3,2\right]
```

```math
T=5
```

```math
\mathrm{shape}\left(\mathrm{input\_ids}\right)=\left[5\right]
```

```math
\mathrm{shape}\left(\mathrm{positions}\right)=\left[5\right]
```

Each layer executes:

```text
hidden_states                 [5,H]
Q                             [5,Nq,D]
K_new, V_new                  [5,Nkv,D]
attention output              [5,H]
router logits                 [5,E]
top-k ids, top-k weights      [5,K], [5,K]
MoE output                    [5,H]
```

Current K/V are written to each layer's KV Cache. Attention only accesses each request's own history and currently visible prefix. In packed layout, although token rows are stored contiguously, metadata prevents cross-request attention.

## 9. Decode Dataflow

Normal decode inputs one token per active request per round:

```math
B_{\mathrm{active}}=B
```

```math
T=B
```

```math
\mathrm{shape}\left(\mathrm{input\_ids}\right)=\left[B\right]
```

```math
\mathrm{shape}\left(\mathrm{positions}\right)=\left[B\right]
```

```math
\mathrm{shape}\left(\mathrm{hidden\_states}\right)=\left[B,H\right]
```

Each layer only computes new Q/K/V:

```math
\mathrm{shape}\left(Q_{\mathrm{new}}\right)=\left[B,\mathrm{Nq},D\right]
```

```math
\mathrm{shape}\left(K_{\mathrm{new}}\right)=\left[B,\mathrm{Nkv},D\right]
```

```math
\mathrm{shape}\left(V_{\mathrm{new}}\right)=\left[B,\mathrm{Nkv},D\right]
```

But each Query needs to read that request's historical KV Cache of length `L_ctx`:

```text
logical K_history: [L_ctx,Nkv,D]
logical V_history: [L_ctx,Nkv,D]
```

Thus decode's arithmetic input tokens are few, while historical KV reads grow with context length. Attention is often memory-bandwidth-bound; MoE still executes the router and selected expert matrix multiplications.

## 10. Complete Shape Example

The values below illustrate shape relationships, not a specific checkpoint:

```math
B=2, \mathrm{extend} \mathrm{lengths}=\left[3,2\right], T=5
```

```math
H=4096,\qquad \mathrm{Nq}=32,\qquad \mathrm{Nkv}=8,\qquad D=128
```

```math
E=64,\qquad K=4,\qquad \mathrm{Ie}=1536,\qquad V=150000
```

Single-layer main path:

| Variable | Shape |
|---|---:|
| `input_ids` | $`\left[5\right]`$ |
| `hidden_states` | $`\left[5,4096\right]`$ |
| `q` | $`\left[5,32,128\right]`$ |
| `k`, `v` | $`\left[5,8,128\right]`$ |
| `attention_output` | $`\left[5,4096\right]`$ |
| `router_logits` | $`\left[5,64\right]`$ |
| `topk_ids`, `topk_weights` | $`\left[5,4\right]`$ |
| dispatched expert rows | At most $`5\cdot 4=20`$ rows, each width `4096` |
| `moe_output` | $`\left[5,4096\right]`$ |

If only computing logits for each request's last position:

```math
\mathrm{shape}\left(\mathrm{selected\_hidden}\right)=\left[2,4096\right]
```

```math
\mathrm{shape}\left(\mathrm{logits}\right)=\left[2,150000\right]
```

```math
\mathrm{shape}\left(\mathrm{sampled\_ids}\right)=\left[2\right]
```

## 11. Parameters, Activations & Runtime State

Distinguish three data categories when understanding model execution:

| Type | Examples | Lifetime |
|---|---|---|
| Parameters | Embedding, QKV, expert, LM Head weights | Persistent after model loading |
| Current Activations | Hidden states, Q, router logits, expert output | One forward pass or one sublayer |
| Cross-Step State | Per-layer KV Cache, request lengths, cache indices | Across multiple decode steps |

Decoder Layer outputs are not independently preserved as history; decode reuses per-layer K/V, not all past layers' hidden states.

## 12. Key Structural Constraints

1. The residual main path must maintain $`\left[T,H\right]`$.
2. `Nq*D` typically equals `H`, while `Nkv*D` can be significantly smaller than `H`.
3. Causal Attention cannot read future tokens or cross-request tokens.
4. MoE parameter count can grow with expert count, but single tokens only activate Top-K experts.
5. Prefill computes multiple new tokens; decode typically computes one new token per request.
6. KV Cache belongs to request runtime state, not model weights.
