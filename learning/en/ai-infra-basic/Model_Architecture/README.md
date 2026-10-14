# Mainstream LLM Architectures & Implementation Principles

[简体中文](../../../zh/ai-infra-basic/Model_Architecture/README.md) | **English**

This topic independently explains the computational structure of large language models themselves, independent of any training or inference framework. Content starts from the standard Transformer and progressively unfolds Attention, KV Cache, Sparse MoE, Multi-head Latent Attention, sparse/compressed attention, recurrent linear attention, state space models, and representative architecture families, continuously tracking tensor shapes, cache/state forms, and data dependencies.

## Topic Files

| File | Content |
|---|---|
| [01-decoder-only-transformer.md](./01-decoder-only-transformer.md) | Decoder-only Transformer: the complete backbone from tokens to logits, residual structure, and inference dataflow |
| [02-gqa-attention-shapes.md](./02-gqa-attention-shapes.md) | MHA/MQA/GQA, QKV projection, QK Norm, RoPE, causal Attention, and KV Cache |
| [03-sparse-moe-routing.md](./03-sparse-moe-routing.md) | Router, Top-K, Dispatch, SwiGLU Expert, Combine, and Expert Parallel |
| [04-multi-head-latent-attention.md](./04-multi-head-latent-attention.md) | Derives MLA low-rank compression, decoupled RoPE, K/V matrix absorption, cache bytes, and every tensor from one complete decode step |
| [05-architecture-families.md](./05-architecture-families.md) | Reconstructs Encoder, Decoder, MoE, MLA, SSM, hybrid, and multimodal architectures as Token Mixer + Channel Mixer + Runtime State |
| [06-efficient-attention-landscape.md](./06-efficient-attention-landscape.md) | Compares MHA/GQA/MLA, SWA/DSA/CSA/HCA, and GDN/KDA along four independent axes: cache width, visited positions, recurrent state, and kernel execution |
| [07-deepseek-sparse-attention.md](./07-deepseek-sparse-attention.md) | Separates indexer and main-attention scores, then traces top-k, paged-address translation, sparse MLA, and training tensor by tensor |
| [08-compressed-sparse-attention.md](./08-compressed-sparse-attention.md) | Uses timelines and numerical examples for CSA overlap compression, HCA, local SWA, entry readiness, heterogeneous caches, and NPU adaptation |
| [09-kimi-delta-attention.md](./09-kimi-delta-attention.md) | Derives KDA decay and delta rule as rewritable key-value memory, including convolution, chunk prefill, fused decode, and serving state |

## Unified Notation

| Symbol | Meaning |
|---|---|
| `B` | batch size |
| `S` | sequence length |
| `T` | total token count after packing variable-length sequences |
| `H` | hidden size |
| `L` | number of layers |
| `Nq` | number of Query heads |
| `Nkv` | number of Key/Value heads |
| `D` | head dimension |
| `V` | vocabulary size |
| `E` | number of routed experts |
| `K` | number of experts selected per token |
| `I` | Dense FFN intermediate size |
| `Ie` | single expert intermediate size |
| `Dc` | MLA KV latent compression dimension |
| `k_attn` | number of historical entries selected by sparse attention |
| `m` | sequence-compression stride |
| `w` | local sliding-window length |
| `M` | recurrent linear-attention state matrix |

Standard Multi-Head Attention typically satisfies $`H=\mathrm{Nq}\cdot D`$; GQA satisfies $`\mathrm{Nq} \gt \mathrm{Nkv}`$; Sparse MoE satisfies `K<<E`; MLA uses `Dc`-dimensional latent state instead of storing full K/V per head; sparse attention usually has `k_attn<<S`; recurrent attention stores `M` instead of a token-length cache.

## Reading Order

1. Start with Decoder-only Transformer to establish the `token -> hidden states -> logits` main path.
2. Unfold Attention, understanding how tokens exchange information and the space costs of MHA, GQA, and KV Cache.
3. From chapter 04 onward, begin with each chapter's reading map and running example; they define the global dataflow and shared shape coordinates for the later equations.
4. Study MLA, then DSA and CSA/HCA, following “compress each position → select token positions → compress the sequence before selection.”
5. Study KDA next and contrast token-addressable history with history folded into a fixed matrix.
6. Return to the efficient-attention landscape and compare cache width, visited positions, state growth, and kernel execution across techniques.
7. Study Sparse MoE, then use architecture families to reassemble Token Mixer, Channel Mixer, and Runtime State into complete models.
