# Mainstream LLM Architectures & Implementation Principles

[简体中文](../../../zh/ai-infra-basic/Model_Architecture/README.md) | **English**

This topic independently explains the computational structure of large language models themselves, independent of any training or inference framework. Content starts from the standard Transformer and progressively unfolds Attention, KV Cache, Sparse MoE, Multi-head Latent Attention, sparse/compressed attention, recurrent linear attention, state space models, and representative architecture families, continuously tracking tensor shapes, cache/state forms, and data dependencies.

## Topic Files

| File | Content |
|---|---|
| [01-decoder-only-transformer.md](./01-decoder-only-transformer.md) | Decoder-only Transformer: the complete backbone from tokens to logits, residual structure, and inference dataflow |
| [02-gqa-attention-shapes.md](./02-gqa-attention-shapes.md) | MHA/MQA/GQA, QKV projection, QK Norm, RoPE, causal Attention, and KV Cache |
| [03-sparse-moe-routing.md](./03-sparse-moe-routing.md) | Router, Top-K, Dispatch, SwiGLU Expert, Combine, and Expert Parallel |
| [04-multi-head-latent-attention.md](./04-multi-head-latent-attention.md) | MLA's low-rank compression, decoupled RoPE, absorption matrices, and compressed KV Cache principles |
| [05-architecture-families.md](./05-architecture-families.md) | Structural comparison of Encoder, Encoder-Decoder, Dense Decoder, MoE, MLA+MoE, and SSM Hybrid |
| [06-efficient-attention-landscape.md](./06-efficient-attention-landscape.md) | Unified map of MHA/GQA/MLA, SWA/NSA/DSA/CSA/HCA, GDN/KDA, and kernel-level optimization |
| [07-deepseek-sparse-attention.md](./07-deepseek-sparse-attention.md) | DSA indexer, token-level top-k, MLA sparse core, cache mapping, training, and Ascend execution path |
| [08-compressed-sparse-attention.md](./08-compressed-sparse-attention.md) | CSA overlapping compression, compressed retrieval, HCA, local SWA, heterogeneous cache, and NPU adaptation |
| [09-kimi-delta-attention.md](./09-kimi-delta-attention.md) | KDA's per-key-channel decay, delta recurrence, chunk prefill, fused decode, and mutable serving state |

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

Standard Multi-Head Attention typically satisfies `H=Nq*D`; GQA satisfies `Nq>Nkv`; Sparse MoE satisfies `K<<E`; MLA uses `Dc`-dimensional latent state instead of storing full K/V per head; sparse attention usually has `k_attn<<S`; recurrent attention stores `M` instead of a token-length cache.

## Reading Order

1. Start with Decoder-only Transformer to establish the `token -> hidden states -> logits` main path.
2. Unfold Attention, understanding how tokens exchange information and the space costs of MHA, GQA, and KV Cache.
3. Use the efficient-attention landscape to separate head sharing, feature compression, sequence sparsity/compression, recurrence, and kernel tiling.
4. Study MLA first, then DSA and CSA/HCA, following the evolution from latent cache to token sparsity and compressed sparsity.
5. Study GDN and KDA together to understand fixed-size recurrent state, chunk prefill, and fused decode.
6. Unfold Sparse MoE, understanding how parameters are sparsely activated per token via Top-K routing.
7. Compare architecture families to understand design choices of mainstream open-source models.
