[中文](./README.md) | [English](./README_EN.md)

# Mainstream LLM Architectures & Implementation Principles

This topic independently explains the computational structure of large language models themselves, independent of any training or inference framework. Content starts from the standard Transformer and progressively unfolds Attention, KV Cache, Sparse MoE, Multi-head Latent Attention, state space models, and representative architecture families, continuously tracking tensor shape changes and data dependencies.

## Topic Files

| File | Content |
|---|---|
| [01-decoder-only-transformer.md](./01-decoder-only-transformer.md) | Decoder-only Transformer: the complete backbone from tokens to logits, residual structure, and inference dataflow |
| [02-gqa-attention-shapes.md](./02-gqa-attention-shapes.md) | MHA/MQA/GQA, QKV projection, QK Norm, RoPE, causal Attention, and KV Cache |
| [03-sparse-moe-routing.md](./03-sparse-moe-routing.md) | Router, Top-K, Dispatch, SwiGLU Expert, Combine, and Expert Parallel |
| [04-multi-head-latent-attention.md](./04-multi-head-latent-attention.md) | MLA's low-rank compression, decoupled RoPE, absorption matrices, and compressed KV Cache principles |
| [05-architecture-families.md](./05-architecture-families.md) | Structural comparison of Encoder, Encoder-Decoder, Dense Decoder, MoE, MLA+MoE, and SSM Hybrid |

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

Standard Multi-Head Attention typically satisfies `H=Nq*D`; GQA satisfies `Nq>Nkv`; Sparse MoE satisfies `K<<E`; MLA uses `Dc`-dimensional latent state instead of storing full K/V per head.

## Reading Order

1. Start with Decoder-only Transformer to establish the `token -> hidden states -> logits` main path.
2. Unfold Attention, understanding how tokens exchange information and the space costs of MHA, GQA, and KV Cache.
3. Unfold Sparse MoE, understanding how parameters are sparsely activated per token via Top-K routing.
4. Unfold MLA, understanding how low-rank compression achieves KV cache savings.
5. Compare architecture families to understand design choices of mainstream open-source models.
