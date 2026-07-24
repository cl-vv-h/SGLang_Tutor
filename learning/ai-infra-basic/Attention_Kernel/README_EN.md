[中文](./README.md) | [English](./README_EN.md)

# Attention Kernel

This topic provides educational implementations of FlashAttention and FlashDecoding algorithms, demonstrating how attention computation is restructured to minimize HBM reads/writes through tiling and recomputation.

## Core Ideas

- **FlashAttention**: Computes attention in blocks (tiles), using online softmax to avoid materializing the full `QK^T` matrix in HBM.
- **FlashDecoding**: Optimizes the decode phase where Q is a single token, splitting KV along the sequence dimension for parallel reduction.

## Files

| File | Content |
|---|---|
| [flash_attention_tutorial.py](./flash_attention_tutorial.py) | Educational implementation of FlashAttention with tiling and online softmax |
| [flash_decoding_tutorial.py](./flash_decoding_tutorial.py) | Educational implementation of FlashDecoding for the single-query case |

## Relationship to SGLang

SGLang's attention backends (FlashInfer, Triton, Ascend) implement these algorithms for production use. Understanding the tiling and online softmax principles helps explain why attention metadata (like `cu_seqlens`, `page_table`) is shaped the way it is.
