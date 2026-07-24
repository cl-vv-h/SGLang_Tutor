[中文](./README.md) | [English](./README_EN.md)

# LoRA (Low-Rank Adaptation)

This topic covers the principles and serving of efficient fine-tuning methods: LoRA, QLoRA, DoRA, and AdaLoRA, with focus on their computational forms and how they integrate into inference systems.

## Core Methods

| Method | Key Idea | Inference Impact |
|---|---|---|
| LoRA | Decomposes weight updates into low-rank matrices: `ΔW = B·A` with `B ∈ R^{d×r}`, `A ∈ R^{r×k}` | Adapter weights merged as `Wx + B(Ax)`, can be pre-merged or computed online |
| QLoRA | LoRA + NF4 quantization of base model weights | Base weights dequantized during forward, adapters kept in FP16/BF16 |
| DoRA | Decomposes pre-trained weights into magnitude + direction; applies LoRA only to direction | Slightly different merge formula |
| AdaLoRA | Adaptive rank allocation across weight matrices | Ranks vary per layer, harder to batch efficiently |

## Multi-LoRA Serving

SGLang supports serving multiple LoRA adapters simultaneously:

- Each request specifies its `lora_id`
- `LoRAMemoryPool` manages adapter weight storage on GPU
- Batched requests with different adapters require segment-based kernel launches (sgmv_shrink/sgmv_expand)
- Adapter hot-loading/unloading supported at runtime
