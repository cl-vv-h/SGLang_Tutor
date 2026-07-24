[中文](./README.md) | [English](./README_EN.md)

# Quantization

Model quantization for LLM inference: weight-only quantization (GPTQ, AWQ), activation quantization (W8A8, FP8), KV cache quantization, and calibration.

## Core Concepts

| Type | What's Quantized | Typical Formats | Trade-off |
|---|---|---|---|
| Weight-only | Model weights | INT4, INT8, NF4 | Smaller model, needs dequant in forward |
| W8A8 | Weights + activations | INT8, FP8 | Faster matmul on supported hardware |
| KV Cache Quant | KV Cache tensors | FP8, INT8 | Less KV memory, slight accuracy loss |

## SGLang Integration

- `LoadConfig` specifies quantization format during model loading
- Weight loaders handle dequantization or fused quantized kernels
- `configure_kv_cache_dtype()` manages KV cache quantization
- FP8 kernels supported via `torch.ops.npu` on Ascend or CUDA FP8
