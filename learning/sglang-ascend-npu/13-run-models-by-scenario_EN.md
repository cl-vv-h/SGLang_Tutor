[中文](./13-run-models-by-scenario.md) | [English](./13-run-models-by-scenario_EN.md)

# 13. Multi-Scenario Model Runbook

## Quick Reference by Scenario

### Single-card Dense Model

```bash
python -m sglang.launch_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --device npu --host 0.0.0.0 --port 8000
```

### Multi-card TP

```bash
python -m sglang.launch_server \
  --model Qwen/Qwen2.5-72B-Instruct \
  --device npu --tp-size 8
```

### PD Disaggregation

```bash
# Prefill worker (node 0)
python -m sglang.launch_server \
  --model MODEL --device npu \
  --disaggregation-mode prefill \
  --disaggregation-prefill-pp 1

# Decode worker (node 1)
python -m sglang.launch_server \
  --model MODEL --device npu \
  --disaggregation-mode decode \
  --disaggregation-decode-pp 1
```

### LoRA Serving

```bash
python -m sglang.launch_server \
  --model MODEL --device npu \
  --lora-paths /path/to/lora/adapters \
  --max-loras-per-batch 4
```

### MoE Model

```bash
python -m sglang.launch_server \
  --model Qwen/Qwen2.5-MoE-57B-A14B \
  --device npu --tp-size 8 --ep-size 4
```

### Quantized Model

```bash
python -m sglang.launch_server \
  --model MODEL-FP8 --device npu \
  --quantization fp8 --dtype float16
```

### Long Context

```bash
python -m sglang.launch_server \
  --model MODEL --device npu \
  --context-length 131072 \
  --chunked-prefill-size 8192
```

### Offline / Local Model

```bash
python -m sglang.launch_server \
  --model /path/to/local/model \
  --device npu
```

## Scenario Selection Guide

| Your Situation | Recommended Path |
|---|---|
| First NPU run | Single-card Dense Model |
| Model doesn't fit one NPU | Multi-card TP |
| Separate prefill/decode | PD Disaggregation |
| Custom fine-tuned adapters | LoRA Serving |
| Large MoE model | MoE Model with EP |
| Memory-constrained | Quantized Model |
| Long document processing | Long Context |
