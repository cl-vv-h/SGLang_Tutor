[中文](./15-accuracy-testing-and-debugging.md) | [English](./15-accuracy-testing-and-debugging_EN.md)

# 15. Accuracy Testing & Debugging

## 1. Three-Layer Accuracy Diagnosis

```text
Layer 1: Task-level accuracy
  → Does the model produce correct answers on benchmarks?

Layer 2: Token/text diff
  → Do generated tokens match the reference output?

Layer 3: Logits diff
  → Are hidden states and logits numerically identical?
```

## 2. Setup Requirements

- Fixed model checkpoint (same commit/hash)
- Fixed input data (JSONL dataset)
- Fixed sampling parameters (temperature=0, fixed seed)
- Fixed chat template
- Reference baseline (GPU output or known-good NPU output)

## 3. End-to-End Accuracy Eval

```bash
# Using EvalScope for accuracy evaluation
python -m evalscope.run \
  --model sglang \
  --api-url http://localhost:8000/v1 \
  --eval-dataset mmlu \
  --eval-limit 100
```

## 4. Token/Text Diff

Compare token-by-token output against reference:

```python
# Pseudo-code for token comparison
for req_id in requests:
    ref_tokens = reference_output[req_id]
    npu_tokens = npu_output[req_id]
    first_diff = find_first_difference(ref_tokens, npu_tokens)
    if first_diff is not None:
        log(f"Request {req_id}: first diff at position {first_diff}")
```

## 5. Logits Diff

Isolate the first differing forward pass:

```text
For each layer:
    Compare hidden states: max(abs(npu - ref))
    If diff > threshold → this layer is the divergence point
    
    Within the diverging layer:
    Check attention output vs FFN/MoE output
    → Attention issue? Check KV cache, backend
    → FFN issue? Check dtype, quantization, kernel
```

## 6. Feature-Specific Accuracy Risks

| Feature | Potential Risk | Mitigation |
|---|---|---|
| NPU Graph | Replay with stale buffers | Verify buffer updates between replays |
| TP | All-reduce precision | Check HCCL reduction order |
| PD Disaggregation | KV transfer corruption | Verify transfer checksums |
| Quantization | FP8/BF16 precision loss | Compare against BF16 reference |
| LoRA | SGMV kernel precision | Verify segmented matmul accuracy |
| MoE | Expert routing non-determinism | Compare router logits first |

## 7. Accuracy Debugging Flow

```text
Accuracy issue detected
  → Task-level eval: which benchmark/task fails?
  → Token diff: find first diverging token
  → Logits diff: find first diverging layer
  → Sub-layer diagnosis: attention or FFN?
  → If attention: KV cache, backend, RoPE
  → If FFN/MoE: dtype, kernel, routing
  → Fix → Re-verify all three layers
```
