[中文](./04-model-runner-attention.md) | [English](./04-model-runner-attention_EN.md)

# ModelRunner & Attention Execution

## 1. ModelRunner: The Execution Core

`ModelRunner` is the central class that orchestrates model execution:

```text
ForwardBatch → ModelRunner.forward() → _forward_raw()
  → forward_decode() / forward_extend() / forward_idle()
    → model.forward()          # The actual transformer forward
    → attention backend reads/writes KV Cache
    → LayerCommunicator handles TP/EP/CP
  → _preprocess_logits()      # Grammar, bias, softcap
  → sample() or compute_logprobs_only()
  → ModelRunnerOutput → back to Scheduler
```

## 2. ForwardBatch → Forward Dispatch

```python
def forward(self, forward_batch: ForwardBatch):
    # Try graph replay first (fast path)
    if can_use_graph(forward_batch):
        return self._forward_via_graph(forward_batch)
    # Fall back to eager execution
    return self._forward_raw(forward_batch)
```

## 3. Attention Backend Selection

SGLang supports multiple attention backends:

| Backend | Best For | Key Characteristics |
|---|---|---|
| FlashInfer | CUDA GPUs | High-performance, paged KV cache |
| Triton | Cross-platform | Flexible, good for custom kernels |
| Ascend | Ascend NPU | NPU-optimized, FRACTAL_NZ layout |
| Hybrid | Mixed workloads | Combines prefill + decode backends |
| TBO (Two Batch Overlap) | High throughput | Overlaps two decode batches |

Backend selection happens in `ModelRunner.init_attention_backend()`:
```python
backend = server_args.attention_backend  # "flashinfer", "triton", "ascend", etc.
prefill_backend = _get_attention_backend(backend, is_prefill=True)
decode_backend = _get_attention_backend(backend, is_decode=True)
```

## 4. RadixAttention: The Attention Layer

`RadixAttention` is SGLang's attention layer that integrates with the KV Cache system:
- Reads KV Cache using `page_table` from `ForwardBatch`
- Writes new K/V to cache slots
- Supports prefill (multiple new tokens) and decode (single token)

## 5. Decode Attention Dataflow

```text
Input: 1 new token per request, B active requests
Q_new: [B, Nq, D]
K_new, V_new: [B, Nkv, D]

For each request:
  K_visible = concat(K_cache[request_history], K_new)  # [Lctx, Nkv, D]
  V_visible = concat(V_cache[request_history], V_new)  # [Lctx, Nkv, D]
  O = softmax(Q_new @ K_visible^T / sqrt(D)) @ V_visible
```

The attention kernel reads `B * Lctx * Nkv * D` elements from KV Cache — the dominant memory operation in decode.

## 6. Prefill Attention Dataflow

```text
Input: S new tokens per request (S varies)
Q: [T, Nq, D]
K, V: [T, Nkv, D]  (T = sum of extend lengths)

For each request:
  causal_mask: lower triangular [S, S] or [S, prefix+S]
  O = softmax(Q @ K^T / sqrt(D) + mask) @ V
```

Prefill is compute-bound (O(S²) attention) rather than memory-bound.
