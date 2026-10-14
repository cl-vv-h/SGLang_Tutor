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

Model-class resolution and weight loading are covered in `learning/en/tp-worker-model-runner/06-model-loading-and-architecture-resolution.md`. That lesson connects `TpModelWorker._init_model_config()`, `ModelConfig.__init__()`, `ModelRegistry.resolve_model_cls()`, and `DefaultModelLoader.load_model()` into the full `model_path -> config.json.architectures -> Python model class -> checkpoint weights` path.

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
```

```math
\mathrm{shape}\left(Q_{\mathrm{new}}\right)=\left[B,\mathrm{Nq},D\right]
```

```text
K_new, V_new: [B, Nkv, D]
For each request:
```

```math
K_{\mathrm{visible}}=\mathrm{concat}\left({K_{\mathrm{cache}}}_{\mathrm{request\_history}},K_{\mathrm{new}}\right)\qquad\text{[Lctx, Nkv, D]}
```

```math
V_{\mathrm{visible}}=\mathrm{concat}\left({V_{\mathrm{cache}}}_{\mathrm{request\_history}},V_{\mathrm{new}}\right)\qquad\text{[Lctx, Nkv, D]}
```

```math
O=\mathrm{softmax}\left(\frac{Q_{\mathrm{new}}\,{K_{\mathrm{visible}}}^{\mathsf T}}{\sqrt{D}}\right)\,V_{\mathrm{visible}}
```

The attention kernel reads $`B\cdot \mathrm{Lctx}\cdot \mathrm{Nkv}\cdot D`$ elements from KV Cache — the dominant memory operation in decode.

## 6. Prefill Attention Dataflow

```text
Input: S new tokens per request (S varies)
```

```math
\mathrm{shape}\left(Q\right)=\left[T,\mathrm{Nq},D\right]
```

```math
\text{K, V}: \left[T, \mathrm{Nkv}, D\right] \left(T = \sum \mathrm{of} \mathrm{extend} \mathrm{lengths}\right)
```

```text
For each request:
causal_mask: lower triangular [S, S] or [S, prefix+S]
```

```math
O=\mathrm{softmax}\left(\frac{Q\,{K}^{\mathsf T}}{\sqrt{D}}+\mathrm{mask}\right)\,V
```

Prefill is compute-bound (O(S²) attention) rather than memory-bound.
