[中文](./01-architecture.md) | [English](./01-architecture_EN.md)

# Architecture Overview

## Two Files' Locations

`tp_worker.py` is at `python/sglang/srt/managers/`, belonging to the runtime manager layer. It directly receives batches from the Scheduler and passes requests into the model execution layer.

`model_runner.py` is at `python/sglang/srt/model_executor/`, belonging to the model execution layer. It organizes model, distributed communication, KV cache, attention backend, CUDA graph, and sampling into a unified runtime.

## Overall Architecture

```mermaid
flowchart TB
    Client["Client / API Server"] --> Scheduler["Scheduler<br/>Request scheduling, continuous batching, KV cache decisions"]
    Scheduler --> TpWorker["TpModelWorker<br/>python/sglang/srt/managers/tp_worker.py"]
    TpWorker --> ForwardBatch["ForwardBatch<br/>python/sglang/srt/model_executor/forward_batch_info.py"]
    TpWorker --> ModelRunner["ModelRunner<br/>python/sglang/srt/model_executor/model_runner.py"]
    ModelRunner --> Dist["Distributed Runtime<br/>TP / PP / DP / EP groups"]
    ModelRunner --> MemPool["KV Cache & Token Pool<br/>ModelRunnerKVCacheMixin"]
    ModelRunner --> Attn["Attention Backend<br/>FlashInfer / Triton / Hybrid / TBO / PDMux"]
    ModelRunner --> Graph["CUDA / CPU / NPU Graph<br/>decode graph & piecewise graph"]
    ModelRunner --> Model["Loaded Model<br/>model.forward(...)"]
    ModelRunner --> Sampler["Sampler<br/>temperature / top-p / grammar / logprob"]
    Sampler --> Result["GenerationBatchResult"]
    Result --> Scheduler
```

## Role Division

| Component | Main Responsibility | Code Location |
| --- | --- | --- |
| `BaseTpWorker` | Defines the worker interface callable by Scheduler, delegates weight updates, LoRA, memory pool, etc. to `ModelRunner` | `tp_worker.py`: `BaseTpWorker` |
| `TpModelWorker` | Initializes `ModelConfig`, `ModelRunner`, tokenizer/processor, PP/TP groups; handles generation/split prefill paths | `tp_worker.py`: `TpModelWorker` |
| `ForwardBatch` | Converts scheduler-level batch into model-level tensor views: input ids, positions, KV cache loc, sampling info | `forward_batch_info.py`: `ForwardBatch` |
| `ModelRunner` | Execution layer orchestrator: distributed init, model loading, KV cache building, attention backend building, forward dispatch, sampling | `model_runner.py`: `ModelRunner` |
| `AttentionBackend` | Prepares workspace/metadata needed by attention kernels based on batch metadata | `model_runner.py`: `init_attention_backend()` & `_get_attention_backend()` |
| `Sampler` | Performs sampling or logprob computation on logits | `model_runner.py`: `sample()` & `compute_logprobs_only()` |

## TpModelWorker Architecture

```mermaid
flowchart TB
    TpInit["TpModelWorker.__init__"] --> Args["Save server_args / gpu_id / tp_rank / pp_rank"]
    Args --> ModelConfig["_init_model_config()<br/>target or draft model config"]
    ModelConfig --> Runner["_init_model_runner()<br/>Create ModelRunner"]
    Runner --> Eagle["_init_multi_layer_eagle_model_runners()<br/>Optional: multi-layer EAGLE"]
    Runner --> DLLM["_init_dllm_algorithm()<br/>Optional: dLLM"]
    Runner --> Tokenizer["Init tokenizer / processor"]
    Tokenizer --> Groups["Read pp_group / world_group"]
    Groups --> Info["Record max_total_num_tokens, max_running_requests, seed, etc."]
```

The key point of `TpModelWorker` is not running the model directly, but handling "this worker's identity in the overall parallel topology." For example:

- Whether this worker is a draft worker determines if it loads the target or draft model.
- Whether this PP rank is the last stage determines if it can sample.
- Whether overlap, grammar, speculative decoding, or dLLM is enabled determines how generation paths branch.
- Whether split prefill is active determines if `ForwardBatch` is reused or newly created.

## ModelRunner Architecture

```mermaid
flowchart TB
    MRInit["ModelRunner.__init__"] --> RuntimeState["Save rank, device, dtype, parallel config, spec config"]
    RuntimeState --> Dist["init_torch_distributed()<br/>Communication group init"]
    Dist --> Initialize["initialize()<br/>Main init pipeline"]
    Initialize --> Load["load_model()<br/>Load weights & model object"]
    Initialize --> MoE["_prepare_moe_topk()<br/>MoE routing preparation"]
    Initialize --> KV["init_memory_pool()<br/>KV cache & request-token pool"]
    Initialize --> Attn["init_attention_backend()<br/>attention backend"]
    Initialize --> Warmup["kernel_warmup() / _dummy_run()"]
    Initialize --> Graph["init_device_graphs()<br/>CUDA/CPU/NPU graph"]
    Initialize --> Piecewise["init_piecewise_cuda_graphs()<br/>piecewise graph"]
    Graph --> Forward["forward() / _forward_raw()"]
```

`ModelRunner` is an "execution environment container." It's not just a thin wrapper around `model.forward()` — it prepares all these states before actual execution:

- Distributed communication groups: TP, PP, DP, attention DP/CP, MoE EP/DP.
- Model weights with dtype/quantization/LoRA/remote weight updates.
- KV cache and request-to-token mapping pools.
- Prefill/decode attention backend.
- CUDA graph or other device graphs.
- MoE, speculative decoding, HiSparse, HiCache, ngram embedding, and other optional paths.

## Data Hierarchy After Request Entry

```mermaid
flowchart LR
    ScheduleBatch["ScheduleBatch<br/>Scheduler-level object"] --> FBInit["ForwardBatch.init_new(...)"]
    FBInit --> ForwardBatch["ForwardBatch<br/>Model-level input"]
    ForwardBatch --> ModelRunnerForward["ModelRunner.forward(...)"]
    ModelRunnerForward --> Raw["_forward_raw(...)"]
    Raw --> Decode["forward_decode(...)"]
    Raw --> Extend["forward_extend(...)"]
    Raw --> Split["forward_split_prefill(...)"]
    Raw --> Idle["forward_idle(...)"]
```

`ScheduleBatch` leans toward the scheduling perspective, recording requests, cache, sampling config, whether prefill-only, etc.

`ForwardBatch` leans toward the model perspective, recording concrete tensors, positions, cache loc, attention metadata, spec info, sampling info.

This is also a main thread for understanding SGLang runtime: Scheduler decides "which requests run together," `TpModelWorker` handles "how this rank receives the batch," and `ModelRunner` handles "how to efficiently execute this batch."
