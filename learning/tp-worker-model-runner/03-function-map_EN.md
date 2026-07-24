[中文](./03-function-map.md) | [English](./03-function-map_EN.md)

# Function & Code Segment Map

This file groups key functions from `tp_worker.py` and `model_runner.py` by reading purpose. It's recommended to first understand function responsibilities, then open the annotated source for segment-by-segment reading.

## tp_worker.py

| Code Location | Purpose | Reading Focus |
| --- | --- | --- |
| `BaseTpWorker` | TP worker abstract base class | Defines what capabilities Scheduler expects: generation, embedding, weight updates, LoRA, memory pool |
| `BaseTpWorker.model_runner` | Abstract property | Subclasses must provide underlying `ModelRunner` |
| `BaseTpWorker.get_memory_pool()` | Read KV cache/request pool | Scheduler can query memory pool state through worker |
| `BaseTpWorker.update_weights*()` | Weight update delegation | Actual update logic is in `ModelRunner` |
| `BaseTpWorker.update_lora_*()` | LoRA update delegation | Worker only does entry-level forwarding |
| `BaseTpWorker.forward_batch_embedding()` | Embedding model execution entry | Embedding path does no token sampling |
| `TpModelWorker.__init__()` | Worker main init | Reads rank/device/server args, creates `ModelConfig`, `ModelRunner`, tokenizer, PP/world group |
| `TpModelWorker._init_model_config()` | Model config init | Draft worker uses `speculative_draft_model_path`, target worker uses main model path |
| `TpModelWorker._init_model_runner()` | Create `ModelRunner` | Passes rank, TP/PP/DP params, memory pool, model config to execution layer |
| `TpModelWorker._init_multi_layer_eagle_model_runners()` | Multi-layer EAGLE init | One worker can hold multiple draft runners for speculative decoding |
| `TpModelWorker._init_dllm_algorithm()` | dLLM init | Diffusion/denoising LLM uses dedicated algorithm object |
| `TpModelWorker.get_worker_info()` | Report worker capabilities | Scheduler uses this to understand worker capacity and config |
| `TpModelWorker.forward_batch_generation()` | Generation main entry | **Most important function**: `ScheduleBatch -> ForwardBatch -> ModelRunner.forward -> sample/result` |
| `TpModelWorker.forward_batch_split_prefill()` | Split prefill entry | Long prefill split into multiple forward segments |

## model_runner.py: Top-Level Utilities & Structure

| Code Location | Purpose | Reading Focus |
| --- | --- | --- |
| `add_mla_attention_backend()` | Dynamically register MLA attention backend | MLA models override attention backend selection |
| `add_chunked_prefix_cache_attention_backend()` | Dynamically register chunked prefix cache backend | Adapts prefix cache to attention backend |
| `resolve_language_model()` | Find the real language model body | Different model wrappers have different layer hierarchies; need unified `.layers` access |
| `RankZeroFilter` | Log filtering | Prevents multi-rank duplicate output |
| `ModelRunnerOutput` | Unified forward return structure | Carries logits, hidden states, spec info, debug/metrics output |
| `ModelRunner` | Model execution core class | All subsequent init, forward, sampling centers around this |

## model_runner.py: Initialization Chain

| Code Location | Purpose | Reading Focus |
| --- | --- | --- |
| `ModelRunner.__init__()` | Entry init | Saves params, sets rank/device/spec/parallel state, calls distributed init and main init |
| `ModelRunner.initialize()` | Main init orchestration | Loads model, prepares MoE, KV cache, attention backend, warmup, graph capture |
| `ModelRunner.init_torch_distributed()` | Distributed init | Establishes TP/PP/DP/EP/attention DP/CP communication groups |
| `ModelRunner.load_model()` | Model loading | Builds `LoadConfig`, calls loader, sets dtype/sliding window/quantization/remote weight logic |
| `ModelRunner._prepare_moe_topk()` | MoE top-k preparation | Compatible with different MoE runner routing logic |
| `ModelRunner.configure_kv_cache_dtype()` | KV cache dtype decision | `auto`, FP8, BF16, FP4 branches all handled here |
| `ModelRunner.init_attention_backend()` | Attention backend init | Normal, Hybrid, PDMux, Two Batch Overlap all fork from here |
| `ModelRunner._get_attention_backend()` | Resolve prefill/decode backend | Prefill and decode can use different backends, combined as Hybrid |
| `ModelRunner._get_attention_backend_from_str()` | Backend instantiation | Creates concrete backend from `ATTENTION_BACKENDS` table |
| `ModelRunner.kernel_warmup()` | Kernel warmup | Autotune/warmup before CUDA graph capture |
| `ModelRunner._dummy_run()` | Dummy batch warmup | Constructs executable `ForwardBatch` for autotune and graph capture |
| `ModelRunner.init_device_graphs()` | Capture device graphs | Decode graph mainly created here |
| `ModelRunner.init_piecewise_cuda_graphs()` | Capture piecewise graphs | Finer-grained graph optimization for attention/MoE layers |

## model_runner.py: Forward Execution Chain

| Code Location | Purpose | Reading Focus |
| --- | --- | --- |
| `ModelRunner.forward()` | External unified forward entry | Wraps profiling, canary, EPLB, error recovery, and `_forward_raw()` |
| `ModelRunner._forward_raw()` | Forward dispatch core | Selects decode, extend, split prefill, idle by `ForwardMode`; prioritizes graph replay |
| `ModelRunner.forward_decode()` | Decode forward | Initializes decode attention metadata, calls `model.forward()` |
| `ModelRunner.forward_extend()` | Extend/prefill forward | Handles input embeds, PP proxy, piecewise graph, and prefill metadata |
| `ModelRunner.forward_split_prefill()` | Split prefill forward | Maintains split index, calls model's `forward_split_prefill()` |
| `ModelRunner.forward_idle()` | Idle forward | DP attention scenarios where idle ranks still participate in synchronization |

## model_runner.py: Sampling & Logprob

| Code Location | Purpose | Reading Focus |
| --- | --- | --- |
| `ModelRunner._preprocess_logits()` | Logits preprocessing | Grammar, bias, softcap, logprob pre-sampling processing |
| `ModelRunner.sample()` | Token sampling | Generates next token from logits and sampling_info |
| `ModelRunner.compute_logprobs_only()` | Logprob-only compute | Used for return_logprob or prefill-only paths |
| `ModelRunner.maybe_init_ngram_embedding()` | Ngram embedding init | Builds token table and module buffers |
| `ModelRunner.maybe_update_ngram_token_table()` | Update ngram token table | Writes new token back to ngram state after sampling |

## Key Integration Points Between the Two Files

| Integration Point | Upstream | Downstream | Meaning |
| --- | --- | --- | --- |
| `ForwardBatch.init_new(batch, model_runner)` | `TpModelWorker.forward_batch_generation()` | `ModelRunner.forward()` | Request transforms from scheduling object to model execution object |
| `model_runner.forward(forward_batch)` | `TpModelWorker` | `ModelRunner._forward_raw()` | Worker enters model execution core |
| `model_runner.sample(logits_output, forward_batch)` | PP last-stage `TpModelWorker` | `Sampler` | Logits become next token |
| `pp_hidden_states_proxy_tensors` | PP non-last-stage `ModelRunner` | Next PP rank | Pipeline parallel intermediate stages only pass hidden states |
| `get_memory_pool()` | Scheduler/Worker | `ModelRunnerKVCacheMixin` | Scheduling layer can sense KV cache capacity |
