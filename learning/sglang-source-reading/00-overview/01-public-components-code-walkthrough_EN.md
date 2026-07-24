[中文](./01-public-components-code-walkthrough.md) | [English](./01-public-components-code-walkthrough_EN.md)

# Public Components Code Walkthrough

## 1. Overall Layering

SGLang's runtime is organized into 6 layers:

```text
Layer 1: Client → HTTP (http_server.py)
Layer 2: HTTP → TokenizerManager (tokenizer_manager.py)
Layer 3: TokenizerManager → Scheduler (scheduler.py) 
Layer 4: Scheduler → TpModelWorker (tp_worker.py)
Layer 5: TpModelWorker → ModelRunner (model_runner.py)
Layer 6: ModelRunner → Model → Layers → Communication
```

## 2. High-Frequency Common Components

| Layer | Component | Source | Responsibility | Downstream |
|---|---|---|---|---|
| Entry | `http_server.py` | `srt/entrypoints/http_server.py` | OpenAI-compatible API | → TokenizerManager |
| Tokenize | `TokenizerManager` | `srt/managers/tokenizer_manager.py` | Tokenize, normalize, distribute | → Scheduler |
| Schedule | `Scheduler` | `srt/managers/scheduler.py` | Batch formation, scheduling | → TpModelWorker |
| Bridge | `TpModelWorker` | `srt/managers/tp_worker.py` | ScheduleBatch → ForwardBatch | → ModelRunner |
| Execute | `ModelRunner` | `srt/model_executor/model_runner.py` | Model forward, sampling | → Model layers |
| Model | `LlamaForCausalLM` etc. | `srt/models/` | Transformer forward | → LayerCommunicator |
| Communicate | `LayerCommunicator` | `srt/layers/communicator.py` | TP/EP/CP communication | → Kernels |

## 3. Main Chain Code Walkthrough

### 3.1 HTTP Request Entry

`http_server.py` registers FastAPI routes. The chat completions endpoint:

```python
@app.post("/v1/chat/completions")
async def openai_v1_chat_completions(request: ChatCompletionRequest):
    # Convert OpenAI format to internal GenerateReqInput
    adapted_request = adapt_chat_completion_request(request)
    # Send to TokenizerManager via IPC
    return await tokenizer_manager.generate_request(adapted_request)
```

### 3.2 TokenizerManager Request Normalization

`tokenizer_manager.py` handles:
1. Apply chat template → convert to token IDs
2. Create `TokenizedGenerateReqInput` with tokenized data
3. Send to Scheduler via IPC (`send_to_scheduler`)
4. Wait for response via `_wait_one_response`

### 3.3 Scheduler Receives Requests

`scheduler.py` in `event_loop_normal`:
1. `recv_requests()` — receive from IPC
2. `process_input_requests()` — dispatch by type
3. `handle_generate_request()` — create `Req` object
4. `_add_request_to_queue()` — append to `waiting_queue`

### 3.4 Req and ScheduleBatch

- `Req` (`schedule_batch.py`): Single request runtime state — token IDs, sampling params, cache indices, finish status
- `ScheduleBatch`: Container for one forward pass — reqs list, forward mode, input tensors, KV cache locs

### 3.5 TpModelWorker Bridges Scheduling to Execution

`tp_worker.py:forward_batch_generation()`:
```python
def forward_batch_generation(self, batch: ScheduleBatch):
    forward_batch = ForwardBatch.init_new(batch, self.model_runner)
    result = self.model_runner.forward(forward_batch)
    if self.pp_group.is_last_rank:
        result = self.model_runner.sample(result, forward_batch)
    return GenerationBatchResult(...)
```

### 3.6 ForwardBatch as Model Execution Metadata Contract

`forward_batch_info.py:ForwardBatch`:
- `input_ids`: `[T]` packed token IDs
- `positions`: `[T]` position indices
- `out_cache_loc`: Where to write KV Cache
- `seq_lens`: Per-request sequence lengths
- `sampling_info`: Temperature, top-p, top-k, etc.
- `spec_info`: Speculative decoding metadata

### 3.7 ModelRunner Enters Real Forward

`model_runner.py:forward()`:
```python
def forward(self, forward_batch: ForwardBatch):
    # Wrap with profiling, error recovery
    return self._forward_raw(forward_batch)

def _forward_raw(self, forward_batch):
    if forward_batch.forward_mode.is_decode():
        return self.forward_decode(forward_batch)
    elif forward_batch.forward_mode.is_extend():
        return self.forward_extend(forward_batch)
    # ...
```

### 3.8 Model Classes and Layer Execution

Models like `LlamaForCausalLM` contain:
- `self.model` — the transformer backbone
- `self.lm_head` — final projection to vocabulary
- Each layer: `Attention + FFN/MoE` with residual connections

### 3.9 LayerCommunicator Handles Intra-Layer Communication

`layers/communicator.py:LayerCommunicator`:
- TP all-reduce/reduce-scatter for attention/FFN outputs
- EP all-to-all for MoE dispatch/combine
- CP communication for sequence parallelism

### 3.10 KV/Cache Layer Throughout

KV Cache connects scheduling (Scheduler manages capacity) and execution (attention reads/writes):

```text
Scheduler: req_to_token_pool, tree_cache
  ↓
ModelRunner: token_to_kv_pool, attention backend
  ↓
Attention: read/write KV Cache per layer per token
```

### 3.11 Output and Detokenize

After sampling: `GenerationBatchResult` → `BatchResultProcessor` → `OutputStreamer` → `DetokenizerManager` → HTTP response.

## 4. Reading Suggestions by Layer

1. Start at `http_server.py` — understand API surface
2. Read `tokenizer_manager.py` — understand request normalization
3. Read `scheduler.py` `event_loop_normal` — core loop
4. Read `schedule_batch.py` — `Req` and `ScheduleBatch`
5. Read `tp_worker.py` — the bridge
6. Read `model_runner.py:forward()` — execution entry
7. Read `forward_batch_info.py` — the metadata contract
8. Read one model (`llama.py`) — concrete layer structure
9. Read `layer_communicator.py` — distributed ops
10. Read `radix_cache.py` — cache system
