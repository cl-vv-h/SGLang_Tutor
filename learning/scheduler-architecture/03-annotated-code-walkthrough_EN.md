[中文](./03-annotated-code-walkthrough.md) | [English](./03-annotated-code-walkthrough_EN.md)

# Scheduler: Educational Code Walkthrough

Note: This document is not modified source code, but an educational code skeleton for `python/sglang/srt/managers/scheduler.py`. Code blocks preserve key control flow with Chinese annotations explaining each section's logic. It is recommended to open the original file alongside this document.

The complete Chinese-annotated code copy is at [05-scheduler-annotated-cn.py](./05-scheduler-annotated-cn.py). That file preserves the full code structure of the original `scheduler.py` with Chinese comments inserted before classes, functions, and key state transitions — for learning cross-reference only.

## 1. Process Entry: run_scheduler_process

Location: `python/sglang/srt/managers/scheduler.py:run_scheduler_process`

```python
def run_scheduler_process(...):
    # 1. Load plugins. SGLang allows plugins to override Scheduler or dependencies,
    #    so hooks are executed before the actual Scheduler is created.
    load_plugins()

    # 2. Configure this scheduler process:
    #    - Process name
    #    - Log prefix, e.g., DP/TP/PP rank
    #    - faulthandler
    #    - CPU affinity / NUMA binding
    dp_rank = configure_scheduler_process(...)

    # 3. Optionally enable tracing, registering Scheduler thread identity.
    if server_args.enable_trace:
        process_tracing_init(...)
        trace_set_thread_info(...)

    try:
        # 4. Initialize Scheduler. This creates the model worker, KV cache, IPC,
        #    grammar, LoRA, metrics, scheduling policy — nearly all runtime components.
        scheduler = Scheduler(...)

        # 5. Send initialization info back to parent process,
        #    informing the upper server that this rank is ready.
        pipe_writer.send(scheduler.get_init_info())

        # 6. Enter blocking event loop. Normally doesn't return during service.
        scheduler.run_event_loop()

    except Exception:
        # 7. Any uncaught exception notifies parent process to exit,
        #    preventing other ranks from appearing alive when one rank dies.
        parent_process.send_signal(signal.SIGQUIT)
```

The key point: Scheduler is a worker process in a multi-process architecture, not a plain class call. After successful initialization, it persists through the event loop.

## 2. Initialization: Scheduler.__init__

Location: `python/sglang/srt/managers/scheduler.py:Scheduler.__init__`

```python
class Scheduler(...):
    def __init__(self, server_args, port_args, gpu_id, tp_rank, ...):
        # 1. Basic runtime flags. is_initializing distinguishes startup from normal service.
        self.is_initializing = True
        self.forward_ct = 0
        self.cur_batch = None

        # 2. Save startup args and extract scheduling-related switches.
        self.server_args = server_args
        self.schedule_policy = server_args.schedule_policy
        self.enable_overlap = server_args.enable_overlap_schedule
        self.enable_lora = server_args.lora_paths is not None
        self.stream_interval = server_args.stream_interval

        # 3. Compute distributed parallel info.
        #    Scheduler needs to know which TP/DP/PP/EP/ATTN_CP rank it occupies,
        #    because batch scheduling and communication depend on these topologies.
        self.ps = ParallelState(...)

        # 4. Initialize model config, metrics, IPC, tokenizer, worker.
        self.init_model_config()
        self.init_metrics_collector()
        self.init_ipc()
        self.init_tokenizer()
        self.init_model_worker()

        # 5. Build KV cache-related pools.
        #    req_to_token_pool: request slot -> token slot mapping.
        #    token_to_kv_pool_allocator: token slot -> KV cache memory.
        #    tree_cache: prefix/radix/hierarchical cache.
        kv_cache = kv_cache_builder.build_kv_cache(...)
        self.req_to_token_pool = kv_cache.req_to_token_pool
        self.token_to_kv_pool_allocator = kv_cache.token_to_kv_pool_allocator
        self.tree_cache = kv_cache.tree_cache

        # 6. Initialize running state: waiting_queue, running_batch, last_batch, etc.
        self.init_running_status()

        # 7. Initialize advanced scheduling features.
        self.init_chunked_prefill()
        self.init_schedule_policy()
        self.init_disaggregation()
        self.init_overlap_schedule()
        self.init_grammar_manager()

        # 8. Initialize request dispatcher.
        #    Subsequent process_input_requests doesn't use raw if/elif;
        #    it dispatches by request type.
        self.init_request_dispatcher()

        self.is_initializing = False
```

Think of `__init__` as "assembling the server runtime." It doesn't handle requests itself, but determines whether the subsequent event loop can access all necessary components.

## 3. Core State: init_running_status

Location: `python/sglang/srt/managers/scheduler.py:Scheduler.init_running_status`

```python
def init_running_status(self):
    # Requests waiting for prefill. New generate requests typically end up here.
    self.waiting_queue: List[Req] = []

    # Batch currently decoding, or requests that have completed prefill,
    # ready to enter decode.
    self.running_batch: ScheduleBatch = ScheduleBatch(reqs=[], batch_is_full=False)

    # Current round batch and previous round batch.
    # last_batch is critical for prefill: after prefill completes,
    # it must be merged into running_batch in the next round.
    self.cur_batch = None
    self.last_batch = None

    # When a health check arrives while busy, defer it instead of going through
    # the full generation pipeline.
    self.return_health_check_ipcs = deque()

    # Master switch for pause_generation/continue_generation.
    self._engine_paused = False
```

This set of state is the heart of Scheduler. When reading subsequent functions, prioritize observing how they change `waiting_queue`, `running_batch`, `cur_batch`, and `last_batch`.

## 4. Request Dispatcher: init_request_dispatcher

Location: `python/sglang/srt/managers/scheduler.py:Scheduler.init_request_dispatcher`

```python
def init_request_dispatcher(self):
    self._request_dispatcher = TypeBasedDispatcher([
        # Generate requests: the most important path, creates Req, enters waiting_queue.
        (TokenizedGenerateReqInput, self.handle_generate_request),

        # Embedding requests follow a similar path, but with different
        # forward mode and result processing.
        (TokenizedEmbeddingReqInput, self.handle_embedding_request),

        # Batch requests are split into individual requests for processing.
        (BatchTokenizedGenerateReqInput, self.handle_batch_generate_request),

        # Control requests: flush cache, abort, session, weight update, profile, LoRA, etc.
        (AbortReq, self.abort_request),
        (FlushCacheReq, self.flush_cache_wrapped),
        (RpcReqInput, self.handle_rpc_request),
        (LoadLoRAAdapterReqInput, self.load_lora_adapter),
        (PauseGenerationReqInput, self.pause_generation),
        (ContinueGenerationReqInput, self.continue_generation),
    ])
```

The design point: the event loop just needs "receive request → dispatcher" without mixing all request type if/elif blocks in the main loop.

## 5. Normal Event Loop: event_loop_normal

Source: `scheduler.py:Scheduler.event_loop_normal`

```python
def event_loop_normal(self):
    while True:
        # 1. Fetch a batch of requests from IPC.
        recv_reqs = self.request_receiver.recv_requests()

        # 2. Convert requests to internal state changes:
        #    - generate -> Req -> waiting_queue
        #    - abort -> mark or remove requests
        #    - flush/rpc/session/profile etc. -> corresponding handler
        self.process_input_requests(recv_reqs)

        # 3. Don't advance GPU forward when paused.
        if self._engine_paused:
            continue

        # 4. Decide what the next GPU forward should run:
        #    - Prioritize prefill for new requests
        #    - Decode running_batch when no prefill is available
        #    - Return None when neither exists
        batch = self.get_next_batch_to_run()
        self.cur_batch = batch

        if batch:
            # 5. Actually call the model worker for forward.
            result = self.run_batch(batch)

            # 6. Process output and state based on prefill/decode/embedding mode.
            self.process_batch_result(batch, result)
        else:
            # 7. When completely idle: cache checks, metrics, sleep.
            self.on_idle()

        # 8. Save this round's batch; next round may need to merge prefill results.
        self.last_batch = batch
```

The normal loop's control flow is very direct — it's the best entry point for understanding Scheduler.

## 6. Overlap Event Loop: event_loop_overlap

Source: `scheduler.py:Scheduler.event_loop_overlap`

The overlap mode's difficulty lies not in having an extra queue, but in state lifetimes: `batch.copy()`, `future_map`, and CUDA stream events all ensure that tensors not yet fully consumed by forward aren't prematurely modified or freed by Scheduler.

## 7. Input Processing: process_input_requests

Source: `scheduler.py:Scheduler.process_input_requests`

This layer is only responsible for "converting input requests into Scheduler internal actions" — it doesn't directly do batch scheduling.

## 8. Generate Request: handle_generate_request

Source: `scheduler.py:Scheduler.handle_generate_request`

The essence of `handle_generate_request` is "converting an external tokenized request into an internal `Req`." Actual batch selection hasn't happened yet.

## 9. Enqueue: _add_request_to_queue

Source: `scheduler.py:Scheduler._add_request_to_queue`

This function is the gateway for all generation requests entering the Scheduler's scheduling system. In normal mode, requests go to `waiting_queue`; in PD disaggregation mode, they enter bootstrap or prealloc queues.

## 10. Select Next Batch: get_next_batch_to_run

Source: `scheduler.py:Scheduler.get_next_batch_to_run`

This is Scheduler's core decision function: first merge the previous round's prefill, then try to schedule new prefill, finally decode. New requests can be inserted into an actively decoding service.

## 11. Construct Prefill Batch: _get_new_batch_prefill_raw

Source: `scheduler.py:Scheduler._get_new_batch_prefill_raw`

This is the densest part of Scheduler. Think of it as: `SchedulePolicy` sorts, `PrefillAdder` manages budget, `ScheduleBatch` carries the final forward input.

## 12. Update Decode Batch: update_running_batch

Source: `scheduler.py:Scheduler.update_running_batch`

The key risk in the decode phase is continuous KV cache growth, so memory checks and retraction are essential here.

## 13. Execute Batch: run_batch

Source: `scheduler.py:Scheduler.run_batch`

The `run_batch` function marks the boundary between "Scheduler (CPU scheduling)" and "GPU execution." Everything after this function enters the model forward path.

## 14. Process Batch Result

Source: `scheduler.py:Scheduler.process_batch_result` and `scheduler_components/batch_result_processor.py`

Result processing is the final step of each round, updating request state and sending output back to the client.

## Key Takeaway

Reading Scheduler means following this chain:

```
receive request → create Req → waiting_queue
→ get_next_batch_to_run → PrefillAdder → ScheduleBatch
→ run_batch → ModelWorker.forward
→ process_batch_result → update Req state → send output
→ loop
```

For the complete Chinese-annotated source code copy, see [05-scheduler-annotated-cn.py](./05-scheduler-annotated-cn.py).
