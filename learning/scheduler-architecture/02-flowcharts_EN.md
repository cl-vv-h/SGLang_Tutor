[中文](./02-flowcharts.md) | [English](./02-flowcharts_EN.md)

# Scheduler Flowcharts

## 1. Scheduler Process Startup

Corresponding source:

- `python/sglang/srt/managers/scheduler.py:run_scheduler_process`
- `python/sglang/srt/managers/scheduler.py:configure_scheduler_process`
- `python/sglang/srt/managers/scheduler.py:Scheduler.__init__`
- `python/sglang/srt/managers/scheduler.py:Scheduler.run_event_loop`

```mermaid
flowchart TD
  A["run_scheduler_process"] --> B["load_plugins"]
  B --> C["configure_scheduler_process: Logging / Process Name / CPU Affinity / NUMA"]
  C --> D["Scheduler(...)"]
  D --> E["Init parallel state and model config"]
  E --> F["Init IPC / tokenizer / metrics / worker"]
  F --> G["Build KV cache: req_to_token_pool / token_to_kv_pool / tree_cache"]
  G --> H["Init scheduling policy / grammar / LoRA / profiler / disaggregation / overlap"]
  H --> I["pipe_writer.send(get_init_info)"]
  I --> J["run_event_loop"]
```

## 2. Event Loop Dispatch

Corresponding source:

- `scheduler.py:Scheduler.run_event_loop`
- `scheduler.py:dispatch_event_loop`

```mermaid
flowchart TD
  A["run_event_loop"] --> B["Create schedule_stream"]
  B --> C["dispatch_event_loop"]
  C --> D{"disaggregation_mode"}
  D -->|NULL| E{"pdmux / pp / overlap"}
  E -->|pdmux| E1["event_loop_pdmux"]
  E -->|pp_size > 1| E2["event_loop_pp"]
  E -->|enable_overlap| E3["event_loop_overlap"]
  E -->|Default| E4["event_loop_normal"]
  D -->|PREFILL| F["prefill disagg event loop"]
  D -->|DECODE| G["decode disagg event loop"]
```

## 3. Normal Event Loop

Corresponding source: `scheduler.py:Scheduler.event_loop_normal`

```mermaid
flowchart TD
  A["while True"] --> B["recv_requests"]
  B --> C["process_input_requests"]
  C --> D{"engine paused?"}
  D -->|Yes| A
  D -->|No| E["get_next_batch_to_run"]
  E --> F{"batch exists?"}
  F -->|Yes| G["cur_batch = batch"]
  G --> H["run_batch"]
  H --> I["process_batch_result"]
  I --> J["last_batch = batch"]
  J --> A
  F -->|No| K["on_idle"]
  K --> J
```

## 4. Overlap Event Loop

Corresponding source:

- `scheduler.py:Scheduler.event_loop_overlap`
- `scheduler.py:Scheduler.is_disable_overlap_for_batch`
- `scheduler.py:Scheduler.launch_batch_sample_if_needed`

```mermaid
flowchart TD
  A["while True"] --> B["recv_requests + process_input_requests"]
  B --> C{"engine paused?"}
  C -->|Yes| A
  C -->|No| D["Wait for forward_stream if needed"]
  D --> E["get_next_batch_to_run"]
  E --> F["Check if temporarily disable overlap"]
  F --> G{"Need to process previous results first?"}
  G -->|Yes| H["pop_and_process: process_batch_result(last_batch)"]
  G -->|No| I["Skip"]
  H --> J["run_batch current batch"]
  I --> J
  J --> K["result_queue.append(batch.copy, result)"]
  K --> L{"Can process previous results?"}
  L -->|Yes| M["pop_and_process"]
  L -->|No| N["Keep result_queue"]
  M --> O["launch_batch_sample_if_needed"]
  N --> O
  O --> P["last_batch = batch"]
  P --> A
```

## 5. Input Request to waiting_queue

Corresponding source:

- `scheduler.py:Scheduler.process_input_requests`
- `scheduler.py:Scheduler.init_request_dispatcher`
- `scheduler.py:Scheduler.handle_generate_request`
- `scheduler.py:Scheduler._add_request_to_queue`

```mermaid
flowchart TD
  A["recv_requests returns recv_req"] --> B["process_input_requests"]
  B --> C{"Health check and GPU busy?"}
  C -->|Yes| D["Defer http_worker_ipc to return_health_check_ipcs"]
  C -->|No| E["TypeBasedDispatcher dispatch by type"]
  E --> F{"TokenizedGenerateReqInput?"}
  F -->|Yes| G["handle_generate_request"]
  G --> H["Create Req / session req"]
  H --> I["Process input_embeds / multimodal / mrope / logprob / grammar"]
  I --> J{"grammar needs waiting?"}
  J -->|Yes| K["Enter grammar_queue"]
  J -->|No| L["_add_request_to_queue"]
  L --> M{"disaggregation_mode"}
  M -->|NULL| N["append waiting_queue"]
  M -->|PREFILL| O["disagg_prefill_bootstrap_queue.add"]
  M -->|DECODE| P["disagg_decode_prealloc_queue.add"]
```

## 6. get_next_batch_to_run Decision

Corresponding source:

- `scheduler.py:Scheduler.get_next_batch_to_run`
- `scheduler.py:Scheduler.get_new_batch_prefill`
- `scheduler.py:Scheduler._get_new_batch_prefill_raw`
- `scheduler.py:Scheduler.update_running_batch`

```mermaid
flowchart TD
  A["get_next_batch_to_run"] --> B["Check waiting/running timeout"]
  B --> C["Process chunked_req / dllm / hisparse special states"]
  C --> D{"last_batch is extend?"}
  D -->|Yes| E["filter last_batch and merge to running_batch"]
  D -->|No| F["Skip"]
  E --> G["Clean up prefill-only running_batch"]
  F --> G
  G --> H["get_new_batch_prefill"]
  H --> I{"new prefill batch?"}
  I -->|Yes| J["Return prefill batch"]
  I -->|No| K{"running_batch can decode?"}
  K -->|Yes| L["update_running_batch"]
  L --> M["Return decode batch"]
  K -->|No| N["Return None"]
```

## 7. Prefill Batch Construction

Corresponding source: `scheduler.py:Scheduler._get_new_batch_prefill_raw`

```mermaid
flowchart TD
  A["_get_new_batch_prefill_raw"] --> B["Re-enqueue grammar-ready requests"]
  B --> C["HiCache async event check"]
  C --> D{"running_batch full and no chunked_req?"}
  D -->|Yes| Z["return None"]
  D -->|No| E["policy.calc_priority(waiting_queue, running_batch)"]
  E --> F["Create PrefillAdder"]
  F --> G{"existing chunked_req?"}
  G -->|Yes| H["adder.add_chunked_req"]
  G -->|No| I["Iterate waiting_queue"]
  H --> I
  I --> J["LoRA / request slot / KV token / HiCache prefetch check"]
  J --> K["req.init_next_round_input"]
  K --> L["adder.add_one_req"]
  L --> M{"Can add more requests?"}
  M -->|Yes| I
  M -->|No| N["Update waiting_queue / preempt_list / chunked_req"]
  N --> O["ScheduleBatch.init_new"]
  O --> P["prepare_for_extend"]
  P --> Q{"mixed chunked prefill?"}
  Q -->|Yes| R["mix_with_running"]
  Q -->|No| S["Return new_batch"]
  R --> S
```

## 8. run_batch and Result Processing

Corresponding source:

- `scheduler.py:Scheduler.run_batch`
- `scheduler.py:Scheduler.process_batch_result`
- `scheduler_components/batch_result_processor.py:process_batch_result_prefill`
- `scheduler_components/batch_result_processor.py:process_batch_result_decode`

```mermaid
flowchart TD
  A["run_batch"] --> B["forward_ct += 1 / profiler"]
  B --> C{"generation or embedding?"}
  C -->|generation| D{"enable_overlap?"}
  D -->|Yes| E["resolve future_map / forward_stream / isolation"]
  E --> F["model_worker.forward_batch_generation"]
  F --> G["stash next token / copy_to_cpu or delayed sample"]
  D -->|No| H["resolve_forward_inputs"]
  H --> I["model_worker.forward_batch_generation"]
  I --> J["future_map.stash / update_cache_from_scheduler"]
  C -->|embedding| K["forward_batch_embedding"]
  G --> L["process_batch_result"]
  J --> L
  K --> L
  L --> M{"forward_mode"}
  M -->|decode| N["process_batch_result_decode"]
  M -->|extend/prefill| O["process_batch_result_prefill or disagg/dllm"]
  M -->|prebuilt| P["process_batch_result_prebuilt"]
  M -->|idle| Q["process_batch_result_idle"]
  N --> R["metrics / health check / cleanup"]
  O --> R
  P --> R
  Q --> R
```
