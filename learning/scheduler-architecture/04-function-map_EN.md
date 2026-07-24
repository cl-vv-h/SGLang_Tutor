[中文](./04-function-map.md) | [English](./04-function-map_EN.md)

# Scheduler Function Map

This file organizes `scheduler.py` by "Entry → Input → Scheduling → Execution → Result → Control". Line numbers come from the current repository snapshot for auxiliary reference.

## Process & Event Loop

| Function | Location | What It Does | Main Next Hop |
| --- | --- | --- | --- |
| `run_scheduler_process` | `scheduler.py:3951` | Scheduler child process entry, loads plugins, configures process, creates Scheduler, notifies parent, enters event loop | `configure_scheduler_process`, `Scheduler.__init__`, `run_event_loop` |
| `configure_scheduler_process` | `scheduler.py:3894` | Sets process name, logging, faulthandler, CPU affinity, NUMA binding | Returns `dp_rank` |
| `Scheduler.__init__` | `scheduler.py:296` | Assembles runtime: parallel state, model config, IPC, worker, KV cache, scheduling policy, overlap, grammar, etc. | `init_running_status`, `init_schedule_policy`, `init_request_dispatcher` |
| `run_event_loop` | `scheduler.py:1404` | Creates schedule stream, hands control to concrete event loop | `dispatch_event_loop` |
| `dispatch_event_loop` | `scheduler.py:3863` | Selects event loop implementation based on disaggregation, PP, overlap, pdmux | `event_loop_normal`, `event_loop_overlap`, PP/disagg loop |
| `event_loop_normal` | `scheduler.py:1425` | Non-overlap main loop: receive requests, select batch, forward, process results | `process_input_requests`, `get_next_batch_to_run`, `run_batch`, `process_batch_result` |
| `event_loop_overlap` | `scheduler.py:1452` | Overlap main loop: current batch forward overlaps with previous batch result processing | `is_disable_overlap_for_batch`, `run_batch`, `launch_batch_sample_if_needed` |

## Input & Enqueue

| Function | Location | Modified Core State | Description |
| --- | --- | --- | --- |
| `init_request_dispatcher` | `scheduler.py:1279` | `self._request_dispatcher` | Builds request type to handler mapping |
| `process_input_requests` | `scheduler.py:1543` | `waiting_queue`, `return_health_check_ipcs`, control state | Dispatches a batch of IPC requests one by one, immediately returns control request output |
| `handle_generate_request` | `scheduler.py:1898` | May create `Req` and enqueue | Handles session, input embeds, multimodal, length validation, logprob, grammar |
| `_add_request_to_queue` | `scheduler.py:2156` | `waiting_queue` or disagg dedicated queues | Normal mode: append waiting queue; PD mode: enter bootstrap/prealloc queues |
| `abort_request` | `scheduler.py:3566` | `waiting_queue`, grammar/disagg queues, `req.to_finish` | Aborts unstarted, waiting, disagg, and running requests |

## Scheduling Decisions

| Function | Location | Input | Output | Description |
| --- | --- | --- | --- | --- |
| `get_next_batch_to_run` | `scheduler.py:2404` | `waiting_queue`, `running_batch`, `last_batch`, `chunked_req` | `ScheduleBatch` or `None` | Scheduler core decision: prefill first, then decode |
| `get_new_batch_prefill` | `scheduler.py:2532` | waiting/running state | prefill `ScheduleBatch` or `None` | Wraps prefill delayer, then calls raw version |
| `_get_new_batch_prefill_raw` | `scheduler.py:2552` | waiting queue, KV cache, policy, chunked_req | prefill `ScheduleBatch` or `None` | Uses `SchedulePolicy` + `PrefillAdder` to select prefilled requests for this round |
| `update_running_batch` | `scheduler.py:2823` | `running_batch` | decode `ScheduleBatch` or empty batch | Cleans up completed requests, checks decode memory, retracts if insufficient, prepares decode tensors |

## Execution & Results

| Function | Location | What It Does | Key Collaborators |
| --- | --- | --- | --- |
| `run_batch` | `scheduler.py:2965` | Calls model worker to execute generation/embedding forward | `model_worker`, `tp_worker`, `future_map`, CUDA streams |
| `_overlap_forward_isolation` | `scheduler.py` | Protects `ScheduleBatch` fields and GPU tensor lifetimes under overlap | `record_batch_in_overlap` |
| `launch_batch_sample_if_needed` | `scheduler.py:3136` | Handles delayed sampling, typically for overlap + structured output scenarios | `future_map`, `delay_sample_func` |
| `process_batch_result` | `scheduler.py:3167` | Dispatches result processing by forward mode, unified metrics/health/cleanup | `BatchResultProcessor` |
| `on_idle` | `scheduler.py:3249` | When idle: memory consistency checks, metrics, KV events, sleep | `invariant_checker`, `metrics_reporter`, `idle_sleeper` |

## Control & Maintenance

| Function | Location | Description |
| --- | --- | --- |
| `is_fully_idle` | `scheduler.py:3285` | Checks if Scheduler is completely idle; looks beyond running/waiting to overlap result queue, grammar, disagg, HiCache, etc. |
| `flush_cache` | `scheduler.py:3432` | Only when fully idle: clears tree cache, req/token pools, grammar, metrics, and device allocator cache |
| `pause_generation` | `scheduler.py:3677` | Pauses generation; can keep state in-place or retract running requests |
| `continue_generation` | `scheduler.py:3739` | Resumes from pause, optionally executes `torch.cuda.empty_cache` |

## Related Supporting Classes

| Class/Function | Location | Why Scheduler Depends on It |
| --- | --- | --- |
| `Req` | `schedule_batch.py:641` | Single-request runtime state: tokens, sampling params, cache indices, finish reason |
| `ScheduleBatch` | `schedule_batch.py:1481` | Batch container for one forward pass |
| `ScheduleBatch.init_new` | `schedule_batch.py:1649` | Creates batch from `Req` list, binds cache/pool/tree_cache |
| `ScheduleBatch.prepare_for_extend` | `schedule_batch.py:1813` | Prepares input tensors and metadata for prefill/extend forward |
| `ScheduleBatch.prepare_for_decode` | `schedule_batch.py:2383` | Prepares next-token input and seq lens for decode forward |
| `ScheduleBatch.filter_batch` | `schedule_batch.py:2477` | Removes finished/aborted/non-runnable requests |
| `ScheduleBatch.merge_batch` | `schedule_batch.py:2560` | Merges last round's prefill-completed requests into running batch |
| `ScheduleBatch.check_decode_mem` | `schedule_batch.py:2261` | Checks if there's KV cache space for next decode step |
| `ScheduleBatch.retract_decode` | `schedule_batch.py:2274` | Retracts some decode requests when memory is insufficient |
| `SchedulePolicy.calc_priority` | `schedule_policy.py:162` | Orders or computes priority for waiting queue |
| `PrefillAdder` | `schedule_policy.py:405` | Selects prefill requests within token/request/cache budget |
| `PrefillAdder.add_one_req` | `schedule_policy.py:828` | Attempts to add a single request to this round's prefill batch |
| `PrefillAdder.add_chunked_req` | `schedule_policy.py:679` | Continues scheduling cross-round chunked prefill requests |
| `PrefillAdder.preempt_to_schedule` | `schedule_policy.py:985` | Preempts lower-priority running requests during priority preemption |
| `BatchResultProcessor.process_batch_result_prefill` | `scheduler_components/batch_result_processor.py:178` | Processes prefill results, first token, prefix cache, output |
| `BatchResultProcessor.process_batch_result_decode` | `scheduler_components/batch_result_processor.py:588` | Processes decode tokens, finish decisions, streaming output, cache release |

## Minimal Reading Path

To understand the path of a single normal generation request, open functions in this order:

1. `scheduler.py:Scheduler.event_loop_normal`
2. `scheduler.py:Scheduler.process_input_requests`
3. `scheduler.py:Scheduler.handle_generate_request`
4. `scheduler.py:Scheduler._add_request_to_queue`
5. `scheduler.py:Scheduler.get_next_batch_to_run`
6. `scheduler.py:Scheduler._get_new_batch_prefill_raw`
7. `scheduler.py:Scheduler.run_batch`
8. `scheduler.py:Scheduler.process_batch_result`
9. `scheduler_components/batch_result_processor.py:process_batch_result_prefill`
10. `scheduler_components/batch_result_processor.py:process_batch_result_decode`
