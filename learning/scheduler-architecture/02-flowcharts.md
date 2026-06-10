# Scheduler 流程图

## 1. Scheduler 进程启动

对应源码：

- `python/sglang/srt/managers/scheduler.py:run_scheduler_process`
- `python/sglang/srt/managers/scheduler.py:configure_scheduler_process`
- `python/sglang/srt/managers/scheduler.py:Scheduler.__init__`
- `python/sglang/srt/managers/scheduler.py:Scheduler.run_event_loop`

```mermaid
flowchart TD
  A["run_scheduler_process"] --> B["load_plugins"]
  B --> C["configure_scheduler_process: 日志 / 进程名 / CPU 亲和性 / NUMA"]
  C --> D["Scheduler(...)"]
  D --> E["初始化并行状态和模型配置"]
  E --> F["初始化 IPC / tokenizer / metrics / worker"]
  F --> G["构建 KV cache: req_to_token_pool / token_to_kv_pool / tree_cache"]
  G --> H["初始化调度策略 / grammar / LoRA / profiler / disaggregation / overlap"]
  H --> I["pipe_writer.send(get_init_info)"]
  I --> J["run_event_loop"]
```

## 2. Event loop 分发

对应源码：

- `scheduler.py:Scheduler.run_event_loop`
- `scheduler.py:dispatch_event_loop`

```mermaid
flowchart TD
  A["run_event_loop"] --> B["创建 schedule_stream"]
  B --> C["dispatch_event_loop"]
  C --> D{"disaggregation_mode"}
  D -->|NULL| E{"pdmux / pp / overlap"}
  E -->|pdmux| E1["event_loop_pdmux"]
  E -->|pp_size > 1| E2["event_loop_pp"]
  E -->|enable_overlap| E3["event_loop_overlap"]
  E -->|默认| E4["event_loop_normal"]
  D -->|PREFILL| F["prefill disagg event loop"]
  D -->|DECODE| G["decode disagg event loop"]
```

## 3. 普通事件循环

对应源码：`scheduler.py:Scheduler.event_loop_normal`

```mermaid
flowchart TD
  A["while True"] --> B["recv_requests"]
  B --> C["process_input_requests"]
  C --> D{"engine paused?"}
  D -->|是| A
  D -->|否| E["get_next_batch_to_run"]
  E --> F{"batch exists?"}
  F -->|是| G["cur_batch = batch"]
  G --> H["run_batch"]
  H --> I["process_batch_result"]
  I --> J["last_batch = batch"]
  J --> A
  F -->|否| K["on_idle"]
  K --> J
```

## 4. Overlap 事件循环

对应源码：

- `scheduler.py:Scheduler.event_loop_overlap`
- `scheduler.py:Scheduler.is_disable_overlap_for_batch`
- `scheduler.py:Scheduler.launch_batch_sample_if_needed`

```mermaid
flowchart TD
  A["while True"] --> B["recv_requests + process_input_requests"]
  B --> C{"engine paused?"}
  C -->|是| A
  C -->|否| D["必要时等待 forward_stream"]
  D --> E["get_next_batch_to_run"]
  E --> F["判断是否临时关闭 overlap"]
  F --> G{"需要先处理上轮结果?"}
  G -->|是| H["pop_and_process: process_batch_result(last_batch)"]
  G -->|否| I["跳过"]
  H --> J["run_batch 当前 batch"]
  I --> J
  J --> K["result_queue.append(batch.copy, result)"]
  K --> L{"可处理上一轮结果?"}
  L -->|是| M["pop_and_process"]
  L -->|否| N["保留 result_queue"]
  M --> O["launch_batch_sample_if_needed"]
  N --> O
  O --> P["last_batch = batch"]
  P --> A
```

## 5. 输入请求到 waiting_queue

对应源码：

- `scheduler.py:Scheduler.process_input_requests`
- `scheduler.py:Scheduler.init_request_dispatcher`
- `scheduler.py:Scheduler.handle_generate_request`
- `scheduler.py:Scheduler._add_request_to_queue`

```mermaid
flowchart TD
  A["recv_requests 得到 recv_req"] --> B["process_input_requests"]
  B --> C{"健康检查且 GPU 忙?"}
  C -->|是| D["暂存 http_worker_ipc 到 return_health_check_ipcs"]
  C -->|否| E["TypeBasedDispatcher 按类型分发"]
  E --> F{"TokenizedGenerateReqInput?"}
  F -->|是| G["handle_generate_request"]
  G --> H["创建 Req / session req"]
  H --> I["处理 input_embeds / 多模态 / mrope / logprob / grammar"]
  I --> J{"grammar 需要等待?"}
  J -->|是| K["进入 grammar_queue"]
  J -->|否| L["_add_request_to_queue"]
  L --> M{"disaggregation_mode"}
  M -->|NULL| N["append waiting_queue"]
  M -->|PREFILL| O["disagg_prefill_bootstrap_queue.add"]
  M -->|DECODE| P["disagg_decode_prealloc_queue.add"]
```

## 6. get_next_batch_to_run 决策

对应源码：

- `scheduler.py:Scheduler.get_next_batch_to_run`
- `scheduler.py:Scheduler.get_new_batch_prefill`
- `scheduler.py:Scheduler._get_new_batch_prefill_raw`
- `scheduler.py:Scheduler.update_running_batch`

```mermaid
flowchart TD
  A["get_next_batch_to_run"] --> B["检查 waiting/running timeout"]
  B --> C["处理 chunked_req / dllm / hisparse 的特殊状态"]
  C --> D{"last_batch 是 extend?"}
  D -->|是| E["filter last_batch 并 merge 到 running_batch"]
  D -->|否| F["跳过"]
  E --> G["清理 prefill-only running_batch"]
  F --> G
  G --> H["get_new_batch_prefill"]
  H --> I{"new prefill batch?"}
  I -->|是| J["返回 prefill batch"]
  I -->|否| K{"running_batch 可 decode?"}
  K -->|是| L["update_running_batch"]
  L --> M["返回 decode batch"]
  K -->|否| N["返回 None"]
```

## 7. Prefill batch 构造

对应源码：`scheduler.py:Scheduler._get_new_batch_prefill_raw`

```mermaid
flowchart TD
  A["_get_new_batch_prefill_raw"] --> B["grammar ready 请求重新入队"]
  B --> C["HiCache 异步事件检查"]
  C --> D{"running_batch full 且没有 chunked_req?"}
  D -->|是| Z["return None"]
  D -->|否| E["policy.calc_priority(waiting_queue, running_batch)"]
  E --> F["创建 PrefillAdder"]
  F --> G{"已有 chunked_req?"}
  G -->|是| H["adder.add_chunked_req"]
  G -->|否| I["遍历 waiting_queue"]
  H --> I
  I --> J["LoRA / request slot / KV token / HiCache prefetch 检查"]
  J --> K["req.init_next_round_input"]
  K --> L["adder.add_one_req"]
  L --> M{"还能继续加请求?"}
  M -->|是| I
  M -->|否| N["更新 waiting_queue / preempt_list / chunked_req"]
  N --> O["ScheduleBatch.init_new"]
  O --> P["prepare_for_extend"]
  P --> Q{"mixed chunked prefill?"}
  Q -->|是| R["mix_with_running"]
  Q -->|否| S["返回 new_batch"]
  R --> S
```

## 8. run_batch 与结果处理

对应源码：

- `scheduler.py:Scheduler.run_batch`
- `scheduler.py:Scheduler.process_batch_result`
- `scheduler_components/batch_result_processor.py:process_batch_result_prefill`
- `scheduler_components/batch_result_processor.py:process_batch_result_decode`

```mermaid
flowchart TD
  A["run_batch"] --> B["forward_ct += 1 / profiler"]
  B --> C{"generation or embedding?"}
  C -->|generation| D{"enable_overlap?"}
  D -->|是| E["resolve future_map / forward_stream / isolation"]
  E --> F["model_worker.forward_batch_generation"]
  F --> G["stash next token / copy_to_cpu or delayed sample"]
  D -->|否| H["resolve_forward_inputs"]
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

