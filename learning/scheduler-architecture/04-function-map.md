# Scheduler 函数地图

本文件按“入口 -> 输入 -> 调度 -> 执行 -> 结果 -> 控制”整理 `scheduler.py`。行号来自当前仓库快照，用于辅助定位；更稳定的定位方式是文件名 + 函数名。

## 进程与事件循环

| 函数 | 位置 | 做什么 | 主要下一跳 |
| --- | --- | --- | --- |
| `run_scheduler_process` | `scheduler.py:3951` | Scheduler 子进程入口，加载插件、配置进程、创建 Scheduler、通知父进程、进入事件循环 | `configure_scheduler_process`, `Scheduler.__init__`, `run_event_loop` |
| `configure_scheduler_process` | `scheduler.py:3894` | 设置进程名、日志、faulthandler、CPU affinity、NUMA 绑定 | 返回 `dp_rank` |
| `Scheduler.__init__` | `scheduler.py:296` | 装配运行时：并行状态、模型配置、IPC、worker、KV cache、调度策略、overlap、grammar 等 | `init_running_status`, `init_schedule_policy`, `init_request_dispatcher` |
| `run_event_loop` | `scheduler.py:1404` | 创建 schedule stream，并把控制权交给具体 event loop | `dispatch_event_loop` |
| `dispatch_event_loop` | `scheduler.py:3863` | 根据 disaggregation、PP、overlap、pdmux 选择事件循环实现 | `event_loop_normal`, `event_loop_overlap`, PP/disagg loop |
| `event_loop_normal` | `scheduler.py:1425` | 非 overlap 主循环：收请求、选 batch、forward、处理结果 | `process_input_requests`, `get_next_batch_to_run`, `run_batch`, `process_batch_result` |
| `event_loop_overlap` | `scheduler.py:1452` | overlap 主循环：当前 batch forward 与上一 batch 结果处理重叠 | `is_disable_overlap_for_batch`, `run_batch`, `launch_batch_sample_if_needed` |

## 输入与入队

| 函数 | 位置 | 改变的核心状态 | 说明 |
| --- | --- | --- | --- |
| `init_request_dispatcher` | `scheduler.py:1279` | `self._request_dispatcher` | 建立请求类型到 handler 的映射 |
| `process_input_requests` | `scheduler.py:1543` | `waiting_queue`, `return_health_check_ipcs`, 控制状态 | 对一批 IPC 请求逐个分发，立即返回控制请求输出 |
| `handle_generate_request` | `scheduler.py:1898` | 可能创建 `Req` 并入队 | 处理 session、input embeds、多模态、长度校验、logprob、grammar |
| `_add_request_to_queue` | `scheduler.py:2156` | `waiting_queue` 或 disagg 专用队列 | 普通模式 append waiting queue，PD 模式进入 bootstrap/prealloc 队列 |
| `abort_request` | `scheduler.py:3566` | `waiting_queue`, grammar/disagg 队列, `req.to_finish` | abort 未开始、等待中、disagg 中和 running 中的请求 |

## 调度决策

| 函数 | 位置 | 输入 | 输出 | 说明 |
| --- | --- | --- | --- | --- |
| `get_next_batch_to_run` | `scheduler.py:2404` | `waiting_queue`, `running_batch`, `last_batch`, `chunked_req` | `ScheduleBatch` 或 `None` | Scheduler 核心决策：优先 prefill，其次 decode |
| `get_new_batch_prefill` | `scheduler.py:2532` | waiting/running 状态 | prefill `ScheduleBatch` 或 `None` | 包装 prefill delayer，然后调用 raw 版本 |
| `_get_new_batch_prefill_raw` | `scheduler.py:2552` | waiting queue、KV cache、policy、chunked_req | prefill `ScheduleBatch` 或 `None` | 用 `SchedulePolicy` + `PrefillAdder` 选出本轮可 prefill 请求 |
| `update_running_batch` | `scheduler.py:2823` | `running_batch` | decode `ScheduleBatch` 或空 batch | 清理完成请求，检查 decode 内存，不足时 retract，再准备 decode 张量 |

## 执行与结果

| 函数 | 位置 | 做什么 | 关键协作者 |
| --- | --- | --- | --- |
| `run_batch` | `scheduler.py:2965` | 调用模型 worker 执行 generation/embedding forward | `model_worker`, `tp_worker`, `future_map`, CUDA streams |
| `_overlap_forward_isolation` | `scheduler.py` | overlap 下保护 `ScheduleBatch` 字段和 GPU tensor 生命周期 | `record_batch_in_overlap` |
| `launch_batch_sample_if_needed` | `scheduler.py:3136` | 处理延迟采样，通常用于 overlap + structured output 场景 | `future_map`, `delay_sample_func` |
| `process_batch_result` | `scheduler.py:3167` | 按 forward mode 分发结果处理，统一 metrics/health/cleanup | `BatchResultProcessor` |
| `on_idle` | `scheduler.py:3249` | 空闲时做内存一致性检查、metrics、KV event、sleep | `invariant_checker`, `metrics_reporter`, `idle_sleeper` |

## 控制与维护

| 函数 | 位置 | 说明 |
| --- | --- | --- |
| `is_fully_idle` | `scheduler.py:3285` | 判断 Scheduler 是否完全空闲；不仅看 running/waiting，还看 overlap result queue、grammar、disagg、HiCache 等 |
| `flush_cache` | `scheduler.py:3432` | 只有完全空闲时清空 tree cache、req/token pools、grammar、metrics 和设备 allocator cache |
| `pause_generation` | `scheduler.py:3677` | 暂停 generation；可 in-place 保留状态，也可 retract running requests |
| `continue_generation` | `scheduler.py:3739` | 解除暂停，可选执行 `torch.cuda.empty_cache` |

## 关联支撑类

| 类/函数 | 位置 | Scheduler 为什么依赖它 |
| --- | --- | --- |
| `Req` | `schedule_batch.py:641` | 单请求运行时状态，保存 token、采样参数、cache 索引、finish reason |
| `ScheduleBatch` | `schedule_batch.py:1481` | 一次 forward 的 batch 容器 |
| `ScheduleBatch.init_new` | `schedule_batch.py:1649` | 从 `Req` 列表创建 batch，并绑定 cache/pool/tree_cache |
| `ScheduleBatch.prepare_for_extend` | `schedule_batch.py:1813` | 为 prefill/extend forward 准备输入张量和元信息 |
| `ScheduleBatch.prepare_for_decode` | `schedule_batch.py:2383` | 为 decode forward 准备 next-token 输入和 seq lens |
| `ScheduleBatch.filter_batch` | `schedule_batch.py:2477` | 移除 finished/aborted/不该继续跑的请求 |
| `ScheduleBatch.merge_batch` | `schedule_batch.py:2560` | 把上一轮 prefill 完成的请求合入 running batch |
| `ScheduleBatch.check_decode_mem` | `schedule_batch.py:2261` | 判断 decode 下一步是否还有 KV cache 空间 |
| `ScheduleBatch.retract_decode` | `schedule_batch.py:2274` | 内存不足时撤回部分 decode 请求 |
| `SchedulePolicy.calc_priority` | `schedule_policy.py:162` | 给 waiting queue 排序或计算优先级 |
| `PrefillAdder` | `schedule_policy.py:405` | 负责在 token/request/cache 预算下选 prefill 请求 |
| `PrefillAdder.add_one_req` | `schedule_policy.py:828` | 尝试把单个请求加入本轮 prefill batch |
| `PrefillAdder.add_chunked_req` | `schedule_policy.py:679` | 继续调度跨轮的 chunked prefill 请求 |
| `PrefillAdder.preempt_to_schedule` | `schedule_policy.py:985` | 优先级抢占时撤回低优先级 running 请求 |
| `BatchResultProcessor.process_batch_result_prefill` | `scheduler_components/batch_result_processor.py:178` | 处理 prefill 结果、首 token、prefix cache、输出 |
| `BatchResultProcessor.process_batch_result_decode` | `scheduler_components/batch_result_processor.py:588` | 处理 decode token、finish 判断、流式输出、cache 释放 |

## 最小阅读路径

如果只想先读懂一条普通生成请求的路径，建议按下面顺序打开函数：

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

