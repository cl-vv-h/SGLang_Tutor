# Scheduler 教学版代码导读

说明：本文不是修改后的源码，而是针对 `python/sglang/srt/managers/scheduler.py` 的教学版代码骨架。代码块保留关键控制流，并加入中文注释解释每段逻辑。阅读时建议同时打开原文件，按“文件 + 函数名”定位。

完整的带中文注释代码副本见 [05-scheduler-annotated-cn.py](./05-scheduler-annotated-cn.py)。该文件保留原始 `scheduler.py` 的完整代码结构，并在类、函数和关键状态转换前插入中文注释，仅用于学习对照。

## 1. 进程入口：run_scheduler_process

定位：`python/sglang/srt/managers/scheduler.py:run_scheduler_process`

```python
def run_scheduler_process(...):
    # 1. 加载插件。SGLang 允许插件覆盖 Scheduler 或相关依赖，
    #    所以真正创建 Scheduler 之前先执行 hook。
    load_plugins()

    # 2. 配置当前 scheduler 进程：
    #    - 进程名
    #    - 日志前缀，例如 DP/TP/PP rank
    #    - faulthandler
    #    - CPU affinity / NUMA 绑定
    dp_rank = configure_scheduler_process(...)

    # 3. 可选开启 trace，把 Scheduler 线程身份写入 tracing 系统。
    if server_args.enable_trace:
        process_tracing_init(...)
        trace_set_thread_info(...)

    try:
        # 4. 初始化 Scheduler。这里会创建模型 worker、KV cache、IPC、
        #    grammar、LoRA、metrics、调度策略等几乎所有运行时组件。
        scheduler = Scheduler(...)

        # 5. 把初始化信息返回给父进程，告诉上层 server 当前 rank 已就绪。
        pipe_writer.send(scheduler.get_init_info())

        # 6. 进入阻塞事件循环。正常服务期间基本不会返回。
        scheduler.run_event_loop()

    except Exception:
        # 7. 任意未捕获异常都通知父进程退出，避免某个 rank 挂掉后其他 rank 假活。
        parent_process.send_signal(signal.SIGQUIT)
```

这一段的重点是：Scheduler 是多进程架构里的一个 worker 进程，不是普通类调用。它初始化成功后，通过 event loop 长期驻留。

## 2. 初始化：Scheduler.__init__

定位：`python/sglang/srt/managers/scheduler.py:Scheduler.__init__`

```python
class Scheduler(...):
    def __init__(self, server_args, port_args, gpu_id, tp_rank, ...):
        # 1. 基础运行标记。is_initializing 用于区分启动阶段和正常服务阶段。
        self.is_initializing = True
        self.forward_ct = 0
        self.cur_batch = None

        # 2. 保存启动参数，并提取调度相关开关。
        self.server_args = server_args
        self.schedule_policy = server_args.schedule_policy
        self.enable_overlap = server_args.enable_overlap_schedule
        self.enable_lora = server_args.lora_paths is not None
        self.stream_interval = server_args.stream_interval

        # 3. 计算分布式并行信息。
        #    Scheduler 需要知道自己处于 TP/DP/PP/EP/ATTN_CP 的哪个 rank，
        #    因为 batch 调度和通信方式依赖这些拓扑。
        self.ps = ParallelState(...)

        # 4. 初始化模型配置、metrics、IPC、tokenizer、worker。
        self.init_model_config()
        self.init_metrics_collector()
        self.init_ipc()
        self.init_tokenizer()
        self.init_model_worker()

        # 5. 构建 KV cache 相关池。
        #    req_to_token_pool: request slot -> token slot 映射。
        #    token_to_kv_pool_allocator: token slot -> KV cache 内存。
        #    tree_cache: prefix/radix/hierarchical cache。
        kv_cache = kv_cache_builder.build_kv_cache(...)
        self.req_to_token_pool = kv_cache.req_to_token_pool
        self.token_to_kv_pool_allocator = kv_cache.token_to_kv_pool_allocator
        self.tree_cache = kv_cache.tree_cache

        # 6. 初始化运行状态：waiting_queue、running_batch、last_batch 等。
        self.init_running_status()

        # 7. 初始化高级调度特性。
        self.init_chunked_prefill()
        self.init_schedule_policy()
        self.init_disaggregation()
        self.init_overlap_schedule()
        self.init_grammar_manager()

        # 8. 初始化 request dispatcher。
        #    后续 process_input_requests 不直接写 if/elif，而是按请求类型派发。
        self.init_request_dispatcher()

        self.is_initializing = False
```

可以把 `__init__` 理解为“把服务端运行时装配起来”。它本身不处理请求，但决定了后续事件循环能不能访问所有必要组件。

## 3. 核心状态：init_running_status

定位：`python/sglang/srt/managers/scheduler.py:Scheduler.init_running_status`

```python
def init_running_status(self):
    # 等待做 prefill 的请求。新 generate 请求通常最终进入这里。
    self.waiting_queue: List[Req] = []

    # 正在 decode 的 batch，或者已经完成 prefill、准备进入 decode 的请求集合。
    self.running_batch: ScheduleBatch = ScheduleBatch(reqs=[], batch_is_full=False)

    # 当前轮 batch 与上一轮 batch。
    # last_batch 对 prefill 很关键：prefill 结束后要在下一轮合并进 running_batch。
    self.cur_batch = None
    self.last_batch = None

    # 忙碌时收到 health check，不立刻走完整生成链路，而是暂存 IPC。
    self.return_health_check_ipcs = deque()

    # pause_generation/continue_generation 使用的总开关。
    self._engine_paused = False
```

这组状态是 Scheduler 的心脏。理解后面的函数时，优先观察它们如何改变 `waiting_queue`、`running_batch`、`cur_batch`、`last_batch`。

补充定位：

- `chunked_req` 初始化于 `Scheduler.init_chunked_prefill`，表示跨多轮 prefill 的大请求。
- `result_queue` 初始化于 `Scheduler.init_overlap`，并在 `event_loop_overlap` 中使用，用来暂存已经 launch 但尚未处理的 forward 结果。

## 4. 请求分发器：init_request_dispatcher

定位：`python/sglang/srt/managers/scheduler.py:Scheduler.init_request_dispatcher`

```python
def init_request_dispatcher(self):
    self._request_dispatcher = TypeBasedDispatcher(
        [
            # 生成请求：最重要路径，最终会创建 Req 并进入 waiting_queue。
            (TokenizedGenerateReqInput, self.handle_generate_request),

            # embedding 请求走类似路径，但 forward mode 和结果处理不同。
            (TokenizedEmbeddingReqInput, self.handle_embedding_request),

            # batch 请求会拆成多个单请求处理。
            (BatchTokenizedGenerateReqInput, self.handle_batch_generate_request),

            # 控制类请求：flush cache、abort、session、权重更新、profile、LoRA 等。
            (AbortReq, self.abort_request),
            (FlushCacheReq, self.flush_cache_wrapped),
            (RpcReqInput, self.handle_rpc_request),
            (LoadLoRAAdapterReqInput, self.load_lora_adapter),
            (PauseGenerationReqInput, self.pause_generation),
            (ContinueGenerationReqInput, self.continue_generation),
        ]
    )
```

这里的设计点是：事件循环只需要“收到请求 -> dispatcher”，不用把所有请求类型的 if/elif 混在主循环里。

## 5. 普通事件循环：event_loop_normal

定位：`python/sglang/srt/managers/scheduler.py:Scheduler.event_loop_normal`

```python
def event_loop_normal(self):
    while True:
        # 1. 从 IPC 中一次取出若干请求。
        recv_reqs = self.request_receiver.recv_requests()

        # 2. 把请求转换为内部状态变化：
        #    - generate -> Req -> waiting_queue
        #    - abort -> 标记或移除请求
        #    - flush/rpc/session/profile 等控制请求 -> 对应 handler
        self.process_input_requests(recv_reqs)

        # 3. pause 状态下不再推进 GPU forward。
        if self._engine_paused:
            continue

        # 4. 决定下一轮 GPU forward 跑什么：
        #    - 优先 prefill 新请求
        #    - 没有 prefill 时 decode running_batch
        #    - 都没有则返回 None
        batch = self.get_next_batch_to_run()
        self.cur_batch = batch

        if batch:
            # 5. 真正调用模型 worker 做 forward。
            result = self.run_batch(batch)

            # 6. 根据 prefill/decode/embedding 等模式处理输出和状态。
            self.process_batch_result(batch, result)
        else:
            # 7. 完全没活时做 cache 检查、metrics、睡眠等维护。
            self.on_idle()

        # 8. 保存本轮 batch，下一轮可能需要合并 prefill 结果。
        self.last_batch = batch
```

普通 loop 的控制流非常直接，是理解 Scheduler 的最佳入口。

## 6. Overlap 事件循环：event_loop_overlap

定位：`python/sglang/srt/managers/scheduler.py:Scheduler.event_loop_overlap`

```python
def event_loop_overlap(self):
    while True:
        recv_reqs = self.request_receiver.recv_requests()
        self.process_input_requests(recv_reqs)

        if self._engine_paused:
            continue

        # 1. schedule stream 与 forward stream 做必要同步。
        #    overlap 的核心是 CPU 侧调度和 GPU forward 尽量并行，
        #    但涉及 WAR 依赖时必须等。
        self.schedule_stream.wait_stream(self.forward_stream)

        # 2. 先选出当前要跑的 batch。
        batch = self.get_next_batch_to_run()
        self.cur_batch = batch

        # 3. 某些场景不能 overlap，例如连续 prefill 或 spec+grammar decode。
        disable_overlap_for_batch = self.is_disable_overlap_for_batch(batch)

        if disable_overlap_for_batch:
            # 如果不能 overlap，就先处理上一轮结果，避免状态交错。
            pop_and_process()

        if batch:
            # 4. 发起本轮 forward，但不一定立刻处理结果。
            batch_result = self.run_batch(batch)
            self.result_queue.append((batch.copy(), batch_result))
        else:
            batch_result = None

        if self.last_batch and not disable_overlap_for_batch:
            # 5. 当前 forward 发起后，再处理上一轮结果，
            #    达成“上一轮 CPU 结果处理”和“本轮 GPU forward”的重叠。
            pop_and_process()
        elif batch is None:
            self.on_idle()

        # 6. 延迟采样必须在上一轮结果处理之后执行，
        #    因为 grammar 等状态可能依赖上一轮结果。
        self.launch_batch_sample_if_needed(batch_result)

        self.last_batch = batch
```

Overlap 模式最难的地方不是多了一个队列，而是状态生命周期：`batch.copy()`、`future_map`、CUDA stream event 都是在保证 forward 还没完全消费完的张量不会被 Scheduler 提前改掉或释放。

## 7. 输入处理：process_input_requests

定位：`python/sglang/srt/managers/scheduler.py:Scheduler.process_input_requests`

```python
def process_input_requests(self, recv_reqs: List):
    # 1. 清理过期 session。
    self.session_controller.maybe_reap(time.monotonic())

    for recv_req in recv_reqs:
        # 2. 健康检查特殊处理：
        #    如果当前不空闲，说明 GPU 仍在推进请求，可以延迟返回轻量信号，
        #    避免 health check 被长 prompt prefill 阻塞。
        if is_health_check_generate_req(recv_req) and not self.is_fully_idle(...):
            self.return_health_check_ipcs.append(recv_req.http_worker_ipc)
            continue

        # 3. 按请求类型分发到 handler。
        output = self._request_dispatcher(recv_req)

        # 4. 某些 handler 会立即返回输出，例如内部状态查询、RPC、错误响应。
        if output is not None:
            if not isinstance(output, RpcReqOutput):
                self.ipc_channels.send_to_tokenizer.send_output(output, recv_req)
            else:
                self.ipc_channels.recv_from_rpc.send_pyobj(output)

    # 5. 检查延迟 flush 和外部 corpus 异步加载。
    self.flush_wrapper.check_pending()
    self.external_corpus_manager.check_pending_load()
```

这一层只负责“把输入请求转换成 Scheduler 内部动作”，不直接做 batch 调度。

## 8. 生成请求：handle_generate_request

定位：`python/sglang/srt/managers/scheduler.py:Scheduler.handle_generate_request`

```python
def handle_generate_request(self, recv_req):
    # 1. session 请求和普通请求分开处理。
    #    session 存在时，Req 会继承 session 的上下文状态。
    if recv_req.session_params is None:
        req = Req(...)
    else:
        session = self.session_controller.get_session(...)
        req = session.create_req(...) if session else aborted_req(...)

    # 2. input_embeds 没有真实 token id 时，用长度构造 fake input ids，
    #    后续调度仍然需要按 token 数估算 KV cache。
    if recv_req.input_embeds is not None:
        recv_req.input_ids = fake_input_ids_by_length(...)

    # 3. 构造 Req：这里会把采样参数、logprob、LoRA、多模态、disagg、
    #    bootstrap、HTTP IPC 等请求属性全部挂到运行时对象上。
    req = Req(
        rid=recv_req.rid,
        origin_input_ids=recv_req.input_ids,
        sampling_params=recv_req.sampling_params,
        stream=recv_req.stream,
        lora_id=recv_req.lora_id,
        ...
    )

    # 4. 多模态请求需要把 image/video/audio 占位 token 展开，
    #    并更新 mrope position、mm offsets 等信息。
    if recv_req.mm_inputs is not None:
        mm_inputs = self._get_multimodal_inputs(recv_req)
        req.extend_image_inputs(mm_inputs)

    # 5. 初始化最大生成长度，并校验 prompt 长度是否超过模型/服务限制。
    self.init_req_max_new_tokens(req)
    error_msg = self.validate_input_length(req)
    if error_msg:
        self._add_request_to_queue(req)
        return

    # 6. logprob / routed experts / grammar 等附加约束校验。
    #    grammar 如果还没 ready，请求先进入 grammar_manager，不直接入 waiting_queue。
    added_to_grammar_queue = self.grammar_manager.process_req_with_grammar(req)
    if not added_to_grammar_queue:
        self._add_request_to_queue(req)
```

`handle_generate_request` 的本质是“把外部 tokenized request 变成内部 `Req`”。真正的 batch 选择还没发生。

## 9. 入队：_add_request_to_queue

定位：`python/sglang/srt/managers/scheduler.py:Scheduler._add_request_to_queue`

```python
def _add_request_to_queue(self, req: Req, is_retracted: bool = False):
    # 1. 设置或校验请求优先级。被 retract 的请求通常需要保留/调整调度位置。
    self._set_or_validate_priority(req, is_retracted)

    if self.disaggregation_mode == DisaggregationMode.NULL:
        # 2. 普通模式：检查队列长度，必要时触发 HiCache 预取，然后进入 waiting_queue。
        self._abort_on_queued_limit(req)
        self._prefetch_kvcache(req)
        self.waiting_queue.append(req)
        req.set_queue_time(...)

    elif self.disaggregation_mode == DisaggregationMode.PREFILL:
        # 3. PD 分离的 prefill worker：请求先进入 bootstrap 队列，
        #    等待和 decode 侧建立 KV 传输关系。
        self._prefetch_kvcache(req)
        self.disagg_prefill_bootstrap_queue.add(req)

    elif self.disaggregation_mode == DisaggregationMode.DECODE:
        # 4. PD 分离的 decode worker：先做 KV 预分配，
        #    等 prefill 侧把 KV cache 传过来。
        self.disagg_decode_prealloc_queue.add(req)
```

这个函数是所有生成请求进入 Scheduler 调度系统的门口。

## 10. 选择下一批：get_next_batch_to_run

定位：`python/sglang/srt/managers/scheduler.py:Scheduler.get_next_batch_to_run`

```python
def get_next_batch_to_run(self):
    # 1. 超时请求先 abort，避免无期限占用队列或 running batch。
    self._abort_on_waiting_timeout()
    self._abort_on_running_timeout()

    # 2. 处理 chunked_req、dLLM、HiSparse 等特殊状态。
    #    这些请求可能跨多轮 prefill，不能被普通 merge 逻辑误处理。
    chunked_req_to_exclude = set()
    if self.chunked_req is not None:
        chunked_req_to_exclude.add(self.chunked_req)
        self.stash_chunked_request(self.chunked_req)

    # 3. 如果上一轮 batch 是 extend/prefill，
    #    prefill 完成且未结束的请求应该并入 running_batch，后续进入 decode。
    if self.last_batch and self.last_batch.forward_mode.is_extend():
        self.last_batch.filter_batch(chunked_req_to_exclude=list(...))
        if not self.last_batch.is_empty():
            self.running_batch.merge_batch(self.last_batch)

    # 4. prefill-only batch 不需要 decode，及时清理完成请求。
    if self.running_batch.is_prefill_only:
        self.running_batch.filter_batch()

    # 5. 先尝试构造新的 prefill batch。
    #    这体现 continuous batching：新请求可以插入到正在 decode 的服务中。
    new_batch = self.get_new_batch_prefill()

    if new_batch is not None:
        # 6. 有 prefill 就优先跑 prefill。
        ret = new_batch
    elif not self.running_batch.is_empty() and not self.running_batch.is_prefill_only:
        # 7. 没有 prefill 时，推进 running_batch decode。
        self.running_batch = self.update_running_batch(self.running_batch)
        ret = self.running_batch if not self.running_batch.is_empty() else None
    else:
        ret = None

    # 8. DP attention、ngram speculative 等附加逻辑可能包装或替换 batch。
    ret = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(ret)
    ret = self._maybe_prepare_ngram_embedding(ret)
    return ret
```

这就是 Scheduler 的核心决策函数：先合并上一轮 prefill，再尽量调度新 prefill，最后才 decode。

## 11. 构造 prefill batch：_get_new_batch_prefill_raw

定位：`python/sglang/srt/managers/scheduler.py:Scheduler._get_new_batch_prefill_raw`

```python
def _get_new_batch_prefill_raw(self, prefill_delayer_single_pass):
    # 1. grammar 请求 ready 后重新进入普通等待队列。
    if self.grammar_manager.has_waiting_grammars():
        for req in self.grammar_manager.get_ready_grammar_requests():
            self._add_request_to_queue(req)

    # 2. HiCache 需要检查异步 load/write/prefetch 事件。
    if self.enable_hierarchical_cache:
        self.tree_cache.check_hicache_events()

    # 3. 如果 running batch 已满且没有 chunked_req，就不能再加 prefill。
    if (self.running_batch.batch_is_full or len(self.waiting_queue) == 0) and self.chunked_req is None:
        return None

    # 4. 调度策略给 waiting_queue 排优先级。
    self.policy.calc_priority(self.waiting_queue, self.running_batch)

    # 5. PrefillAdder 是预算控制器：
    #    它知道 page size、KV cache 可用量、最大 prefill token、chunk size、
    #    running batch 大小、优先级抢占阈值等。
    adder = PrefillAdder(
        self.page_size,
        self.tree_cache,
        self.token_to_kv_pool_allocator,
        self.running_batch,
        self.new_token_ratio_tracker.current,
        self.max_prefill_tokens,
        chunked_prefill_size,
        ...
    )

    # 6. 如果上轮留下 chunked_req，先尝试继续加它。
    if self.chunked_req is not None:
        self.chunked_req.init_next_round_input()
        self.chunked_req = adder.add_chunked_req(self.chunked_req)

    # 7. 遍历 waiting_queue，逐个判断能否进入本轮 prefill。
    for req in self.waiting_queue:
        # LoRA 限制：同一 batch 里能加载的 adapter 数量有限。
        if self.enable_lora and not self._can_schedule_lora_req(req, running_loras):
            continue

        # request slot 或 KV token 不够时，标记 running_batch full。
        if len(adder.can_run_list) >= self.get_num_allocatable_reqs(running_bs):
            self.running_batch.batch_is_full = True

        # HiCache storage 预取没完成时，暂时跳过该请求。
        if self.enable_hicache_storage and not self.tree_cache.check_prefetch_progress(req.rid):
            continue

        # 计算本轮增量输入，做 prefix cache 匹配，准备写 KV cache。
        req.init_next_round_input(self.tree_cache)

        # 真正尝试加入 prefill batch。
        res = adder.add_one_req(req, has_chunked_req=..., truncation_align_size=...)

        # 不能继续加时退出循环。常见原因：token 预算不足、request 槽不足、
        # chunked prefill 截断、优先级抢占失败等。
        if res != AddReqResult.CONTINUE:
            break

    # 8. 没有任何请求能跑，返回 None。
    if len(adder.can_run_list) == 0:
        return None

    # 9. 从 waiting_queue 移除已经被选中的请求；
    #    被 preempt 的请求重新入队。
    self.waiting_queue = [x for x in self.waiting_queue if x not in can_run_set]
    for req in adder.preempt_list:
        self._add_request_to_queue(req)

    # 10. 记录新的 chunked_req。
    if adder.new_chunked_req is not None:
        self.chunked_req = adder.new_chunked_req

    # 11. 用 can_run_list 创建 ScheduleBatch，并准备 extend/prefill forward。
    new_batch = ScheduleBatch.init_new(...)
    new_batch.prepare_for_extend()

    # 12. mixed chunked prefill 可以把 prefill 和已有 decode 混在一个 batch 里，
    #     但受 logprob/input_embeds 等限制。
    if self.is_mixed_chunk and not self.running_batch.is_empty():
        self.running_batch.prepare_for_decode()
        new_batch.mix_with_running(self.running_batch)
        self.running_batch = ScheduleBatch(reqs=[], batch_is_full=...)

    return new_batch
```

这段是 Scheduler 最密集的地方。可以把它理解为：`SchedulePolicy` 排顺序，`PrefillAdder` 管预算，`ScheduleBatch` 承载最终 forward 输入。

## 12. 更新 decode batch：update_running_batch

定位：`python/sglang/srt/managers/scheduler.py:Scheduler.update_running_batch`

```python
def update_running_batch(self, batch: ScheduleBatch):
    # 1. 清理已经完成或被过滤的请求。
    batch.filter_batch(v1_spec_info_filtered=True)
    if batch.is_empty():
        batch.batch_is_full = False
        return batch

    # 2. 层级缓存场景下，释放已完成 write-through 节点的锁，
    #    让它们可以被驱逐，给后续调度腾空间。
    if self.enable_hierarchical_cache:
        self.tree_cache.flush_write_through_acks()

    # 3. 检查 decode 阶段是否还有足够 KV cache。
    #    decode 每生成一个 token 都需要继续写 KV cache。
    if not batch.check_decode_mem():
        # 4. 如果 KV cache 满了，撤回一部分请求。
        #    被撤回的请求释放 GPU KV cache，然后重新进入 waiting_queue，
        #    后面可再次 prefill/恢复。
        retracted_reqs, new_token_ratio, reqs_to_abort = batch.retract_decode(...)
        self.new_token_ratio_tracker.current = new_token_ratio

        for req in reqs_to_abort:
            self.ipc_channels.send_to_tokenizer.send_output(AbortReq(...), req)

        for req in retracted_reqs:
            self._add_request_to_queue(req, is_retracted=True)
    else:
        # 5. 没有内存压力时，逐步衰减 token ratio 估计。
        self.new_token_ratio_tracker.decay_step()

    # 6. batch 变小后，不再认为它是 full，允许后续 prefill 插入。
    if batch.batch_size() < initial_bs:
        batch.batch_is_full = False

    # 7. 为 decode forward 准备 input_ids、seq_lens、sampling_info 等张量。
    batch.prepare_for_decode()
    return batch
```

decode 阶段的关键风险是 KV cache 持续增长，所以这里必须有内存检查与 retraction。

## 13. 执行 batch：run_batch

定位：`python/sglang/srt/managers/scheduler.py:Scheduler.run_batch`

```python
def run_batch(self, batch: ScheduleBatch, pp_proxy_tensors=None):
    # 1. 递增 forward 计数，供 profiler、测试 retraction、metrics 使用。
    self.forward_ct += 1
    batch.forward_iter = self.forward_ct
    self.profiler_manager._profile_batch_predicate(batch)

    # 2. prebuilt batch 是 disaggregation decode 的占位/预构造路径。
    if batch.forward_mode.is_prebuilt():
        return self._run_batch_prebuilt(batch)

    if self.is_generation:
        if self.enable_overlap:
            # 3. overlap 模式：
            #    - 从 future_map 解析上一轮延迟传递的 token/seq_lens
            #    - 在 forward_stream 上运行模型
            #    - 用 isolation 保护 ScheduleBatch 字段不被 forward 中途修改后污染调度侧
            self.future_map.resolve_seq_lens_cpu(batch)
            with self.forward_stream_ctx:
                self.forward_stream.wait_stream(self.schedule_stream)
                resolve_forward_inputs(batch, self.future_map)

                with self._overlap_forward_isolation(batch):
                    batch_result = self.model_worker.forward_batch_generation(batch, ...)
                    self.future_map.publish(...)
                    self.future_map.stash(...next_token_ids...)
                    batch_result.copy_to_cpu(...)

            # 4. overlap 下下一轮 input_ids 从 future_map 取，所以这里清空 batch.input_ids。
            batch.input_ids = None

        else:
            # 5. 普通 generation 路径：
            #    解析输入 -> model worker forward -> 保存 next token -> 更新 cache。
            resolve_forward_inputs(batch, self.future_map)
            batch_result = self.model_worker.forward_batch_generation(batch, ...)
            self.future_map.stash(batch.req_pool_indices, batch_result.next_token_ids)
            self.update_cache_from_scheduler(batch, batch_result)

        # 6. logprob 处理依赖每个请求的 extend 长度。
        #    overlap 调度可能改变 req 字段，所以这里拷贝一份到 result。
        if batch.return_logprob:
            batch_result.extend_input_len_per_req = [req.extend_input_len for req in batch.reqs]

        return batch_result

    else:
        # 7. embedding/reward model 路径不生成 next_token_ids，
        #    forward 后返回 embedding 或 pooled_hidden_states。
        pooler_output = self.tp_worker.forward_batch_embedding(batch)
        return EmbeddingBatchResult(...)
```

`run_batch` 是 Scheduler 与模型执行层的边界。Scheduler 到这里已经决定了“跑什么”，worker 负责“怎么算”。

## 14. 结果处理：process_batch_result

定位：`python/sglang/srt/managers/scheduler.py:Scheduler.process_batch_result`

```python
def process_batch_result(self, batch, result):
    # 1. 发布当前负载快照，供 /get_loads、DP balancing 等使用。
    self.publish_load_snapshot(force=batch.forward_mode.is_extend())

    # 2. 根据 forward mode 分发。
    if batch.forward_mode.is_decode():
        # decode：追加新 token、判断 finish、发送流式输出、释放完成请求资源。
        self.batch_result_processor.process_batch_result_decode(batch, result)

    elif batch.forward_mode.is_extend():
        if batch.is_dllm():
            self.process_batch_result_dllm(batch, result)
        elif self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.process_batch_result_disagg_prefill(batch, result)
        else:
            # prefill：处理首 token、logprob、prefix cache 插入，
            # 未完成请求后续会被合并到 running_batch。
            self.batch_result_processor.process_batch_result_prefill(batch, result)

    elif batch.forward_mode.is_prebuilt():
        self.batch_result_processor.process_batch_result_prebuilt(batch)

    elif batch.forward_mode.is_idle():
        self.batch_result_processor.process_batch_result_idle(batch, result)

    # 3. 统一收尾：metrics、FPM、清理多模态输入、health check、device timer。
    self.metrics_reporter.log_batch_result_stats(batch, result)
    self._maybe_clear_mm_inputs(batch)
    self.maybe_send_health_check_signal()
    self.metrics_reporter.update_device_timer()
```

这层不再做调度决策，而是把 forward 结果落回请求状态和输出通道。

## 15. Abort：abort_request

定位：`python/sglang/srt/managers/scheduler.py:Scheduler.abort_request`

```python
def abort_request(self, recv_req: AbortReq):
    # 1. waiting_queue 中尚未开始的请求可以直接 pop。
    for req in matched_waiting_reqs:
        self.waiting_queue.pop(...)
        self.ipc_channels.send_to_tokenizer.send_output(AbortReq(rid=req.rid), req)
        release_kv_cache_if_needed(req)

    # 2. grammar_queue 中的请求不能简单 pop，
    #    通常设置 abort 标记，并让它走一次廉价 prefill 后清理。
    self.grammar_manager.abort_requests(recv_req)

    # 3. disaggregation 模式下，还要 abort bootstrap/prealloc/transfer/inflight 队列。
    if self.disaggregation_mode == DisaggregationMode.PREFILL:
        abort_prefill_bootstrap_and_inflight(...)
    elif self.disaggregation_mode == DisaggregationMode.DECODE:
        abort_decode_prealloc_and_transfer(...)

    # 4. running_batch 中已经在 GPU 路径上的请求不能直接移除，
    #    需要设置 req.to_finish = FINISH_ABORT()。
    #    后续 decode/prefill 结果处理会走统一清理逻辑。
    for req in running_reqs:
        if matched(req):
            req.to_finish = FINISH_ABORT()
```

Abort 有三种处理方式：未开始的直接移除，grammar/disagg 队列通知对应子系统，运行中的设置完成原因并等待统一清理。

## 16. 空闲维护：on_idle / is_fully_idle / flush_cache

定位：

- `scheduler.py:Scheduler.on_idle`
- `scheduler.py:Scheduler.is_fully_idle`
- `scheduler.py:Scheduler.flush_cache`

```python
def is_fully_idle(self, for_health_check=False):
    # 判断标准不只是 running_batch 为空。
    # 还要检查 waiting_queue、last_batch、cur_batch、result_queue、
    # PP microbatch、grammar queue、disagg 队列、HiSparse staging、HiCache 异步事件等。
    idle = self.running_batch.is_empty() and len(self.waiting_queue) == 0 and ...
    return idle

def on_idle(self):
    if not self.is_fully_idle():
        return

    # 空闲时做一致性检查，发现 pool 泄漏或 tree cache 异常要报告。
    self.invariant_checker._check_all_pools(...)
    self.invariant_checker._check_tree_cache()

    # 刷新 metrics、KV events、token ratio、device timer。
    self.metrics_reporter._maybe_log_idle_metrics()
    self.kv_events_publisher.publish_kv_events()
    self.new_token_ratio_tracker.reset()

    # 进入 idle sleep，降低空转开销。
    self.maybe_sleep_on_idle()

def flush_cache(self, empty_cache=True):
    # 只有完全空闲时才允许 flush。
    if self.is_fully_idle():
        self.cur_batch = None
        self.last_batch = None
        self.tree_cache.reset()
        self.req_to_token_pool.clear()
        self.token_to_kv_pool_allocator.clear()
        self.grammar_manager.clear()
        current_platform.empty_cache()
        return True
    return False
```

`flush_cache` 必须依赖 `is_fully_idle`，否则可能清掉仍被 running request 使用的 KV cache。
