# 详细流程图

## 1. 从 Scheduler 到模型执行

```mermaid
flowchart TB
    Scheduler["Scheduler 选出可运行请求"] --> ScheduleBatch["构造 / 更新 ScheduleBatch"]
    ScheduleBatch --> WorkerCall["TpModelWorker.forward_batch_generation(batch)"]
    WorkerCall --> SetConsumer["设置 HiCache consumer<br/>可选"]
    SetConsumer --> FB["ForwardBatch.init_new(batch, model_runner)"]
    FB --> DLLM{"是否 dLLM worker?"}
    DLLM -- 是 --> DLLMPath["_forward_batch_generation_dllm()"]
    DLLM -- 否 --> PP{"当前 PP rank 是否最后一级?"}
    PP -- 否 --> PPForward["ModelRunner.forward()<br/>返回 PPProxyTensors"]
    PPForward --> PPResult["GenerationBatchResult<br/>只携带 hidden state proxy"]
    PP -- 是 --> MRForward["ModelRunner.forward()"]
    MRForward --> Verify{"is_verify?"}
    Verify -- 是 --> ReturnLogits["直接返回 logits/hidden states<br/>不采样"]
    Verify -- 否 --> Sampling{"是否 prefill-only?"}
    Sampling -- 否 --> Sample["ModelRunner.sample(...)"]
    Sampling -- 是 --> Logprob["compute_logprobs_only 或 dummy token"]
    Sample --> Result["GenerationBatchResult"]
    Logprob --> Result
```

关键代码段：

- `TpModelWorker.forward_batch_generation()`：生成入口。
- `ForwardBatch.init_new(batch, self.model_runner)`：调度 batch 到模型 batch 的转换。
- `self.pp_group.is_last_rank`：区分 PP 中间级和最后一级。
- `self.model_runner.sample(...)`：只有最后一级且需要生成 token 时才采样。

## 2. TpModelWorker 初始化

```mermaid
flowchart TB
    Start["TpModelWorker.__init__"] --> Save["保存 server_args、rank、device、port 等"]
    Save --> Config["_init_model_config()"]
    Config --> Runner["_init_model_runner()"]
    Runner --> MultiEagle{"是否 multi-layer EAGLE?"}
    MultiEagle -- 是 --> CreateMany["创建多个 draft ModelRunner"]
    MultiEagle -- 否 --> Continue["继续"]
    CreateMany --> Continue
    Continue --> DLLM["_init_dllm_algorithm()"]
    DLLM --> Tokenizer["get_tokenizer(...) / get_processor(...)"]
    Tokenizer --> Groups["get_pp_group() / get_world_group()"]
    Groups --> Capacity["读取 max_total_num_tokens / max_running_requests"]
    Capacity --> Seed["同步 random seed"]
    Seed --> Ready["worker ready"]
```

`TpModelWorker` 初始化期间会立即创建 `ModelRunner`。因此模型加载、显存池、attention backend 等昂贵初始化，大多是在 `ModelRunner.__init__()` 和 `ModelRunner.initialize()` 中发生的。

## 3. ModelRunner 初始化主流程

```mermaid
flowchart TB
    Init["ModelRunner.__init__"] --> SaveArgs["保存模型路径、rank、device、dtype、并行拓扑、spec 配置"]
    SaveArgs --> Dist["init_torch_distributed()"]
    Dist --> Memory0["记录加载前可用显存"]
    Memory0 --> Initialize["initialize(pre_model_load_memory)"]
    Initialize --> LoadModel["load_model()"]
    LoadModel --> MoE["_prepare_moe_topk()"]
    MoE --> Layers["计算 start_layer / end_layer / effective layers"]
    Layers --> KVType["configure_kv_cache_dtype()"]
    KVType --> Pool["init_memory_pool()"]
    Pool --> Aux["初始化 ngram、HiSparse、hidden-state capture 等可选模块"]
    Aux --> Backend["init_attention_backend()"]
    Backend --> Warmup["kernel_warmup()"]
    Warmup --> Graph["init_device_graphs()"]
    Graph --> Piecewise["init_piecewise_cuda_graphs()"]
    Piecewise --> Ready["ModelRunner ready"]
```

关键代码段：

- `ModelRunner.__init__()`：收集运行时配置并触发初始化。
- `init_torch_distributed()`：初始化通信组。
- `initialize()`：主体初始化编排。
- `load_model()`：加载权重和模型对象。
- `init_memory_pool()`：来自 `ModelRunnerKVCacheMixin`，建立 KV cache 与 token pool。
- `init_attention_backend()`：建立 prefill/decode attention backend。
- `init_device_graphs()`：捕获 decode graph。

## 4. 分布式初始化流程

```mermaid
flowchart TB
    DistStart["init_torch_distributed()"] --> Device["设置当前 device / gpu_id"]
    Device --> Backend["选择 distributed backend"]
    Backend --> Mem["读取加载前可用显存"]
    Mem --> InitEnv["init_distributed_environment(...)"]
    InitEnv --> Parallel["initialize_model_parallel(...)"]
    Parallel --> DPAttn["initialize_dp_attention(...)"]
    DPAttn --> Groups["保存 tp_group / pp_group / attention_tp_group"]
    Groups --> Balance["检查 TP rank 间显存是否平衡"]
    Balance --> Return["返回 pre_model_load_memory"]
```

这一步决定当前进程在所有并行维度中的位置。后续 `ModelRunner.forward()` 能否走 PP proxy、attention TP scatter/gather、MoE EP、DP attention，都依赖这里建立的通信组。

## 5. forward() 到 _forward_raw() 的分发

```mermaid
flowchart TB
    Forward["ModelRunner.forward(forward_batch)"] --> Meta["profiling / canary / expert distribution recorder"]
    Meta --> Raw["_forward_raw(forward_batch)"]
    Raw --> Context["建立 ForwardContext(attn_backend)"]
    Context --> GraphCheck{"graph_runner 可回放?"}
    GraphCheck -- 是 --> Replay["graph_runner.replay(forward_batch)"]
    GraphCheck -- 否 --> Mode{"ForwardMode"}
    Mode -- DECODE --> Decode["forward_decode()"]
    Mode -- EXTEND / TARGET_VERIFY --> Extend["forward_extend()"]
    Mode -- SPLIT_PREFILL --> Split["forward_split_prefill()"]
    Mode -- IDLE --> Idle["forward_idle()"]
    Decode --> Output["ModelRunnerOutput"]
    Extend --> Output
    Split --> Output
    Idle --> Output
    Replay --> Output
    Output --> Metrics["追加指标、EPLB、debug dump、错误恢复"]
```

`forward()` 更像“外壳”，负责观测、容错和平衡逻辑。真正选择 decode/prefill/idle/split 的地方是 `_forward_raw()`。

## 6. decode 路径

```mermaid
flowchart TB
    Decode["forward_decode()"] --> Meta["准备模型特定 metadata"]
    Meta --> Backend{"是否 PDMux?"}
    Backend -- 是 --> DecodeBackend["decode_attn_backend.init_forward_metadata()"]
    Backend -- 否 --> MainBackend["attn_backend.init_forward_metadata()"]
    DecodeBackend --> Call["model.forward(input_ids, positions, forward_batch)"]
    MainBackend --> Call
    Call --> Logits["返回 logits / hidden states"]
```

decode 通常每个请求推进一个 token，形状更稳定，所以最容易被 `init_device_graphs()` 捕获并在 `_forward_raw()` 中回放。

## 7. extend / prefill 路径

```mermaid
flowchart TB
    Extend["forward_extend()"] --> Kwargs["准备 PP proxy、input_embeds、embedding mode 等 kwargs"]
    Kwargs --> Piecewise{"piecewise graph 可用?"}
    Piecewise -- 是 --> PieceReplay["piecewise_cuda_graph_runner.replay()"]
    Piecewise -- 否 --> Meta["prepare_forward_extend_metadata()"]
    Meta --> Attn["attn_backend.init_forward_metadata()"]
    Attn --> Call["model.forward(input_ids, positions, forward_batch, **kwargs)"]
    PieceReplay --> Output["logits / hidden states"]
    Call --> Output
```

extend/prefill 处理 prompt token 或新扩展 token，token 数和 prefix 长度更动态，所以通常比 decode 更难完全 graph 化。

## 8. split prefill 路径

```mermaid
flowchart TB
    Split["TpModelWorker.forward_batch_split_prefill()"] --> FB{"是否已有 forward_batch?"}
    FB -- 否 --> NewFB["ForwardBatch.init_new(batch, model_runner)"]
    FB -- 是 --> Reuse["复用已有 ForwardBatch"]
    NewFB --> MR["ModelRunner.forward(split_forward_count=...)"]
    Reuse --> MR
    MR --> RunnerSplit["ModelRunner.forward_split_prefill()"]
    RunnerSplit --> Range["计算 split_index / next_split_index"]
    Range --> Model["model.forward_split_prefill(...)"]
    Model --> MaybeSample{"是否产生 logits?"}
    MaybeSample -- 是 --> Sample["ModelRunner.sample(...)"]
    MaybeSample -- 否 --> Continue["继续下个 split"]
```

split prefill 的核心是把一次长 prefill 拆成多个片段，让显存压力和单步延迟更可控。

## 9. sampling 流程

```mermaid
flowchart TB
    Sample["ModelRunner.sample(logits_output, forward_batch)"] --> Pre["_preprocess_logits()"]
    Pre --> Pos{"decode 还是 prefill?"}
    Pos -- decode --> DecodePos["positions = arange(batch_size)"]
    Pos -- prefill --> PrefillPos["positions = seq_lens - 1"]
    DecodePos --> Sampler["sampler(logits, sampling_info, positions)"]
    PrefillPos --> Sampler
    Sampler --> Update["maybe_update_ngram_token_table()"]
    Update --> Out["SampleOutput"]
```

`sample()` 不只是从 logits 里取 token。它还会处理 grammar、logit bias、temperature/top-p/top-k、return_logprob，以及 ngram embedding token table 的维护。
