# 第 6 讲：多进程 / 多卡 / 分布式执行

这一讲接在第 5 讲之后。前五讲已经把单个请求从 HTTP、Tokenizer、Scheduler、KV cache、ModelRunner、Speculative Decoding 串起来了；现在要回答一个更工程化的问题：

> SGLang 真正服务时，不是一个 Python 函数在跑，而是一组进程、一组 GPU rank、一组 ZMQ/Torch distributed 通信在协作。它们分别是谁？请求和 batch 如何跨进程、跨 rank 流动？

本讲目标：

- 看懂 SGLang 标准引擎由哪些进程组成：HTTP/Engine/Tokenizer、Scheduler、Detokenizer、可选 DP Controller。
- 看懂 `PortArgs` 如何定义 IPC / TCP 通道。
- 看懂 TP / PP / DP / DP attention / CP / EP / MoE DP 这些 rank 如何影响 Scheduler 和 ModelRunner。
- 看懂 rank0 Scheduler 如何收请求，再广播给同一个并行组里的其它 rank。
- 看懂多节点下为什么非 0 节点不跑 tokenizer/detokenizer。
- 看懂 DP attention 为什么需要 DataParallelController 和 MLP sync。

---

## 0. 一张总图

```mermaid
flowchart TD
  U["HTTP / OpenAI API 请求"] --> H["HTTP Server + Engine<br/>主进程"]
  H --> T["TokenizerManager<br/>主进程"]
  T -->|ZMQ PUSH| S0["Scheduler rank leader<br/>子进程"]
  S0 -->|broadcast_pyobj / p2p| S1["其它 TP / CP / PP Scheduler ranks"]
  S0 --> W0["TpModelWorker / ModelRunner<br/>GPU rank 0"]
  S1 --> W1["TpModelWorker / ModelRunner<br/>其它 GPU ranks"]
  W0 -->|token ids / embeddings| S0
  S0 -->|ZMQ PUSH| D["DetokenizerManager<br/>子进程"]
  D -->|ZMQ PUSH| T
  T --> H
  H --> U
```

如果开启 DP：

```mermaid
flowchart TD
  T["TokenizerManager"] --> C["DataParallelController"]
  C --> DP0["DP0 Scheduler group"]
  C --> DP1["DP1 Scheduler group"]
  C --> DP2["DP2 Scheduler group"]
  DP0 --> D["DetokenizerManager"]
  DP1 --> D
  DP2 --> D
```

一句话版：

> TokenizerManager 负责把外部请求变成内部 tokenized 请求；Scheduler rank leader 接收请求并在并行组内广播；每个 GPU rank 上的 Scheduler/Worker 按相同 batch 元信息执行自己那份模型分片；DetokenizerManager 再把 token 输出变成文本。

---

## 1. 关键文件跳转表

| 主题 | 文件 | 具体定位 |
|---|---|---|
| Python Engine 入口 | `python/sglang/srt/entrypoints/engine.py` | `Engine`、`_launch_subprocesses()` |
| HTTP server 入口 | `python/sglang/srt/entrypoints/http_server.py` | `launch_server()` |
| 进程启动：Scheduler | `python/sglang/srt/entrypoints/engine.py` | `_launch_scheduler_processes()` |
| 进程启动：Detokenizer | `python/sglang/srt/entrypoints/engine.py` | `_launch_detokenizer_subprocesses()` |
| 进程间通道定义 | `python/sglang/srt/server_args.py` | `PortArgs`、`PortArgs.init_new()` |
| Tokenizer 主进程 | `python/sglang/srt/managers/tokenizer_manager.py` | `TokenizerManager`、请求状态 `ReqState` |
| Detokenizer 子进程 | `python/sglang/srt/managers/detokenizer_manager.py` | `DetokenizerManager.__init__()`、`event_loop()` |
| Scheduler 子进程 | `python/sglang/srt/managers/scheduler.py` | `run_scheduler_process()`、`configure_scheduler_process()` |
| Scheduler IPC | `python/sglang/srt/managers/scheduler_components/ipc_channels.py` | `SchedulerIpcChannels.create()` |
| Scheduler 收请求 | `python/sglang/srt/managers/scheduler_components/request_receiver.py` | `SchedulerRequestReceiver.recv_requests()` |
| TP worker | `python/sglang/srt/managers/tp_worker.py` | `TpModelWorker.__init__()`、`forward_batch_generation()` |
| 分布式初始化 | `python/sglang/srt/distributed/parallel_state.py` | `init_distributed_environment()`、`initialize_model_parallel()` |
| rank 映射 | `python/sglang/srt/entrypoints/engine.py` | `_calculate_rank_ranges()`、`_compute_parallelism_ranks()` |
| DP controller | `python/sglang/srt/managers/data_parallel_controller.py` | `run_data_parallel_controller_process()`、`launch_tensor_parallel_group()` |
| DP attention | `python/sglang/srt/layers/dp_attention.py` | `compute_dp_attention_world_info()`、`set_attention_dp_info()` |
| DP Scheduler sync | `python/sglang/srt/managers/scheduler_components/dp_attn.py` | `prepare_mlp_sync_batch_raw()`、`MLPSyncBatchInfo` |
| PP Scheduler mixin | `python/sglang/srt/managers/scheduler_pp_mixin.py` | pipeline parallel event loop / microbatch 相关逻辑 |

---

## 2. 标准引擎的三个核心组件

`Engine` 的类注释已经直接说明了标准 SRT 引擎的组成：

| 组件 | 进程 | 主要职责 |
|---|---|---|
| HTTP Server / Engine | 主进程 | 提供 API、初始化引擎、维护事件循环 |
| TokenizerManager | 主进程 | tokenization、请求状态管理、把 tokenized 请求送给 Scheduler |
| Scheduler | 子进程，一个或多个 | 调度 batch、执行模型 forward、输出 token ids |
| DetokenizerManager | 子进程，一个或多个 | 把 token ids 增量 decode 成文本，返回 TokenizerManager |

```mermaid
flowchart LR
  A["HTTP Server"] --> B["Engine"]
  B --> C["TokenizerManager<br/>main process"]
  C --> D["Scheduler<br/>subprocess"]
  D --> E["DetokenizerManager<br/>subprocess"]
  E --> C
```

注意：TokenizerManager 和 Engine 在主进程里，不是独立子进程。Scheduler 和 DetokenizerManager 是 multiprocessing 子进程。

---

## 3. `Engine._launch_subprocesses()`：服务启动主线

源码定位：`python/sglang/srt/entrypoints/engine.py:Engine._launch_subprocesses()`

启动顺序：

```mermaid
flowchart TD
  A["Engine.__init__"] --> B["_launch_subprocesses"]
  B --> C["configure_logger / env / plugins / args check"]
  C --> D["PortArgs.init_new<br/>分配 IPC/TCP 地址"]
  D --> E["_launch_scheduler_processes"]
  E --> F{"node_rank >= 1?"}
  F -->|"Yes"| G["只等待 Scheduler ready<br/>不启动 tokenizer/detokenizer"]
  F -->|"No"| H["_launch_detokenizer_subprocesses"]
  H --> I["init_tokenizer_manager<br/>主进程"]
  I --> J["wait_for_scheduler_ready"]
  J --> K["把 max_req_input_len 等信息回填给 tokenizer"]
  K --> L["SubprocessWatchdog"]
```

两个细节很重要：

1. **先启动 Scheduler，再启动 Detokenizer/Tokenizer。**
   Scheduler 需要加载模型，通常最慢。Engine 会通过 pipe 等待 Scheduler ready。

2. **多节点时，只有 `node_rank == 0` 跑 tokenizer/detokenizer。**
   非 0 节点只启动本节点的 Scheduler rank，然后阻塞等待；外部请求从 0 节点进入，再通过分布式通信同步到其它节点。

---

## 4. `PortArgs`：进程间通信地址表

源码定位：`python/sglang/srt/server_args.py:PortArgs`

`PortArgs` 保存了几类通信地址：

| 字段 | 用途 |
|---|---|
| `tokenizer_ipc_name` | Detokenizer / Scheduler 结果返回 TokenizerManager |
| `scheduler_input_ipc_name` | TokenizerManager 发送请求给 Scheduler rank leader |
| `detokenizer_ipc_name` | Scheduler 发送 token id 输出给 DetokenizerManager |
| `rpc_ipc_name` | Engine 与 Scheduler 的 RPC 控制请求 |
| `metrics_ipc_name` | Scheduler 发送 metrics |
| `nccl_port` | Torch distributed / NCCL 初始化 |
| `tokenizer_worker_ipc_name` | multi-tokenizer worker 模式 |
| `load_collector_ipc_name` | DP attention TCP 模式下负载快照收集 |

### 普通模式：本机 IPC

当没有启用 DP attention 时，`PortArgs.init_new()` 会创建多个：

```text
ipc://<tempfile>
```

这些是本机 ZMQ IPC 地址，适合单机多进程。

### DP attention 模式：TCP

启用 DP attention 时，`PortArgs` 会使用 TCP 地址：

```text
tcp://host:port
```

原因是 DP attention 可能跨节点，也需要 DP Controller 给不同 DP rank 分配不同的 worker port。

---

## 5. IPC 方向：请求和结果怎么流动

```mermaid
flowchart TD
  A["TokenizerManager"] -->|"scheduler_input_ipc_name<br/>TokenizedGenerateReqInput"| B["Scheduler rank leader"]
  B -->|"detokenizer_ipc_name<br/>BatchTokenIDOutput"| C["DetokenizerManager"]
  C -->|"tokenizer_ipc_name<br/>BatchStrOutput"| A
  D["Engine RPC"] -->|"rpc_ipc_name"| B
  B -->|"tokenizer_ipc_name<br/>embedding / health / control output"| A
```

Scheduler 侧通道创建在 `SchedulerIpcChannels.create()`：

- 只有 rank leader 绑定 `recv_from_tokenizer` 和 `recv_from_rpc`。
- 非 leader rank 的 `recv_from_tokenizer` 是 `None`。
- rank leader 可以把结果发给 tokenizer 或 detokenizer。

这里的 rank leader 不是笼统的 “GPU 0”，而是满足并行维度条件的 rank：

- 普通 TP 场景：通常是 `tp_rank == 0`。
- DP attention 场景：更细地看 `attn_tp_rank == 0 and attn_cp_rank == 0`。
- PP 场景：第一个 pipeline stage 收请求，后续 stage 从前一 stage 接收对象。

---

## 6. Scheduler 进程如何启动

标准 `dp_size == 1` 时，`Engine._launch_scheduler_processes()` 会按 `pp_rank_range` 和 `tp_rank_range` 启动多个 Scheduler 子进程：

```mermaid
flowchart TD
  A["_launch_scheduler_processes"] --> B{"dp_size == 1?"}
  B -->|"Yes"| C["计算 pp_rank_range / tp_rank_range"]
  C --> D["for pp_rank"]
  D --> E["for tp_rank"]
  E --> F["计算 gpu_id"]
  F --> G["_compute_parallelism_ranks<br/>attn_cp_rank / moe_dp_rank / moe_ep_rank"]
  G --> H["mp.Process(target=run_scheduler_process)"]
  H --> I["Scheduler 子进程"]
  B -->|"No"| J["启动 DataParallelController"]
```

每个 Scheduler 进程会收到：

- `gpu_id`
- `tp_rank`
- `attn_cp_rank`
- `moe_dp_rank`
- `moe_ep_rank`
- `pp_rank`
- `dp_rank`
- `PortArgs`
- pipe writer

这些 rank 会继续传入 `Scheduler.__init__()` 和 `TpModelWorker.__init__()`。

---

## 7. Scheduler 子进程内部做什么

入口：`python/sglang/srt/managers/scheduler.py:run_scheduler_process()`

```mermaid
flowchart TD
  A["run_scheduler_process"] --> B["load_plugins"]
  B --> C["configure_scheduler_process<br/>进程名 / 日志 / CPU affinity / NUMA"]
  C --> D["Scheduler(...)"]
  D --> E["init_model_worker"]
  E --> F["TpModelWorker / draft worker"]
  F --> G["get_init_info"]
  G --> H["pipe_writer.send(init_info)"]
  H --> I["run_event_loop"]
```

`configure_scheduler_process()` 会把进程名设置成类似：

```text
sglang::scheduler_DP0_PP0_TP1_EP0
```

这不是装饰性日志。多 GPU、多 rank 场景下定位问题时，进程名和日志前缀是你判断哪个 rank 出问题的第一线索。

---

## 8. Tensor Parallel：同一个 batch，不同 rank 算不同分片

TP 场景下，多个 Scheduler 进程对应多个 GPU rank。它们需要看到一致的请求和 batch 元信息，然后各自执行模型分片。

请求同步发生在：

`python/sglang/srt/managers/scheduler_components/request_receiver.py:SchedulerRequestReceiver.recv_requests()`

```mermaid
flowchart TD
  A["rank leader 从 ZMQ 收请求"] --> B["_pull_raw_reqs"]
  B --> C{"tp_size > 1?"}
  C -->|"Yes"| D["broadcast_pyobj 到 tp_cpu_group"]
  C -->|"No"| E["直接使用本地请求"]
  D --> F["所有 TP rank 得到同一批 recv_reqs"]
  E --> F
  F --> G["process_input_requests"]
```

普通 TP 场景中：

- `tp_rank == 0` 从 TokenizerManager 收请求。
- 其它 TP rank 不直接连 tokenizer。
- rank0 通过 CPU group 把 Python 请求对象广播给其它 TP rank。
- 所有 rank 在各自进程里执行相同的调度逻辑，保持 batch 结构一致。
- 模型层的参数、attention heads、MLP 等按 TP 分片执行。

第 4 讲讲过的 `ForwardBatch`、attention backend、`tensor_model_parallel_all_reduce` 等逻辑，就运行在这组 TP rank 上。

---

## 9. Pipeline Parallel：请求在 PP stage 之间传递

PP 维度把模型层分成多个 pipeline stage。`SchedulerRequestReceiver._pull_raw_reqs()` 中可以看到：

- `pp_rank == 0` 的 leader 从 tokenizer/RPC 收请求。
- `pp_rank > 0` 的 leader 通过 `point_to_point_pyobj()` 从前一个 PP stage 接收请求对象。

```mermaid
flowchart LR
  A["TokenizerManager"] --> B["PP0 leader"]
  B --> C["PP0 TP ranks"]
  B -->|"point_to_point_pyobj"| D["PP1 leader"]
  D --> E["PP1 TP ranks"]
  D -->|"point_to_point_pyobj"| F["PP2 leader"]
```

PP 的完整 event loop 在 `scheduler_pp_mixin.py` 中。第一遍读源码时不建议直接从 PP loop 开始，因为它会把调度、microbatch、pipeline send/recv 混在一起。先理解普通 Scheduler，再看 PP 会轻松很多。

---

## 10. 分布式 group：World / TP / PP / ATTN / MoE

核心文件：`python/sglang/srt/distributed/parallel_state.py`

分布式初始化分两层：

1. `init_distributed_environment()`
   - 调 `torch.distributed.init_process_group()`
   - 创建 `_WORLD`
   - 设置 local rank
   - 可选创建 global TCPStore

2. `initialize_model_parallel()`
   - 根据 world size、TP、PP、attention DP/CP、MoE EP/DP 创建子 group

```mermaid
flowchart TD
  A["torch.distributed WORLD"] --> B["TP group"]
  A --> C["PP group"]
  B --> D["Attention TP group"]
  B --> E["Attention CP group"]
  B --> F["MoE EP group"]
  B --> G["MoE DP group"]
```

常见 group：

| group | 含义 |
|---|---|
| world group | 所有分布式 rank |
| TP group | tensor parallel rank 组 |
| PP group | pipeline parallel rank 组 |
| ATTN TP group | attention tensor parallel 组 |
| ATTN CP group | attention context parallel 组 |
| MoE EP group | expert parallel 组 |
| MoE DP group | MoE data parallel 组 |

`initialize_model_parallel()` 的注释给了一个很好例子：8 GPU、TP=2、PP=4 时，会创建 4 个 TP group 和 2 个 PP group。

---

## 11. Rank 计算：为什么一个 tp_rank 会拆出多个 rank

在 `Engine._launch_scheduler_processes()` 和 `DataParallelController.launch_tensor_parallel_group()` 中，`tp_rank` 会进一步拆成：

- `attn_cp_rank`
- `moe_dp_rank`
- `moe_ep_rank`

这是因为 “tensor parallel” 是一个总 GPU 维度，但 attention 和 MoE 可能有不同的切法。

```mermaid
flowchart TD
  A["tp_rank"] --> B["Attention hierarchy"]
  B --> C["attn_dp_rank"]
  B --> D["attn_cp_rank"]
  B --> E["attn_tp_rank"]
  A --> F["MoE hierarchy"]
  F --> G["moe_dp_rank"]
  F --> H["moe_ep_rank"]
  F --> I["moe_tp_rank"]
```

在 DP attention 场景下，`compute_dp_attention_world_info()` 会从 `tp_rank`、`tp_size`、`dp_size`、`attn_cp_size` 推导：

- attention TP size
- attention TP rank
- attention DP rank
- attention CP rank

所以不要把 `tp_rank` 简单理解成“第几张卡”。它更像一个总 rank，后续会被不同并行策略重新解释。

---

## 11.1 先统一术语：进程、GPU、rank、group 到底是什么

读 SGLang 分布式代码时，最容易混淆的是这四个词：

| 概念 | 在 SGLang 中大致对应什么 | 是否一一对应 | 说明 |
|---|---|---|---|
| OS 进程 | `Scheduler` 子进程、`DataParallelController` 子进程、`DetokenizerManager` 子进程、主进程中的 `TokenizerManager` | 不完全一一对应 | 模型计算 rank 通常是一个 Scheduler 子进程；Tokenizer/Detokenizer 不一定绑定 GPU。 |
| GPU / device | `gpu_id` / `local_rank` | 通常一个模型 rank 绑定一个 GPU | `Engine._launch_scheduler_processes()` 计算 `gpu_id`，传给 `run_scheduler_process()`、`TpModelWorker`、`ModelRunner`。 |
| global rank | torch distributed world 内的 rank | 一个模型计算进程一个 global rank | 在 `ModelRunner.init_torch_distributed()` 中，rank 计算为 `tp_size * pp_rank + tp_rank`。 |
| logical rank | `tp_rank`、`pp_rank`、`dp_rank`、`attn_cp_rank`、`moe_ep_rank` 等 | 一个进程同时拥有多个 logical rank | 同一个进程/GPU 会在 TP、PP、DP attention、MoE EP 等多个维度中分别有自己的 rank。 |
| process group | `get_tp_group()`、`get_pp_group()`、`get_attn_tp_group()` 等 | 一个进程可加入多个 group | group 是 collective 的边界；不同并行策略使用不同 group 通信。 |

一句话：

> 一个模型计算进程通常绑定一张 GPU，并在 torch distributed world 里拥有一个 global rank；但这个进程同时属于多个逻辑 group，所以它会有 `tp_rank`、`pp_rank`、`attn_tp_rank`、`attn_cp_rank`、`moe_ep_rank`、`moe_dp_rank` 等多个身份。

---

## 11.2 SGLang 一共有哪几种并行

下面这张表按“切分对象”来区分不同并行。这样比只背缩写更稳。

| 并行类型 | 缩写 | 切分对象 | 是否复制完整请求 | 是否复制完整模型 | 主要类 / 函数 | 典型通信 |
|---|---|---|---|---|---|---|
| Tensor Parallel | TP | 单层权重、attention heads、MLP hidden dim | 是，同一个请求给 TP 组内所有 rank | 否，每个 rank 持有权重分片 | `TpModelWorker`、`ModelRunner.init_torch_distributed()`、`initialize_model_parallel()`、`get_tp_group()` | all-reduce、all-gather、broadcast object |
| Pipeline Parallel | PP | transformer layers | 请求沿 stage 流动 | 否，每个 PP stage 持有部分层 | `SchedulerPPMixin`、`PPProxyTensors`、`TpModelWorker.forward_batch_generation()` | stage 间 send/recv hidden states、p2p object |
| Data Parallel | DP | 请求流量 / worker group | 否，不同请求路由到不同 DP group | 是，每个 DP group 有一份模型并行副本 | `DataParallelController`、`launch_tensor_parallel_group()`、`round_robin_scheduler()` | controller 到各 DP group 的 ZMQ / TCP 路由 |
| DP Attention | Attention DP | attention 层 token 维度和 DP 维度 | 各 DP rank 有本地 batch，但部分 MLP/attention 需同步 | 通常在 TP group 内重解释 rank | `compute_dp_attention_world_info()`、`initialize_dp_attention()`、`SchedulerDPAttnAdapter` | all-gather / all-reduce / idle batch 同步 |
| Context Parallel | CP / ATTN CP | 长上下文 token 片段 | 同一请求的上下文可被切分 | 模型权重仍按 TP/PP/EP 规则 | `attn_cp_size`、`get_attn_cp_group()`、`ForwardBatch` metadata | attention context 相关 gather/scatter |
| Expert Parallel | EP / MoE EP | MoE experts | token 会路由到不同 expert rank | dense 层未必切，expert 被切 | `moe_ep_rank`、`get_moe_ep_group()`、`ModelRunner.update_expert_location()` | token dispatch/combine、expert all-to-all |
| MoE Data Parallel | MoE DP | expert 副本 / expert 数据并行维度 | expert 维度上的数据并行 | 部分 expert 或 expert group 复制 | `moe_dp_rank`、`get_moe_dp_group()`、`moe_dp_size` | expert 侧 reduce / gather |
| Expert Load Balancing | EPLB | expert 放置和负载 | 不直接切请求 | 动态调整 expert location | `eplb`、`ModelRunner.update_expert_location()`、`maybe_recover_ep_ranks()` | expert metadata broadcast |
| Prefill/Decode Disaggregation | PD | 部署角色：prefill worker / decode worker | 请求生命周期被拆成 prefill 与 decode | 取决于部署，通常各角色有各自模型 worker | `SchedulerDisaggregationPrefillMixin`、`SchedulerDisaggregationDecodeMixin`、KV sender/receiver | KV transfer |

注意：

- **TP / PP / EP 是模型内部切分。** 它们让多个 GPU 合作计算一个模型。
- **DP 是服务实例级复制。** 它让多个 worker group 分摊不同请求。
- **DP attention 不是普通 DP。** 它是 attention/MLP 维度上的混合并行，会让同一个 TP group 内的 rank 被重新解释成 attention DP、attention CP、attention TP。
- **PD disaggregation 不是传统模型并行。** 它是把 prefill 和 decode 部署角色拆开，核心数据流是 KV cache transfer。

---

## 11.3 rank 的层级：从 world rank 到 TP / PP / Attention / MoE

`ModelRunner.init_torch_distributed()` 中初始化 torch distributed 时，global rank 是：

```text
global_rank = tp_size * pp_rank + tp_rank
world_size  = tp_size * pp_size
```

这意味着在一个 DP group 内，SGLang 的模型计算 world 可以先看成二维矩阵：

```text
                 tp_rank
              0    1    2    3
pp_rank 0    r0   r1   r2   r3
pp_rank 1    r4   r5   r6   r7
```

如果 `tp_size=4`、`pp_size=2`，则：

- `pp_rank=0,tp_rank=0` 的 global rank 是 `0`
- `pp_rank=0,tp_rank=3` 的 global rank 是 `3`
- `pp_rank=1,tp_rank=0` 的 global rank 是 `4`
- `pp_rank=1,tp_rank=3` 的 global rank 是 `7`

`initialize_model_parallel()` 会基于这个 world 创建不同 group：

```mermaid
flowchart TD
  W["WORLD<br/>tp_size * pp_size ranks"] --> TP["TP groups<br/>同一个 PP stage 内的连续 tp ranks"]
  W --> PP["PP groups<br/>同一个 tp_rank 跨不同 PP stage"]
  TP --> ATTNDP["Attention DP 维度"]
  TP --> ATTNCP["Attention CP groups"]
  TP --> ATTNTP["Attention TP groups"]
  TP --> MOEDP["MoE DP groups"]
  TP --> MOEEP["MoE EP groups"]
  TP --> MOETP["MoE TP 剩余维度"]
```

### TP 与 PP group 怎么看

以 8 个 rank、`tp_size=2`、`pp_size=4` 为例，`initialize_model_parallel()` 注释中的 group 是：

| group 类型 | group 列表 | 含义 |
|---|---|---|
| TP group | `[g0,g1]`、`[g2,g3]`、`[g4,g5]`、`[g6,g7]` | 每个 PP stage 内，两个 TP rank 合作算该 stage 的层。 |
| PP group | `[g0,g2,g4,g6]`、`[g1,g3,g5,g7]` | 固定同一个 TP 分片编号，跨 PP stage 传 hidden states。 |

换句话说：

- TP group 横向切同一 stage 的层内矩阵。
- PP group 纵向串起不同 stage 的层。

### Attention rank 怎么从 tp_rank 拆出来

`_compute_parallelism_ranks()` 与 `compute_dp_attention_world_info()` 使用同一个层级思想：

```text
Attention hierarchy:
tp_rank = (attn_dp_rank * attn_cp_size + attn_cp_rank) * attn_tp_size + attn_tp_rank

attn_tp_size = tp_size / attn_dp_size / attn_cp_size
attn_dp_size = dp_size if enable_dp_attention else 1
```

如果 `tp_size=8`、`enable_dp_attention=True`、`dp_size=2`、`attn_cp_size=2`：

```text
attn_tp_size = 8 / 2 / 2 = 2

tp_rank 0 -> attn_dp_rank=0, attn_cp_rank=0, attn_tp_rank=0
tp_rank 1 -> attn_dp_rank=0, attn_cp_rank=0, attn_tp_rank=1
tp_rank 2 -> attn_dp_rank=0, attn_cp_rank=1, attn_tp_rank=0
tp_rank 3 -> attn_dp_rank=0, attn_cp_rank=1, attn_tp_rank=1
tp_rank 4 -> attn_dp_rank=1, attn_cp_rank=0, attn_tp_rank=0
tp_rank 5 -> attn_dp_rank=1, attn_cp_rank=0, attn_tp_rank=1
tp_rank 6 -> attn_dp_rank=1, attn_cp_rank=1, attn_tp_rank=0
tp_rank 7 -> attn_dp_rank=1, attn_cp_rank=1, attn_tp_rank=1
```

对应关系图：

```mermaid
flowchart LR
  TP0["tp_rank 0"] --> A00["attn_dp 0 / cp 0 / attn_tp 0"]
  TP1["tp_rank 1"] --> A01["attn_dp 0 / cp 0 / attn_tp 1"]
  TP2["tp_rank 2"] --> A10["attn_dp 0 / cp 1 / attn_tp 0"]
  TP3["tp_rank 3"] --> A11["attn_dp 0 / cp 1 / attn_tp 1"]
  TP4["tp_rank 4"] --> B00["attn_dp 1 / cp 0 / attn_tp 0"]
  TP5["tp_rank 5"] --> B01["attn_dp 1 / cp 0 / attn_tp 1"]
  TP6["tp_rank 6"] --> B10["attn_dp 1 / cp 1 / attn_tp 0"]
  TP7["tp_rank 7"] --> B11["attn_dp 1 / cp 1 / attn_tp 1"]
```

这也是为什么 DP attention 下不能把 `tp_rank` 直接当作“TP 内第几张卡”。它会再拆成 attention DP、attention CP、attention TP 三个维度。

### MoE rank 怎么从 tp_rank 拆出来

MoE 的层级是：

```text
MoE hierarchy:
Global(TP) -> MOE_DP -> EP -> MOE_TP

moe_dp_rank = tp_rank // (tp_size / moe_dp_size)
moe_ep_rank = tp_rank % (tp_size / moe_dp_size) // (tp_size / moe_dp_size / ep_size)
moe_tp_size = tp_size / moe_dp_size / ep_size
```

如果 `tp_size=8`、`moe_dp_size=2`、`ep_size=2`：

```text
moe_tp_size = 8 / 2 / 2 = 2

tp_rank 0 -> moe_dp_rank=0, moe_ep_rank=0, moe_tp_rank=0
tp_rank 1 -> moe_dp_rank=0, moe_ep_rank=0, moe_tp_rank=1
tp_rank 2 -> moe_dp_rank=0, moe_ep_rank=1, moe_tp_rank=0
tp_rank 3 -> moe_dp_rank=0, moe_ep_rank=1, moe_tp_rank=1
tp_rank 4 -> moe_dp_rank=1, moe_ep_rank=0, moe_tp_rank=0
tp_rank 5 -> moe_dp_rank=1, moe_ep_rank=0, moe_tp_rank=1
tp_rank 6 -> moe_dp_rank=1, moe_ep_rank=1, moe_tp_rank=0
tp_rank 7 -> moe_dp_rank=1, moe_ep_rank=1, moe_tp_rank=1
```

可以把 dense 层和 MoE 层理解成对同一个 TP 维度的两种解释：

```mermaid
flowchart TB
  TP["tp_rank 0..7"] --> Dense["Dense / Attention 解释<br/>attn_dp + attn_cp + attn_tp"]
  TP --> MoE["MoE 解释<br/>moe_dp + moe_ep + moe_tp"]
  Dense --> DenseComm["attention all-gather / all-reduce / CP 通信"]
  MoE --> MoEComm["expert dispatch / combine / EP 通信"]
```

---

## 11.4 进程、GPU 与 rank 的实际映射

标准 `dp_size == 1` 时，`Engine._launch_scheduler_processes()` 双层循环启动 Scheduler：

```text
for pp_rank in pp_rank_range:
  for tp_rank in tp_rank_range:
    gpu_id = base_gpu_id
           + ((pp_rank % pp_size_per_node) * tp_size_per_node)
           + (tp_rank % tp_size_per_node) * gpu_id_step
    attn_cp_rank, moe_dp_rank, moe_ep_rank = _compute_parallelism_ranks(server_args, tp_rank)
    mp.Process(target=run_scheduler_process, args=(..., gpu_id, tp_rank, ..., pp_rank, dp_rank=None))
```

因此一个模型计算子进程启动时已经知道：

| 参数 | 从哪里来 | 进入哪些类 | 用途 |
|---|---|---|---|
| `gpu_id` | `Engine._launch_scheduler_processes()` | `Scheduler`、`TpModelWorker`、`ModelRunner` | 当前进程绑定哪张 GPU。 |
| `tp_rank` | loop 变量 | `Scheduler`、`TpModelWorker`、`ModelRunner` | 当前 rank 在 TP 维度的位置。 |
| `pp_rank` | loop 变量 | `Scheduler`、`TpModelWorker`、`ModelRunner` | 当前 rank 属于哪个 pipeline stage。 |
| `attn_cp_rank` | `_compute_parallelism_ranks()` | `TpModelWorker`、`ModelRunner` | attention context parallel 的 rank。 |
| `moe_dp_rank` | `_compute_parallelism_ranks()` | `TpModelWorker`、`ModelRunner` | MoE data parallel 的 rank。 |
| `moe_ep_rank` | `_compute_parallelism_ranks()` | `TpModelWorker`、`ModelRunner` | expert parallel 的 rank。 |
| `dp_rank` | 普通模式为 `None`，DP controller 模式下指定 | `Scheduler`、`TpModelWorker`、`ModelRunner` | 当前 worker group 属于哪个 DP 副本。 |

这些参数的传递链路是：

```mermaid
flowchart LR
  Engine["_launch_scheduler_processes<br/>计算 gpu_id / tp_rank / pp_rank"] --> Run["run_scheduler_process"]
  Run --> Scheduler["Scheduler.__init__"]
  Scheduler --> Worker["TpModelWorker.__init__"]
  Worker --> Runner["ModelRunner.__init__"]
  Runner --> Dist["ModelRunner.init_torch_distributed"]
  Dist --> Groups["initialize_model_parallel<br/>创建 TP/PP/ATTN/MoE groups"]
```

### 单机 TP=4、PP=1 的直观映射

```text
进程数: 4 个 Scheduler 子进程
GPU 数: 4 张 GPU
world_size: 4

process 0 -> gpu 0 -> pp_rank 0, tp_rank 0 -> global_rank 0
process 1 -> gpu 1 -> pp_rank 0, tp_rank 1 -> global_rank 1
process 2 -> gpu 2 -> pp_rank 0, tp_rank 2 -> global_rank 2
process 3 -> gpu 3 -> pp_rank 0, tp_rank 3 -> global_rank 3
```

请求关系：

- `tp_rank=0` 的 Scheduler leader 从 TokenizerManager 收请求。
- 4 个 TP rank 通过 `tp_cpu_group` 广播 Python 请求对象。
- 4 个 `ModelRunner` 分别在自己的 GPU 上执行权重分片。
- 必要时通过 TP group 做 all-reduce / all-gather。

### 单机 TP=2、PP=2 的直观映射

```text
进程数: 4 个 Scheduler 子进程
GPU 数: 4 张 GPU
world_size: 4

process 0 -> gpu 0 -> pp_rank 0, tp_rank 0 -> global_rank 0
process 1 -> gpu 1 -> pp_rank 0, tp_rank 1 -> global_rank 1
process 2 -> gpu 2 -> pp_rank 1, tp_rank 0 -> global_rank 2
process 3 -> gpu 3 -> pp_rank 1, tp_rank 1 -> global_rank 3
```

group 关系：

```text
TP groups: [rank0, rank1], [rank2, rank3]
PP groups: [rank0, rank2], [rank1, rank3]
```

执行关系：

- PP0 的两个 TP rank 计算前半部分层。
- PP1 的两个 TP rank 计算后半部分层。
- PP0 输出 `PPProxyTensors`，通过 PP group 传给 PP1。
- 只有 PP 最后一级能得到最终 logits 并采样。

### DP=2、TP=2 的直观映射

普通 DP 会多一层 `DataParallelController`。每个 DP rank 是一套 TP group：

```text
DP rank 0:
  process 0 -> gpu 0 -> tp_rank 0
  process 1 -> gpu 1 -> tp_rank 1

DP rank 1:
  process 2 -> gpu 2 -> tp_rank 0
  process 3 -> gpu 3 -> tp_rank 1
```

```mermaid
flowchart TD
  T["TokenizerManager"] --> C["DataParallelController"]
  C -->|"request A"| DP0["DP rank 0<br/>TP group: gpu0 + gpu1"]
  C -->|"request B"| DP1["DP rank 1<br/>TP group: gpu2 + gpu3"]
  DP0 --> D["DetokenizerManager"]
  DP1 --> D
```

这里要特别注意：

- DP 不是让 request A 同时给 DP0 和 DP1 算。
- DP 是 controller 把不同请求路由到不同模型副本。
- 每个 DP group 内部仍然可以有 TP/PP/EP。

---

## 11.5 各并行策略和 SGLang 类的对应关系

| 并行策略 | 创建/计算 rank 的位置 | 保存 rank 的类 | 实际使用 rank 的类/模块 | 你读源码时应关注的对象 |
|---|---|---|---|---|
| TP | `Engine._launch_scheduler_processes()`、`_compute_parallelism_ranks()`、`ModelRunner.init_torch_distributed()` | `Scheduler`、`TpModelWorker`、`ModelRunner` | `distributed.communication_op`、TP linear layers、attention backend、sampler token sync | `tp_rank`、`get_tp_group()`、`tensor_model_parallel_all_reduce()` |
| PP | `_calculate_rank_ranges()`、`initialize_model_parallel()` | `Scheduler`、`TpModelWorker`、`ModelRunner` | `SchedulerPPMixin`、`PPProxyTensors`、`TpModelWorker.forward_batch_generation()` | `pp_rank`、`pp_group.is_last_rank`、`pp_proxy_tensors` |
| DP | `DataParallelController.launch_tensor_parallel_group()` | `DataParallelController`、`Scheduler`、`TpModelWorker`、`ModelRunner` | DP controller routing、worker status、load collector | `dp_rank`、每个 DP group 的 scheduler ports |
| DP attention | `compute_dp_attention_world_info()`、`initialize_dp_attention()` | `ModelRunner`、dp attention globals | `SchedulerDPAttnAdapter`、`ForwardBatch.global_num_tokens_*`、MLP sync | `attn_dp_rank`、`attn_tp_rank`、idle batch |
| CP / ATTN CP | `_compute_parallelism_ranks()`、`initialize_model_parallel()` | `TpModelWorker`、`ModelRunner` | attention backend、`ForwardBatch` metadata | `attn_cp_rank`、`get_attn_cp_group()` |
| EP / MoE EP | `_compute_parallelism_ranks()`、`initialize_model_parallel()` | `TpModelWorker`、`ModelRunner` | MoE layers、EPLB、expert location updater | `moe_ep_rank`、`ep_size`、expert location |
| MoE DP | `_compute_parallelism_ranks()`、`initialize_model_parallel()` | `TpModelWorker`、`ModelRunner` | MoE token dispatch/combine、expert routing | `moe_dp_rank`、`moe_dp_size` |

从类的角度看：

```mermaid
flowchart TB
  Engine["Engine<br/>负责启动进程和计算 gpu_id/tp/pp 范围"] --> Scheduler["Scheduler<br/>保存 rank 并执行调度循环"]
  Scheduler --> TpWorker["TpModelWorker<br/>把 rank 传入 ModelRunner"]
  TpWorker --> ModelRunner["ModelRunner<br/>初始化 torch distributed 和各类 group"]
  ModelRunner --> ParallelState["parallel_state.initialize_model_parallel<br/>创建 TP/PP/ATTN/MoE groups"]
  ModelRunner --> Layers["模型层 / attention backend / MoE layer<br/>实际使用 group 做 collective"]
  Scheduler --> Receiver["SchedulerRequestReceiver<br/>根据 rank leader 规则收请求和广播"]
  Scheduler --> DPAttn["SchedulerDPAttnAdapter<br/>DP attention 下同步 batch 形状"]
  Engine --> DPController["DataParallelController<br/>dp_size > 1 时负责路由和启动每个 DP group"]
```

---

## 11.6 不同并行下，请求在 rank 之间怎么流动

### TP：同一个请求复制到组内所有 rank

```mermaid
sequenceDiagram
  participant T as TokenizerManager
  participant S0 as Scheduler tp_rank=0
  participant S1 as Scheduler tp_rank=1
  participant W0 as ModelRunner rank0
  participant W1 as ModelRunner rank1

  T->>S0: TokenizedGenerateReqInput
  S0->>S1: broadcast_pyobj(recv_reqs)
  S0->>S0: 构造相同 ScheduleBatch
  S1->>S1: 构造相同 ScheduleBatch
  S0->>W0: ForwardBatch，本 rank 权重分片
  S1->>W1: ForwardBatch，本 rank 权重分片
  W0-->>W1: TP collective
  W1-->>W0: TP collective
  W0-->>S0: logits / token result leader path
```

### PP：请求和 hidden states 沿 stage 流动

```mermaid
sequenceDiagram
  participant T as TokenizerManager
  participant P0 as PP0 Scheduler / Worker
  participant P1 as PP1 Scheduler / Worker

  T->>P0: recv_reqs
  P0->>P1: point_to_point_pyobj(recv_reqs)
  P0->>P0: forward 前半层
  P0->>P1: PPProxyTensors(hidden states)
  P1->>P1: forward 后半层 + logits
  P1-->>P0: 必要的调度/结果同步
```

### DP：不同请求路由到不同副本

```mermaid
sequenceDiagram
  participant T as TokenizerManager
  participant C as DataParallelController
  participant D0 as DP0 Scheduler group
  participant D1 as DP1 Scheduler group

  T->>C: request A
  T->>C: request B
  C->>D0: route request A
  C->>D1: route request B
  D0-->>T: output A
  D1-->>T: output B
```

### DP attention：各 DP rank 可能有不同本地 batch，但需要同步形状

```mermaid
sequenceDiagram
  participant D0 as attn_dp_rank 0
  participant D1 as attn_dp_rank 1
  participant A as SchedulerDPAttnAdapter
  participant M as ModelRunner

  D0->>A: local batch num_tokens=128
  D1->>A: local batch num_tokens=0
  A->>A: all_gather global_num_tokens
  A->>D1: 构造 idle batch
  D0->>M: 正常 forward
  D1->>M: idle forward 参与 collective
```

### EP：token 根据 expert 路由到不同 rank

```mermaid
flowchart LR
  Tokens["MoE layer input tokens"] --> Router["router / top-k gate"]
  Router --> E0["expert group rank 0"]
  Router --> E1["expert group rank 1"]
  Router --> E2["expert group rank 2"]
  Router --> E3["expert group rank 3"]
  E0 --> Combine["combine outputs"]
  E1 --> Combine
  E2 --> Combine
  E3 --> Combine
```

EP 和 TP 的直觉差异：

- TP 是每个 token 的 dense 矩阵计算被拆到多个 rank。
- EP 是不同 token 被路由到不同 expert rank，expert 计算后再 combine。

---

## 11.7 多节点时 rank 如何落到节点和 GPU

`_calculate_rank_ranges()` 会根据 `nnodes`、`node_rank`、`tp_size`、`pp_size` 决定当前节点启动哪些 `pp_rank` 和 `tp_rank`。

核心思想：

```text
pp_size_per_node = max(pp_size // nnodes, 1)
nnodes_per_pp_rank = max(nnodes // pp_size, 1)
tp_size_per_node = tp_size // nnodes_per_pp_rank
```

这段逻辑服务两个场景：

1. **PP stage 跨节点放置。**
   如果 `pp_size >= nnodes`，每个节点可能负责一个或多个 PP stage。

2. **TP group 跨节点放置。**
   如果 `nnodes > pp_size`，一个 PP stage 可能跨多个节点组成 TP group。

简化例子：

| 配置 | 节点上的 rank 分布直觉 |
|---|---|
| `nnodes=1,tp_size=8,pp_size=1` | 单节点 8 个 TP rank。 |
| `nnodes=2,tp_size=8,pp_size=1` | 一个 TP group 跨 2 节点，每节点 4 个 TP rank。 |
| `nnodes=2,tp_size=4,pp_size=2` | 每个节点可负责一个 PP stage，每个 stage 内 4 个 TP rank 的一部分或全部，取决于 `tp_size_per_node`。 |
| `nnodes=4,tp_size=8,pp_size=2` | 每个 PP stage 跨 2 个节点组成 TP group。 |

多节点时还要记住第 17 节的结论：

- `node_rank == 0`：运行 HTTP、TokenizerManager、DetokenizerManager，以及本节点 Scheduler ranks。
- `node_rank > 0`：只运行本节点 Scheduler ranks 和 dummy health check。

所以“哪个节点接用户请求”和“哪个节点参与模型计算”不是同一件事。用户请求从 node 0 进入，但模型计算 rank 可以分布在所有节点。

---

## 11.8 一张总表：并行维度、进程、GPU、类、通信

| 维度 | 是否增加 Scheduler 进程数 | 是否增加 GPU 数 | 一个请求是否进入多个 rank | 主要通信对象 | 主要代码入口 |
|---|---:|---:|---|---|---|
| `tp_size` | 是 | 是 | 是，同一 TP group 内所有 rank | `tp_group`、`tp_cpu_group` | `_launch_scheduler_processes()`、`initialize_model_parallel()` |
| `pp_size` | 是 | 是 | 是，但按 stage 流动 | `pp_group`、`PPProxyTensors` | `_calculate_rank_ranges()`、`SchedulerPPMixin` |
| `dp_size` | 是，通常由 DP controller 启动多个 group | 是 | 否，请求被路由到一个 DP group | controller sockets、worker status | `DataParallelController` |
| `enable_dp_attention` | 不一定单独增加进程，重解释 TP 内 rank | 通常依赖已有 TP/DP 配置 | 部分同步，可能产生 idle batch | attention DP group、MLP sync info | `compute_dp_attention_world_info()`、`SchedulerDPAttnAdapter` |
| `attn_cp_size` | 不一定单独增加进程，要求 TP 维度可拆 | 使用 TP group 内 rank | 是，同一上下文被切分 | `attn_cp_group` | `_compute_parallelism_ranks()`、`initialize_model_parallel()` |
| `ep_size` | 不一定单独增加进程，使用 TP 内 rank | 使用 TP group 内 rank | token 被路由到 expert rank | `moe_ep_group` | `_compute_parallelism_ranks()`、MoE layers |
| `moe_dp_size` | 不一定单独增加进程，使用 TP 内 rank | 使用 TP group 内 rank | expert 维度数据并行 | `moe_dp_group` | `_compute_parallelism_ranks()`、MoE layers |

判断一个并行配置时，可以按这个顺序问：

1. 它是否复制一整套模型服务来分摊请求？如果是，优先看 DP。
2. 它是否把同一个模型层切到多 GPU？如果是，优先看 TP。
3. 它是否把不同 transformer layers 放到不同 GPU？如果是，优先看 PP。
4. 它是否只影响 attention 的 token/context 组织？如果是，看 DP attention / CP。
5. 它是否只影响 MoE expert 的放置和路由？如果是，看 EP / MoE DP / EPLB。

---

## 12. DataParallelController：DP 模式下的请求路由器

当 `server_args.dp_size > 1` 时，标准 Engine 不再直接启动所有 Scheduler，而是先启动：

`python/sglang/srt/managers/data_parallel_controller.py:run_data_parallel_controller_process()`

DP Controller 的职责：

1. 启动多个 DP worker group。
2. 为每个 DP rank 管理一个 ZMQ worker socket。
3. 从 TokenizerManager 收请求。
4. 根据策略把请求路由到某个 DP rank。
5. 收集/维护 worker status 和 load。

```mermaid
flowchart TD
  A["TokenizerManager"] --> B["DataParallelController"]
  B --> C{"routing policy"}
  C --> D["DP rank 0 Scheduler group"]
  C --> E["DP rank 1 Scheduler group"]
  C --> F["DP rank 2 Scheduler group"]
  D --> G["DetokenizerManager"]
  E --> G
  F --> G
```

你会在 DP Controller 里看到几类路由：

- `round_robin_scheduler()`
- `maybe_external_dp_rank_routing()`
- `follow_bootstrap_room_scheduler()`
- 结合 load snapshot 的调度

这和 TP 不同：TP 是同一个请求被广播给同组所有 rank 共同算；DP 是不同请求被分配给不同 DP group。

---

## 13. DP attention：不是普通 DP

DP attention 比普通请求级 DP 更复杂。它让 attention 层按 DP/TP/CP 组合切分，同时 MLP 侧可能仍需要同步 token 信息。

核心文件：

- `python/sglang/srt/layers/dp_attention.py`
- `python/sglang/srt/managers/scheduler_components/dp_attn.py`

DP attention 下，Scheduler 每轮 batch 决策后可能调用：

```text
dp_attn_adapter.maybe_prepare_mlp_sync_batch(...)
```

底层对应 `prepare_mlp_sync_batch_raw()`。

```mermaid
flowchart TD
  A["每个 DP rank 本地选 batch"] --> B["计算本地 num_tokens / forward_mode"]
  B --> C["MLPSyncBatchInfo.all_gather"]
  C --> D["得到 global_num_tokens / global_forward_mode"]
  D --> E{"其它 DP rank 有活吗?"}
  E -->|"Yes，本地无活"| F["构造 idle batch 参与同步"]
  E -->|"No"| G["正常本地 batch"]
  F --> H["MLP / attention 同步形状一致"]
  G --> H
```

为什么需要 idle batch？

因为某些分布式 collective 要求所有相关 rank 都进入同一类通信。如果某个 DP rank 没请求，但其它 DP rank 正在跑 forward，它也可能需要进入同步路径，否则会死锁或 shape 不一致。

---

## 14. 请求接收：rank leader 与广播

`SchedulerRequestReceiver.recv_requests()` 可以分四步看：

1. `_pull_raw_reqs()`
2. `input_blocker.handle()`
3. `_broadcast_reqs_across_ranks()`
4. `_apply_mm_receiver()` 与 `_finalize_shm_features()`

```mermaid
flowchart TD
  A["recv_requests"] --> B["_pull_raw_reqs"]
  B --> C["只有 leader 真正从 ZMQ / p2p 收对象"]
  C --> D["input_blocker 可过滤输入"]
  D --> E["_broadcast_reqs_across_ranks"]
  E --> F{"enable_dp_attention?"}
  F -->|"No"| G["broadcast to tp_group"]
  F -->|"Yes"| H["work req 广播到 attn_tp/attn_cp<br/>control req 可广播到 tp 或本地组"]
  G --> I["所有相关 rank 得到 recv_reqs"]
  H --> I
```

DP attention 下会把请求分为：

- work requests：真正生成/embedding 请求。
- control requests：flush、abort、profile、logging、RPC 等控制请求。

这样可以减少全 TP group 的昂贵同步。

---

## 15. DetokenizerManager：为什么单独一个进程

Scheduler 输出的是 token id，而 HTTP 层需要文本增量。DetokenizerManager 单独成进程有几个好处：

1. 把 tokenizer decode 的 CPU 工作从 GPU Scheduler 进程中挪走。
2. 支持增量 decode、stop string trim、工具调用 parser 等逻辑。
3. Scheduler 可以专注 GPU 调度和 forward。

核心函数：

| 文件 | 函数 | 作用 |
|---|---|---|
| `detokenizer_manager.py` | `DetokenizerManager.__init__()` | 初始化 IPC、tokenizer、状态、dispatcher |
| `detokenizer_manager.py` | `event_loop()` | 从 Scheduler 收 token id 输出，dispatch 后发回 TokenizerManager |
| `detokenizer_manager.py` | `handle_batch_token_id_out()` | 处理生成 token ids，增量 decode 成文本 |
| `detokenizer_manager.py` | `trim_matched_stop()` | 根据 stop string / stop token 裁剪输出 |

```mermaid
flowchart LR
  A["Scheduler<br/>BatchTokenIDOutput"] --> B["DetokenizerManager"]
  B --> C["decode token ids"]
  C --> D["trim stop"]
  D --> E["BatchStrOutput"]
  E --> F["TokenizerManager"]
```

---

## 16. Multi-tokenizer / Multi-detokenizer

SGLang 还支持多个 tokenizer worker 或 detokenizer worker。

相关入口：

- `Engine._launch_detokenizer_subprocesses()`
- `MultiTokenizerRouter`
- `run_multi_detokenizer_router_process()`

当 `detokenizer_worker_num > 1` 时：

```mermaid
flowchart TD
  A["Scheduler"] --> B["DetokenizerRouter"]
  B --> C["DetokenizerWorker 0"]
  B --> D["DetokenizerWorker 1"]
  B --> E["DetokenizerWorker N"]
  C --> F["TokenizerManager / TokenizerWorker"]
  D --> F
  E --> F
```

当 `tokenizer_worker_num > 1` 时，HTTP worker 会有自己的 TokenizerManager/TokenizerWorker，主进程通过 shared memory 写入 `PortArgs`、`ServerArgs`、`scheduler_info`。这部分是为了提升高并发场景下的 CPU tokenization/JSON 处理吞吐。

---

## 17. 多节点：node 0 负责入口，其它节点只跑计算 rank

在 `Engine._launch_subprocesses()` 中：

```python
if server_args.node_rank >= 1:
    scheduler_init_result.wait_for_ready()
    launch_dummy_health_check_server(...)
    scheduler_init_result.wait_for_completion()
```

含义是：

- node 0：运行 HTTP、TokenizerManager、DetokenizerManager、Scheduler ranks。
- node 1..N：只运行本节点 Scheduler ranks，并启动 dummy health check server。

```mermaid
flowchart TD
  subgraph N0["Node 0"]
    A["HTTP / Engine / Tokenizer"]
    B["Detokenizer"]
    C["Scheduler ranks"]
  end
  subgraph N1["Node 1"]
    D["Scheduler ranks only"]
    E["dummy health check"]
  end
  subgraph N2["Node 2"]
    F["Scheduler ranks only"]
    G["dummy health check"]
  end
  A --> C
  C <--> D
  C <--> F
```

这样做可以避免每个节点都暴露完整 API 服务，也避免请求入口分散导致状态管理复杂。

---

## 18. 模型 worker：rank 信息如何进入 ModelRunner

`TpModelWorker.__init__()` 接收：

- `tp_rank`
- `moe_ep_rank`
- `pp_rank`
- `attn_cp_rank`
- `moe_dp_rank`
- `dp_rank`
- `is_draft_worker`
- `req_to_token_pool`
- `token_to_kv_pool_allocator`

然后创建 `ModelRunner`：

```mermaid
flowchart TD
  A["Scheduler process args"] --> B["TpModelWorker"]
  B --> C["ModelConfig.from_server_args"]
  B --> D["ModelRunner(...)"]
  D --> E["加载本 rank 的模型权重分片"]
  D --> F["初始化 attention backend / CUDA graph / memory pool"]
```

在 TP 下，每个 rank 的 `ModelRunner` 只持有自己那部分权重或 heads；在 PP 下，每个 rank 只持有部分层；在 EP/MoE 下，每个 rank 只持有部分专家或参与专家通信。

---

## 19. 数据路径与控制路径要分开看

学习 SGLang 分布式时，最好把两类路径分开：

### 数据路径

```mermaid
flowchart LR
  A["GenerateReqInput"] --> B["TokenizedGenerateReqInput"]
  B --> C["ScheduleBatch"]
  C --> D["ForwardBatch"]
  D --> E["GPU forward"]
  E --> F["BatchTokenIDOutput"]
  F --> G["BatchStrOutput"]
```

### 控制路径

```mermaid
flowchart LR
  A["Flush / Abort / Profile / LoRA / WeightUpdate"] --> B["TokenizerManager / RPC"]
  B --> C["Scheduler dispatcher"]
  C --> D["control handler"]
  D --> E["broadcast / barrier / result"]
```

控制请求经常需要广播到多个 rank 或做 barrier；数据请求则更关注 batch、KV cache 和 forward。

---

## 20. 常见困惑

### 20.1 Scheduler 是一个进程还是多个进程？

一个 GPU rank 通常对应一个 Scheduler 子进程。TP/PP/DP 等并行度越高，Scheduler 进程越多。它们不是互相独立的 HTTP server，而是共同服务一个模型实例。

### 20.2 为什么只有 rank0 从 tokenizer 收请求？

因为 Python 请求对象只需要从 ZMQ 读一次。随后通过 `broadcast_pyobj` 或 p2p 传给其它 rank，保证所有相关 rank 看到一致的调度输入。

### 20.3 DP 和 TP 的区别是什么？

TP 是同一个请求由多个 rank 合作计算一个模型分片；DP 是不同请求可以被路由到不同 worker group。DP attention 又是更细的 attention/MLP 组合切分，不等同于简单复制模型。

### 20.4 为什么 Detokenizer 要单独进程？

decode 文本、stop 裁剪、工具调用解析都是 CPU 工作。单独进程可以避免阻塞 Scheduler 的 GPU 调度热路径。

### 20.5 为什么多节点非 0 节点不跑 tokenizer？

请求入口集中在 node 0，非 0 节点只贡献计算 rank。这样 API、状态、detokenization 和外部连接都更容易管理。

### 20.6 为什么 DP attention 需要 idle batch？

分布式 collective 要求相关 rank 同步进入。某个 DP rank 没有本地请求时，也可能需要用 idle batch 参与形状同步，避免其它 rank 在 collective 上等待它。

---

## 21. 本讲阅读任务

按下面顺序打开源码，跟读一遍：

| 顺序 | 文件 | 函数 / 代码段 | 阅读重点 |
|---:|---|---|---|
| 1 | `python/sglang/srt/entrypoints/engine.py` | `Engine` 类注释、`Engine.__init__()` | 看标准引擎由哪些组件组成。 |
| 2 | `python/sglang/srt/entrypoints/engine.py` | `_launch_subprocesses()` | 看启动顺序：PortArgs、Scheduler、Detokenizer、Tokenizer。 |
| 3 | `python/sglang/srt/server_args.py` | `PortArgs`、`PortArgs.init_new()` | 看普通 IPC 和 DP attention TCP 模式的地址差异。 |
| 4 | `python/sglang/srt/entrypoints/engine.py` | `_launch_scheduler_processes()` | 看 tp/pp rank 如何映射到 GPU 和 Scheduler 子进程。 |
| 5 | `python/sglang/srt/managers/scheduler.py` | `run_scheduler_process()`、`configure_scheduler_process()` | 看 Scheduler 子进程如何初始化并进入 event loop。 |
| 6 | `python/sglang/srt/managers/scheduler_components/ipc_channels.py` | `SchedulerIpcChannels.create()` | 看哪些 rank 绑定 ZMQ socket，哪些 rank 只是空 sender。 |
| 7 | `python/sglang/srt/managers/scheduler_components/request_receiver.py` | `SchedulerRequestReceiver.recv_requests()` | 看 rank leader 如何收请求并广播给 TP/CP/PP ranks。 |
| 8 | `python/sglang/srt/distributed/parallel_state.py` | `init_distributed_environment()`、`initialize_model_parallel()` | 看 world、TP、PP、attention、MoE groups 如何创建。 |
| 9 | `python/sglang/srt/managers/data_parallel_controller.py` | `launch_tensor_parallel_group()`、`round_robin_scheduler()` | 看 DP worker group 如何启动和路由。 |
| 10 | `python/sglang/srt/managers/scheduler_components/dp_attn.py` | `prepare_mlp_sync_batch_raw()` | 看 DP attention 为什么需要 all_gather 和 idle batch。 |
| 11 | `python/sglang/srt/managers/detokenizer_manager.py` | `DetokenizerManager.event_loop()`、`handle_batch_token_id_out()` | 看 token id 如何变回文本。 |

---

## 22. 你应该带走的心智模型

```mermaid
flowchart TD
  A["Engine 负责拉起进程"] --> B["PortArgs 定义通信地址"]
  B --> C["TokenizerManager 负责入口 tokenization"]
  C --> D["Scheduler rank leader 收请求"]
  D --> E["TP/CP/PP ranks 广播或 p2p 同步请求"]
  E --> F["ModelRunner 在每个 GPU rank 上执行本地分片"]
  F --> G["Scheduler rank leader 汇总/发送输出"]
  G --> H["DetokenizerManager 解码文本"]
  H --> C
```

如果你能用自己的话解释下面这句话，就说明这一讲过关了：

> SGLang 的分布式执行不是让每个 GPU 各自跑一个完整服务，而是由 Engine 拉起一组协作进程；TokenizerManager 统一入口，Scheduler rank leader 收请求并广播到并行组，多个 GPU rank 按 TP/PP/DP/EP 等 group 执行各自模型分片，最后由 DetokenizerManager 把 token 输出还原成文本。

---

## 23. 下一讲预告

下一讲建议进入 **Disaggregation / PD 分离**：

- prefill worker 和 decode worker 为什么要拆开？
- bootstrap、prealloc、transfer queue 分别做什么？
- KV cache 如何从 prefill worker 传到 decode worker？
- Mooncake / NIXL / ZMQ 等 transfer backend 在代码里如何接入？
- PD disaggregation 与 Scheduler 主循环、KV cache 生命周期有什么关系？
