# 第 7 讲：Disaggregation / PD 分离

这一讲接在第 6 讲之后。第 6 讲已经解释了一个统一模型实例内部的多进程、多卡、TP/PP/DP/EP rank 关系；这一讲开始看更高一层的部署拆分：

> Prefill 和 Decode 为什么可以拆成两类 server？请求在两个 server 之间如何交接？KV cache 又是怎么从 prefill 侧传到 decode 侧的？

本讲目标：

- 看懂 `disaggregation_mode = prefill / decode / null` 分别代表什么。
- 看懂 PD 分离中一个请求的生命周期。
- 看懂 prefill server 的 Bootstrap Queue、Waiting Queue、Inflight Queue。
- 看懂 decode server 的 PreallocQueue、TransferQueue、WaitingQueue、RunningBatch。
- 看懂 bootstrap、prealloc、metadata、KV sender/receiver、KV manager 的关系。
- 看懂 Mooncake / NIXL / Mori / Ascend / Fake 这些 transfer backend 如何接入统一抽象。
- 看懂 PD 分离和 Scheduler、Req、ScheduleBatch、KV cache pool、Radix cache 的关系。

---

## 0. 一张总图

```mermaid
flowchart TD
  U["用户请求"] --> DAPI["Decode server<br/>入口与 decode loop"]
  DAPI --> DReq["Decode Scheduler<br/>创建 Req"]
  DReq --> Prealloc["PreallocQueue<br/>预分配 decode 侧 KV slot"]
  Prealloc --> Receiver["KVReceiver<br/>把 decode 侧 KV 地址发给 prefill"]

  U -.bootstrap info.-> PAPI["Prefill server"]
  PAPI --> PReq["Prefill Scheduler<br/>创建 Req"]
  PReq --> Bootstrap["PrefillBootstrapQueue<br/>创建 KVSender"]
  Bootstrap --> PForward["Prefill forward<br/>写入 prefill 侧 KV cache"]
  PForward --> Sender["KVSender.send<br/>把 KV 传到 decode 侧"]
  Sender --> Receiver

  Receiver --> Transfer["TransferQueue<br/>轮询 KV transfer 状态"]
  Transfer --> DWait["Decode WaitingQueue<br/>构造 prebuilt extend batch"]
  DWait --> Running["RunningBatch<br/>进入 decode loop"]
  Running --> Out["输出 token"]
```

一句话版：

> PD 分离把 prompt prefill 和 token decode 拆到两个 server。decode server 负责请求入口、KV 预分配和后续 decode；prefill server 负责计算 prompt 的 KV cache，并通过 KV transfer backend 把 KV 写到 decode server 已经预留好的位置。

---

## 1. 关键文件跳转表

| 主题 | 文件 | 具体定位 |
|---|---|---|
| Scheduler 中的模式初始化 | `python/sglang/srt/managers/scheduler.py` | `Scheduler.__init__()` 中 `self.disaggregation_mode = DisaggregationMode(...)`；prefill/decode 初始化分支 |
| 请求入口如何写入 bootstrap 信息 | `python/sglang/srt/managers/scheduler.py` | `Scheduler.handle_generate_request()`、`handle_batch_generate_request()` |
| prefill 侧生命周期 | `python/sglang/srt/disaggregation/prefill.py` | 文件头生命周期注释、`PrefillBootstrapQueue`、`SchedulerDisaggregationPrefillMixin` |
| prefill 侧 bootstrap 队列 | `python/sglang/srt/disaggregation/prefill.py` | `PrefillBootstrapQueue.__init__()`、`_init_kv_manager()`、`create_sender()`、`add()` |
| prefill 侧 bootstrap 状态处理 | `python/sglang/srt/disaggregation/prefill.py` | `SchedulerDisaggregationPrefillMixin.handle_pending_bootstrap()`、`check_bootstrap()` |
| decode 侧生命周期 | `python/sglang/srt/disaggregation/decode.py` | 文件头生命周期注释、`DecodeRequest`、`SchedulerDisaggregationDecodeMixin` |
| decode 侧 req pool | `python/sglang/srt/disaggregation/decode.py` | `DecodeReqToTokenPool`、`HybridMambaDecodeReqToTokenPool` |
| decode 侧 prealloc/transfer 队列 | `python/sglang/srt/disaggregation/decode.py` | `PreallocQueue`、`TransferQueue` 相关 `add()` / poll 逻辑 |
| decode 侧 batch mixin | `python/sglang/srt/disaggregation/decode_schedule_batch_mixin.py` | `ScheduleBatchDisaggregationDecodeMixin` |
| KV transfer 抽象 | `python/sglang/srt/disaggregation/base/conn.py` | `KVArgs`、`KVPoll`、`BaseKVManager`、`BaseKVSender`、`BaseKVReceiver`、`BaseKVBootstrapServer` |
| 通用 KV manager | `python/sglang/srt/disaggregation/common/conn.py` | `CommonKVManager.__init__()`、`register_to_bootstrap()`、`try_ensure_parallel_info()` |
| transfer backend 注册 | `python/sglang/srt/disaggregation/utils.py` | `KVClassType`、`get_kv_class()` |
| Mooncake 后端 | `python/sglang/srt/disaggregation/mooncake/conn.py` | `MooncakeKVManager`、`MooncakeKVSender`、`MooncakeKVReceiver`、`MooncakeKVBootstrapServer` |
| NIXL / Mori / Ascend 后端 | `python/sglang/srt/disaggregation/nixl/conn.py`、`mori/conn.py`、`ascend/conn.py` | 各自的 `KVManager`、`KVSender`、`KVReceiver`、`KVBootstrapServer` |
| ServerArgs 配置 | `python/sglang/srt/server_args.py` | `disaggregation_mode`、`disaggregation_bootstrap_port`、transfer backend 相关字段 |

---

## 2. PD 分离解决什么问题

LLM serving 里 prefill 和 decode 的计算形态很不一样：

| 阶段 | 输入形态 | 计算特点 | 资源压力 |
|---|---|---|---|
| Prefill | prompt 的多个 token | 大 batch、大 token 数、attention 写入整段 KV | 算力和显存带宽压力大，单次延迟峰值高 |
| Decode | 每个请求每轮 1 个或少量 token | 高频小步循环，强依赖 KV cache | 低延迟、连续调度、KV cache 常驻 |

统一部署时，一个 Scheduler 同时处理 prefill 和 decode。PD 分离把两者拆开：

```mermaid
flowchart LR
  A["Prefill server<br/>适合吞吐型 prompt 计算"] -->|"KV cache transfer"| B["Decode server<br/>适合低延迟 token loop"]
```

这样做的收益：

- prefill server 可以专门处理长 prompt、chunked prefill、prefix 计算。
- decode server 可以专注持续 decode，减少大 prefill 对低延迟 token loop 的干扰。
- prefill 和 decode 可以独立扩缩容。
- 在大模型或长上下文场景中，可以把 KV cache transfer 作为部署层的显式数据流管理。

代价也很明确：

- 必须引入 bootstrap 协议，让两侧知道同一个请求对应哪个 transfer room。
- decode 侧必须先预留 KV cache 位置，否则 prefill 侧不知道要把 KV 写到哪里。
- 必须处理 transfer backend 的失败、超时、abort、重试。
- Scheduler 的 waiting/running 状态不再只有普通队列，还多了 bootstrap、prealloc、transfer 等中间队列。

---

## 3. 三种 `DisaggregationMode`

核心枚举在 `python/sglang/srt/disaggregation/utils.py` 的 `DisaggregationMode`。

| 模式 | 含义 | Scheduler 行为 |
|---|---|---|
| `NULL` | 不启用 PD 分离 | 普通 SGLang 主线：请求在同一个 Scheduler 中 prefill + decode。 |
| `PREFILL` | 当前实例是 prefill server | 负责 prompt prefill，计算 KV cache，并把 KV transfer 到 decode server。 |
| `DECODE` | 当前实例是 decode server | 负责请求入口、预分配 decode KV slot、接收 prefill KV，然后进入 decode loop。 |

Scheduler 初始化时会根据 `server_args.disaggregation_mode` 建立不同对象：

```mermaid
flowchart TD
  A["Scheduler.__init__"] --> B["self.disaggregation_mode"]
  B --> C{"mode"}
  C -->|"NULL"| N["普通 Scheduler<br/>无 PD 队列"]
  C -->|"PREFILL"| P["PrefillBootstrapQueue<br/>KVManager(PREFILL)<br/>bootstrap server / sender"]
  C -->|"DECODE"| D["Decode 侧队列<br/>KVManager(DECODE)<br/>receiver / prealloc / transfer"]
```

在第 6 讲的多进程模型里，一个 Scheduler 子进程绑定一个 GPU rank；在 PD 模式下，这个 Scheduler 还会额外带上 prefill 或 decode 的职责。

---

## 4. 请求里的 bootstrap 信息

PD 分离要让 prefill 和 decode 两个 server 对同一个请求达成共识，需要几个关键字段：

| 字段 | 所在对象 | 作用 |
|---|---|---|
| `bootstrap_host` | `Req` / request input | prefill 或 decode 对端的 host。 |
| `bootstrap_port` | `Req` / request input | bootstrap server 的端口，默认来自 `server_args.disaggregation_bootstrap_port`。 |
| `bootstrap_room` | `Req` / request input | 一次请求对应的 room id，用来匹配 sender 和 receiver。 |
| `pending_bootstrap` | `Req` | prefill 侧表示 sender 还没有完成握手/预分配。 |
| `disagg_kv_sender` | `Req` | prefill 侧持有的 KV sender。 |
| `metadata_buffer_index` | `Req` / `DecodeRequest` | 用于传输辅助 metadata 的 buffer slot。 |
| `kv_committed_len` | `Req` | decode 侧已经确认可用的 KV 长度。 |

`Scheduler.handle_generate_request()` 会补默认 `bootstrap_port`，并检查 PD 模式下请求是否携带足够 bootstrap 信息。`handle_batch_generate_request()` 再根据模式把请求放入不同队列。

```mermaid
flowchart TD
  A["TokenizedGenerateReqInput"] --> B["Scheduler.handle_generate_request"]
  B --> C["创建 Req"]
  C --> D{"disaggregation_mode"}
  D -->|"NULL"| W["普通 waiting_queue"]
  D -->|"PREFILL"| PB["disagg_prefill_bootstrap_queue.add(req)"]
  D -->|"DECODE"| DA["decode prealloc / transfer 相关队列"]
```

---

## 5. KV transfer 抽象层

PD 分离的关键不是 HTTP，而是 KV cache 如何从一个 GPU/rank 写到另一个 GPU/rank。SGLang 把这个能力抽象成几组类。

### 5.1 `KVArgs`

`KVArgs` 描述当前 rank 的 KV cache 内存布局：

| 字段 | 含义 |
|---|---|
| `engine_rank` | 当前 prefill/decode rank。 |
| `kv_data_ptrs` / `kv_data_lens` / `kv_item_lens` | KV cache buffer 的地址、长度、单 item 大小。 |
| `aux_data_ptrs` / `aux_data_lens` / `aux_item_lens` | 辅助 metadata buffer。 |
| `state_types` / `state_data_ptrs` | Mamba、SWA、DSA 等额外状态缓存。 |
| `kv_head_num` / `total_kv_head_num` | 当前 rank 与全局 KV head 数。 |
| `page_size` | paged KV cache 的 page 大小。 |
| `system_dp_rank` | 系统级 DP rank。 |
| `pp_rank` / `prefill_start_layer` / `prefill_end_layer` | PP 场景下当前 prefill stage 负责的层范围。 |

Prefill 侧的 `PrefillBootstrapQueue._init_kv_manager()` 会从 `token_to_kv_pool.get_contiguous_buf_infos()` 取出这些 buffer 信息，然后创建 `KVManager`。

Decode 侧也会创建自己的 `KVManager`，但它的角色是 receiver：告诉 prefill 侧“请把 KV 写到我这里的这些地址/indices”。

### 5.2 `KVPoll`

`KVPoll` 是 transfer 状态机：

| 状态 | 含义 |
|---|---|
| `Failed` | transfer 失败。 |
| `Bootstrapping` | sender/receiver 正在握手。 |
| `WaitingForInput` | receiver 已准备好，等待 prefill 侧真正产生 KV。 |
| `Transferring` | KV 正在传输。 |
| `Success` | KV 已传输完成，decode 可继续。 |

### 5.3 `BaseKVManager / BaseKVSender / BaseKVReceiver`

```mermaid
classDiagram
  class BaseKVManager {
    +register_to_bootstrap()
  }
  class BaseKVSender {
    +init(num_kv_indices, aux_index)
    +send(kv_indices, state_indices)
    +poll() KVPoll
    +failure_exception()
  }
  class BaseKVReceiver {
    +init(prefill_dp_rank)
    +send_metadata(kv_indices, aux_index, state_indices, decode_prefix_len)
    +poll() KVPoll
    +failure_exception()
  }
  class BaseKVBootstrapServer

  BaseKVManager <|-- CommonKVManager
  BaseKVSender <|-- CommonKVSender
  BaseKVReceiver <|-- CommonKVReceiver
  BaseKVBootstrapServer <|-- MooncakeKVBootstrapServer
```

角色分工：

| 对象 | 在 prefill 侧 | 在 decode 侧 |
|---|---|---|
| `KVManager` | 管理 prefill 侧 KV buffer，注册到 bootstrap server。 | 管理 decode 侧 KV buffer，查询 prefill parallel info。 |
| `KVSender` | 持有一个请求的 sender，负责把 prefill KV 发出去。 | 不使用。 |
| `KVReceiver` | 不使用。 | 持有一个请求的 receiver，负责把 decode 侧 KV indices/metadata 发给 prefill，并轮询 transfer。 |
| `KVBootstrapServer` | 暴露 prefill server 的 parallel / KV 信息。 | decode 侧通过 bootstrap 地址查询 prefill 信息。 |

### 5.4 backend 注册

`python/sglang/srt/disaggregation/utils.py:get_kv_class()` 根据 transfer backend 返回具体类：

| backend | Manager | Sender | Receiver | BootstrapServer |
|---|---|---|---|---|
| Mooncake | `MooncakeKVManager` | `MooncakeKVSender` | `MooncakeKVReceiver` | `MooncakeKVBootstrapServer` |
| NIXL | `NixlKVManager` | `NixlKVSender` | `NixlKVReceiver` | `NixlKVBootstrapServer` |
| Mori | `MoriKVManager` | `MoriKVSender` | `MoriKVReceiver` | `MoriKVBootstrapServer` |
| Ascend | `AscendKVManager` | `AscendKVSender` | `AscendKVReceiver` | `AscendKVBootstrapServer` |
| Fake | `FakeKVManager` | `FakeKVSender` | `FakeKVReceiver` | Fake/testing backend |

第一遍读源码时不要一上来读 Mooncake/NIXL 的底层传输细节。先理解 `BaseKVSender.init/send/poll` 与 `BaseKVReceiver.init/send_metadata/poll` 这两个抽象，后端只是把这几个动作落到不同通信库上。

---

## 6. Prefill server 生命周期

`python/sglang/srt/disaggregation/prefill.py` 文件头已经给了最好的主线：

```text
1. Bootstrap Queue
2. Waiting Queue
3. Inflight Queue
```

展开后是：

```mermaid
flowchart TD
  A["请求进入 prefill Scheduler"] --> B["PrefillBootstrapQueue.add(req)"]
  B --> C["create_sender(req)"]
  C --> D["KVSender.init<br/>握手 / 通知 KV 长度"]
  D --> E["queue 中等待 bootstrap 完成"]
  E --> F{"check_bootstrap(req)"}
  F -->|"未完成"| E
  F -->|"完成"| G["进入 waiting_queue"]
  G --> H["Scheduler 组 prefill batch"]
  H --> I["ModelRunner.forward_extend<br/>写入 prefill KV cache"]
  I --> J["KVSender.send(kv_indices, state_indices)"]
  J --> K["Inflight Queue 轮询 sender.poll"]
  K --> L{"transfer success?"}
  L -->|"否"| K
  L -->|"是"| M["请求在 prefill 侧完成"]
```

### 6.1 Bootstrap Queue

`PrefillBootstrapQueue` 的职责是“在真正 prefill forward 之前，把 transfer 的控制面准备好”。

关键函数：

| 函数 | 做什么 |
|---|---|
| `__init__()` | 保存 KV pool、metadata buffer、rank、bootstrap port、scheduler，并创建 `kv_manager`。 |
| `_init_kv_manager()` | 从 `token_to_kv_pool`、draft KV pool、metadata buffer 中收集指针和长度，构造 `KVArgs`，再通过 `get_kv_class()` 创建 backend manager。 |
| `create_sender(req, num_kv_heads)` | 为单个请求创建 `KVSender`，绑定 `bootstrap_addr`、`bootstrap_room`、目标 TP rank 等信息。 |
| `ensure_metadata_buffer(req)` | 为请求分配辅助 metadata buffer slot。 |
| `add(req, num_kv_heads)` | 把请求加入 bootstrap queue，等待 sender/receiver 握手和 decode 侧预分配完成。 |

`create_sender()` 里最关键的是这段关系：

```text
req.disagg_kv_sender = kv_sender_class(
    mgr=self.kv_manager,
    bootstrap_addr=f"{req.bootstrap_host}:{self.bootstrap_port}",
    bootstrap_room=req.bootstrap_room,
    dest_tp_ranks=[self.tp_rank],
    pp_rank=self.pp_rank,
)
```

它说明 sender 是“按请求”创建的，而 manager 是“按 rank / scheduler”创建的。

### 6.2 Waiting Queue

当 `check_bootstrap(req)` 返回完成后，请求进入普通 waiting queue。此时它和普通 prefill 请求很像：Scheduler 会把它放进 `ScheduleBatch`，执行 extend/prefill forward。

不同点在于：

- 这个请求已经有 `disagg_kv_sender`。
- 它可能有 `metadata_buffer_index`。
- prefill 完成后不能直接进入本地 decode，而是要发 KV 给 decode 侧。

### 6.3 Inflight Queue

prefill forward 写完 KV cache 后，prefill 侧会调用 sender 的 `send()`，把指定 `kv_indices` 对应的 KV 传输给 decode。

Inflight Queue 负责轮询 transfer：

```mermaid
flowchart LR
  A["prefill forward done"] --> B["KVSender.send"]
  B --> C["Inflight Queue"]
  C --> D["sender.poll"]
  D --> E{"KVPoll"}
  E -->|"Transferring"| C
  E -->|"Success"| F["释放/完成 prefill 请求"]
  E -->|"Failed"| G["失败处理 / abort"]
```

---

## 7. Decode server 生命周期

`python/sglang/srt/disaggregation/decode.py` 文件头把 decode 侧分成四段：

```text
1. PreallocQueue
2. TransferQueue
3. WaitingQueue
4. RunningBatch
```

展开后是：

```mermaid
flowchart TD
  A["请求进入 decode Scheduler"] --> B["创建 DecodeRequest"]
  B --> C["创建 KVReceiver"]
  C --> D["PreallocQueue"]
  D --> E{"decode 侧 KV slot 足够?"}
  E -->|"否"| D
  E -->|"是"| F["分配 req_pool_idx / KV indices / metadata buffer"]
  F --> G["KVReceiver.send_metadata<br/>把 decode KV 地址发给 prefill"]
  G --> H["TransferQueue"]
  H --> I["receiver.poll"]
  I --> J{"KVPoll"}
  J -->|"WaitingForInput / Transferring"| H
  J -->|"Success"| K["WaitingQueue"]
  K --> L["构造 PrebuiltExtendBatch<br/>跳过本地 prefill forward"]
  L --> M["合入 RunningBatch"]
  M --> N["decode loop"]
```

### 7.1 `DecodeReqToTokenPool`

普通 `ReqToTokenPool` 的容量约束是：

```text
#pre-allocated + #transfer + #running <= max_running_requests
```

decode 侧为了让 prefill 尽早开始，需要提前预分配一些还没进入 running batch 的请求。因此 `DecodeReqToTokenPool` 扩展了容量：

```text
#running <= max_running_requests
#pre-allocated + #transfer <= pre_alloc_size
```

这也是 decode 侧和普通 Scheduler 最大的内存池差异之一：decode server 需要容纳“正在等 KV transfer 的请求”。

### 7.2 PreallocQueue

PreallocQueue 的职责：

1. 创建或持有 `KVReceiver`。
2. 等待 decode 侧 KV cache 有足够空间。
3. 分配 `req_pool_idx` 与 KV indices。
4. 调用 `receiver.send_metadata(...)`，把 decode 侧地址告诉 prefill。
5. 把请求移动到 TransferQueue。

### 7.3 TransferQueue

TransferQueue 负责轮询 receiver：

```mermaid
flowchart LR
  A["KVReceiver.send_metadata"] --> B["TransferQueue"]
  B --> C["receiver.poll"]
  C --> D{"状态"}
  D -->|"Bootstrapping"| B
  D -->|"WaitingForInput"| B
  D -->|"Transferring"| B
  D -->|"Success"| E["进入 decode waiting queue"]
  D -->|"Failed"| F["失败 / abort / cleanup"]
```

这里最重要的概念是：decode 侧并不计算 prompt prefill，但它要先知道 prompt 的 KV cache 已经被写入自己的 KV pool。只有 `KVPoll.Success` 后，请求才能进入后续 decode。

### 7.4 WaitingQueue 到 RunningBatch

当 transfer 成功后，decode 侧会构造一个“prebuilt extend batch”。它不是为了重新跑 prefill，而是为了把请求的 metadata、seq len、KV indices、prefix 状态等放到 Scheduler 能理解的 batch 结构里。

然后请求合入 `running_batch`，之后就和普通 decode 请求一样，每轮生成新 token。

---

## 8. Prefill 与 Decode 的镜像关系

| 维度 | Prefill server | Decode server |
|---|---|---|
| 请求入口 | 接收带 bootstrap 信息的 prefill 请求 | 通常作为用户入口，创建 decode 请求 |
| 核心队列 | Bootstrap Queue、Waiting Queue、Inflight Queue | PreallocQueue、TransferQueue、WaitingQueue、RunningBatch |
| KV 对象 | `KVSender` | `KVReceiver` |
| KVManager 角色 | 暴露 prefill KV buffer，注册 bootstrap server | 查询 prefill parallel info，管理 decode KV buffer |
| 计算动作 | 真的执行 prompt prefill forward | 不重新算 prompt prefill，只接收 KV |
| 完成条件 | prefill forward + KV transfer success | KV transfer success 后进入 decode loop |
| 失败处理 | sender failure / bootstrap timeout / abort | receiver failure / prealloc 不足 / transfer timeout / abort |

可以把它想成一次“搬家”：

- Decode 侧先准备好房间和门牌号：KV slot、metadata buffer、bootstrap room。
- Prefill 侧负责生产家具：prompt KV cache。
- Transfer backend 负责把家具搬到 decode 侧指定房间。
- Decode 侧确认家具到位后，开始正常生活：进入 decode loop。

---

## 9. 和 Scheduler 主循环的关系

PD 分离并没有换掉 Scheduler，而是让 Scheduler 在不同模式下多维护几类队列。

```mermaid
flowchart TD
  S["Scheduler event loop"] --> R["recv_requests"]
  R --> H["handle_generate_request / handle_batch_generate_request"]
  H --> M{"disaggregation_mode"}
  M -->|"NULL"| W["waiting_queue"]
  M -->|"PREFILL"| PB["disagg_prefill_bootstrap_queue"]
  M -->|"DECODE"| DP["decode prealloc / transfer queues"]

  PB --> PW["bootstrap done -> waiting_queue"]
  DP --> DW["transfer done -> waiting_queue / prebuilt batch"]

  W --> B["get_next_batch_to_run"]
  PW --> B
  DW --> B
  B --> F["TpModelWorker / ModelRunner"]
```

### 9.1 `is_idle()` 为什么要看更多队列

普通模式下，Scheduler 是否 idle 主要看 waiting/running queue。PD 模式下还要看：

- prefill 的 bootstrap queue
- prefill 的 inflight transfer
- decode 的 prealloc queue
- decode 的 transfer queue

否则会出现“Scheduler 以为自己空闲，但其实还有请求在等待 KV transfer”的错误判断。

### 9.2 abort 为什么更复杂

普通请求 abort 只需要从 waiting/running 中移除并释放 KV。PD 请求可能处在：

- prefill bootstrap queue
- prefill waiting queue
- prefill inflight transfer
- decode prealloc queue
- decode transfer queue
- decode running batch

不同阶段需要清理的对象不同：metadata buffer、req pool slot、KV slot、sender/receiver 状态、bootstrap room 状态都可能要处理。

---

## 10. 和 KV Cache / Radix Cache 的关系

PD 分离不是替代 KV cache，而是改变 KV cache 的生产位置和消费位置。

```mermaid
flowchart LR
  P["Prefill KV pool"] -->|"transfer kv_indices 对应内容"| D["Decode KV pool"]
  D --> R["Decode running batch"]
  R --> A["Attention backend 读取 decode KV pool"]
```

### 10.1 prefill 侧

prefill 侧会正常执行 prompt forward，因此它会：

- 分配 prefill 侧 KV cache slot。
- 写入 prompt 的 KV。
- 根据 `kv_indices` 把这些 KV 发送出去。
- transfer 完成后释放或回收 prefill 侧请求资源。

### 10.2 decode 侧

decode 侧需要提前分配目标 KV slot，因为 transfer 要写入这些 slot：

- `DecodeReqToTokenPool` 记录请求到 token slot 的映射。
- KV allocator 分配 decode 侧目标 KV indices。
- `KVReceiver.send_metadata()` 把这些 indices 通知 prefill。
- transfer 完成后，decode attention backend 就能像普通请求一样读取这些 KV。

### 10.3 Radix cache / HiCache

PD 模式也会遇到 prefix cache：

- decode 侧可能先做 prefix match，判断哪些 KV 已经可以复用。
- prefill 侧可能只需要计算未命中的部分。
- HiCache 模式下，decode 侧还有 restore/load-back 相关状态，`decode_hicache_mixin.py` 会参与 prealloc/transfer 流程。

第一遍读 PD 时，可以先按“没有 prefix cache 命中”的路径理解。第二遍再叠加 Radix/HiCache。

---

## 11. 和 TP / PP / DP 的关系

PD 分离本身不是 TP/PP/DP 的替代品。prefill server 和 decode server 内部仍然可以各自使用 TP、PP、DP、EP。

```mermaid
flowchart TB
  subgraph Prefill["Prefill server"]
    P0["TP/PP/EP ranks"]
    PS["Prefill Scheduler"]
    PK["Prefill KV pool"]
  end

  subgraph Decode["Decode server"]
    D0["TP/PP/EP ranks"]
    DS["Decode Scheduler"]
    DK["Decode KV pool"]
  end

  PK -->|"KV transfer backend"| DK
```

几个关键点：

| 并行 | 在 PD 中的影响 |
|---|---|
| TP | prefill 和 decode 侧可能都有 TP rank。transfer backend 需要知道目标 TP rank 和 KV head 分片。 |
| PP | prefill 侧每个 PP stage 可能只负责部分 layer，因此 `KVArgs` 里有 `pp_rank`、`prefill_start_layer`、`prefill_end_layer`。 |
| DP | 多个 prefill/decode 副本时，bootstrap room 和 routing 必须确保同一请求的两端匹配。 |
| DP attention / CP | transfer 时要考虑 attention TP/CP 的 KV 切分方式，metadata 里要保留足够信息。 |
| EP / MoE | MoE 不直接改变 KV cache 的语义，但会影响模型 rank 和 forward 执行过程。 |

`KVArgs` 中的 `kv_head_num`、`total_kv_head_num`、`pp_rank`、`prefill_start_layer`、`prefill_end_layer` 就是在为这些并行组合提供信息。

---

## 12. 一次 PD 请求的完整时序

```mermaid
sequenceDiagram
  participant User as Client
  participant Decode as Decode Scheduler
  participant Receiver as KVReceiver
  participant Prefill as Prefill Scheduler
  participant Sender as KVSender
  participant PModel as Prefill ModelRunner
  participant DModel as Decode ModelRunner

  User->>Decode: GenerateReqInput
  Decode->>Decode: 创建 Req / bootstrap_room
  Decode->>Receiver: init(prefill_dp_rank)
  Decode->>Decode: prealloc req_pool_idx / kv_indices / metadata buffer
  Decode->>Receiver: send_metadata(kv_indices, aux_index, state_indices)

  User->>Prefill: Prefill 请求或路由后的请求
  Prefill->>Sender: create_sender(req)
  Sender->>Receiver: bootstrap handshake
  Prefill->>Prefill: bootstrap done -> waiting_queue
  Prefill->>PModel: prefill forward
  PModel-->>Prefill: KV cache written
  Prefill->>Sender: send(kv_indices, state_indices)
  Sender-->>Receiver: KV transfer
  Receiver-->>Decode: poll() = Success
  Decode->>Decode: transfer queue -> waiting queue
  Decode->>DModel: decode forward
  DModel-->>Decode: next token
  Decode-->>User: stream output
```

---

## 13. 失败、超时和重试

PD 分离把一次请求拆成两个 server 的协作，因此失败场景比普通模式多。

| 场景 | 可能原因 | 相关代码 |
|---|---|---|
| bootstrap 超时 | prefill 或 decode 侧没有及时完成握手 | `KVPoll.Bootstrapping`、sender/receiver `failure_exception()` |
| decode 侧 prealloc 不足 | KV cache 不够，无法给请求预留位置 | decode `PreallocQueue`、`DecodeReqToTokenPool.available_size()` |
| transfer 失败 | 后端连接失败、对端退出、IB/NIXL/Mooncake 错误 | backend `KVSender.poll()`、`KVReceiver.poll()` |
| abort | 用户取消请求或 Scheduler 控制请求 | `prepare_abort()`、prefill/decode 各自队列清理 |
| optimistic prefill retry | prefill 先做乐观计算，但 bootstrap 没及时完成，需要回退 | `should_force_retry()`、`handle_pending_bootstrap()` |

读源码时要注意：很多失败处理不会直接抛到最外层，而是先把状态写成 `KVPoll.Failed`，然后由队列轮询阶段统一清理。

---

## 14. 第一遍阅读建议：先走最简单路径

建议第一遍假设：

- 单节点。
- 无 PP。
- 无 DP attention。
- 无 HiCache restore。
- 无 prefix cache 命中。
- transfer backend 先当作抽象，不深入 Mooncake/NIXL 细节。

最简路径：

```mermaid
flowchart TD
  A["Decode request"] --> B["Decode prealloc KV slot"]
  B --> C["Receiver.send_metadata"]
  C --> D["Prefill create_sender"]
  D --> E["Prefill forward"]
  E --> F["Sender.send KV"]
  F --> G["Receiver.poll success"]
  G --> H["Decode running batch"]
```

掌握这条路径后，再打开复杂分支：

1. chunked prefill 与 optimistic retry。
2. PP 下 layer 范围和 `prefill_start_layer/end_layer`。
3. HiCache restore。
4. Mooncake/NIXL 的真实 transfer 线程。
5. 多 DP prefill/decode routing。

---

## 15. 常见困惑

### 15.1 PD 分离是不是把模型切成两半？

不是。PD 分离不是 layer 级切分。prefill server 和 decode server 通常都可以加载模型，只是它们服务的阶段不同：prefill server 负责 prompt prefill，decode server 负责后续 autoregressive decode。

### 15.2 decode server 为什么要先分配 KV？

因为 prefill 侧要把 KV 写入 decode 侧指定位置。没有目标 KV indices，transfer backend 不知道该写到 decode 侧哪个 slot。

### 15.3 prefill server 算完 prompt 后还会 decode 吗？

PD 模式下通常不会。prefill 侧计算 prompt KV，并把 KV transfer 出去；decode 侧接收 KV 后负责后续 token loop。

### 15.4 bootstrap room 是什么？

可以理解为一次请求的 transfer 房间号。sender 和 receiver 通过同一个 `bootstrap_room` 找到彼此，并区分不同请求的 transfer 状态。

### 15.5 为什么需要 metadata buffer？

KV cache 之外还有辅助状态需要同步，例如 request 校验信息、Mamba/SWA/DSA 等状态，或者 transfer 所需的额外 metadata。`MetadataBuffers` 和 `metadata_buffer_index` 就是为这些信息准备的。

### 15.6 PD 和 Radix cache 会不会冲突？

不会，但状态更复杂。Radix cache 关注 prefix KV 是否可复用；PD transfer 关注 KV 在 prefill 和 decode 两侧如何交接。decode 侧如果已有 prefix KV，可以减少 prefill 侧需要计算和传输的部分。

---

## 16. 本讲阅读任务

按下面顺序打开源码：

| 顺序 | 文件 | 函数 / 代码段 | 阅读重点 |
|---:|---|---|---|
| 1 | `python/sglang/srt/server_args.py` | `disaggregation_mode`、`disaggregation_bootstrap_port`、transfer backend 参数 | 先看有哪些启动开关。 |
| 2 | `python/sglang/srt/managers/scheduler.py` | `Scheduler.__init__()` 中 disaggregation 初始化分支 | 看 Scheduler 如何按 prefill/decode 模式创建不同队列。 |
| 3 | `python/sglang/srt/managers/scheduler.py` | `handle_generate_request()`、`handle_batch_generate_request()` | 看请求如何携带 bootstrap 信息并进入不同队列。 |
| 4 | `python/sglang/srt/disaggregation/base/conn.py` | `KVArgs`、`KVPoll`、`BaseKVSender`、`BaseKVReceiver` | 先理解统一抽象，不急着读后端实现。 |
| 5 | `python/sglang/srt/disaggregation/utils.py` | `KVClassType`、`get_kv_class()` | 看 backend 如何映射到 manager/sender/receiver。 |
| 6 | `python/sglang/srt/disaggregation/prefill.py` | `PrefillBootstrapQueue` | 看 prefill 侧如何创建 sender、准备 KVArgs、等待 bootstrap。 |
| 7 | `python/sglang/srt/disaggregation/prefill.py` | `SchedulerDisaggregationPrefillMixin.handle_pending_bootstrap()`、`check_bootstrap()` | 看 prefill 请求如何从 bootstrap queue 进入 waiting queue。 |
| 8 | `python/sglang/srt/disaggregation/decode.py` | `DecodeReqToTokenPool`、`DecodeRequest` | 看 decode 侧为什么需要预分配 request/token pool。 |
| 9 | `python/sglang/srt/disaggregation/decode.py` | PreallocQueue / TransferQueue 相关 `add()` 和 poll 逻辑 | 看 decode 侧如何等待 KV transfer 完成。 |
| 10 | `python/sglang/srt/disaggregation/mooncake/conn.py` | `MooncakeKVSender`、`MooncakeKVReceiver`、`MooncakeKVManager` | 第二遍再读真实 backend 的线程和网络细节。 |

---

## 17. 你应该带走的心智模型

```mermaid
flowchart TD
  A["Decode 侧先接请求"] --> B["预分配 decode KV slot"]
  B --> C["Receiver 把目标 KV indices 发给 Prefill"]
  C --> D["Prefill 侧计算 prompt KV"]
  D --> E["Sender 把 KV 写入 Decode 侧目标位置"]
  E --> F["Decode 确认 transfer success"]
  F --> G["请求进入 decode running batch"]
```

如果你能用自己的话解释下面这句话，就说明这一讲过关了：

> PD 分离不是把模型层切开，而是把请求生命周期切成 prefill server 和 decode server 两段；decode 侧先预留 KV cache 位置并通过 receiver 发出 metadata，prefill 侧计算 prompt KV 后通过 sender 传输到这些位置，transfer 成功后 decode 侧才开始正常的 token-by-token decode。

---

## 18. 下一讲预告

下一讲建议进入 **LoRA Serving / Adapter 热加载**：

- LoRA adapter 在请求、Scheduler、ModelRunner 中如何传递？
- 为什么 LoRA 会影响 batch 混排？
- `LoRAManager` 如何加载、缓存、卸载 adapter？
- LoRA 与 TP、CUDA graph、MoE buffer 有什么关系？
- 在线加载 LoRA 和权重热更新有什么区别？
