# 第 9 讲：SGLang Router 架构与源码解析

前面几讲我们一直在讲 TokenizerManager、Scheduler、ModelRunner、KV cache、PD 分离和 LoRA，但有一个词很容易被遗漏：**router**。

在 SGLang 代码里，`router` 不是单一模块，而是出现在多个层次：

- **服务入口路由**：例如 Ollama SmartRouter，把请求转到本地 Ollama 或远端 SGLang。
- **调度模拟路由**：在 schedule simulator 中，把模拟请求分配到某张 GPU。
- **PD 分离 bootstrap route**：Prefill/Decode 分离部署中，用 `/route` 注册和查询 rank 连接信息。
- **MoE expert router**：模型内部把 token 路由到 top-k experts。
- **routed experts 捕获链路**：把 MoE router 选中的专家 ID 返回给用户做观测和调试。

本仓库此前为了教学目的裁剪掉了大量非 Python 代码，因此如果你说的是上游项目里独立的 `sglang-router` 服务，本仓库当前没有保留那部分完整源码。本讲会基于当前仓库中可见的 Python router 相关实现，建立一张完整的“router 分层地图”，并深入讲清楚其中最关键的 MoE expert router 和服务入口 SmartRouter。

---

## 0. 一张图区分四种 Router

```mermaid
flowchart TD
  Client["Client Request"] --> Entry["EntryPoint / API Layer"]

  Entry --> Smart["Ollama SmartRouter<br/>请求级路由"]
  Smart --> Local["Local Ollama"]
  Smart --> Remote["Remote SGLang"]

  Entry --> Engine["SGLang Engine"]
  Engine --> Tokenizer["TokenizerManager"]
  Tokenizer --> Scheduler["Scheduler"]
  Scheduler --> ModelRunner["ModelRunner"]

  ModelRunner --> Model["Model Forward"]
  Model --> MoeRouter["MoE Expert Router<br/>token -> top-k experts"]
  MoeRouter --> Experts["FusedMoE / Expert Parallel"]
  MoeRouter --> Capturer["RoutedExpertsCapturer"]
  Capturer --> Detok["DetokenizerManager"]
  Detok --> Response["Response meta_info.routed_experts"]

  Sim["Schedule Simulator"] --> SimRouter["RouterPolicy<br/>Random / RoundRobin / Sticky"]
  SimRouter --> SimGPU["Simulated GPUState"]

  PD["PD Bootstrap Server"] --> RouteAPI["/route"]
  RouteAPI --> RankTable["Prefill rank table<br/>room -> dp rank"]
```

这张图先帮你把概念拆开：

- SmartRouter 是**服务选择器**，决定请求去本地还是远端。
- Schedule simulator router 是**实验工具**，用于研究请求分配策略。
- PD `/route` 是**连接信息注册表**，服务 Prefill/Decode 分离。
- MoE router 是**模型内部计算路径**，决定每个 token 用哪些 experts。

其中最核心、最值得读源码的是 MoE expert router，因为它直接参与 forward，影响性能、专家负载和结果。

---

## 1. 关键文件跳转表

| 主题 | 文件 / 函数或类 | 作用 |
| --- | --- | --- |
| Ollama SmartRouter | `python/sglang/srt/entrypoints/ollama/smart_router.py` / `SmartRouter` | 用 LLM judge 判断请求复杂度，在本地 Ollama 和远端 SGLang 之间选择。 |
| SmartRouter 分类 | `smart_router.py` / `_classify_with_llm()` | 构造分类 prompt，调用 judge model，返回 SIMPLE / COMPLEX。 |
| SmartRouter 非流式请求 | `smart_router.py` / `chat()` | 决策目标 backend，执行请求，失败时 fallback。 |
| SmartRouter 流式请求 | `smart_router.py` / `chat_stream()` | 与 `chat()` 类似，但直接 yield streaming chunks。 |
| 调度模拟 router 抽象 | `python/sglang/srt/debug_utils/schedule_simulator/routers/base.py` / `RouterPolicy.route()` | 定义模拟请求到 GPU ID 的路由接口。 |
| 随机路由 | `routers/random_router.py` / `RandomRouter.route()` | 随机选择 GPU。 |
| 轮询路由 | `routers/round_robin_router.py` / `RoundRobinRouter.route()` | 按 counter 轮询选择 GPU。 |
| 粘性路由 | `routers/sticky_router.py` / `StickyRouter.route()` | 同一 `group_id` 固定映射到同一 GPU。 |
| 模拟器使用 router | `python/sglang/srt/debug_utils/schedule_simulator/simulator.py` / `_route_requests()` | 调用 `router.route(req)`，把请求放入对应 GPU 的 pending 队列。 |
| PD route 服务 | `python/sglang/srt/disaggregation/common/conn.py` / `_setup_routes()`、`_handle_route()` | 给 bootstrap server 注册 `/route`、`/register_dp_rank`、`/query_dp_ranks`。 |
| MoE router kernel | `python/sglang/srt/layers/moe/router.py` / `fused_moe_router_cudacore_kernel()`、`fused_moe_router_tensorcore_kernel()` | Triton kernel，计算 router logits 并选 top-k experts。 |
| MoE router Python 入口 | `python/sglang/srt/layers/moe/router.py` / `fused_moe_router_shim()`、`FusedMoeRouter.forward_cuda()` | 根据 batch / expert / hidden 形状选择 cudacore 或 tensorcore kernel。 |
| MoE top-k 后处理 | `python/sglang/srt/layers/moe/topk.py` / `_post_process_topk_ids()` | 捕获 routed experts，并处理 DeepEP / shared experts / EPLB 映射。 |
| routed experts 捕获 | `python/sglang/srt/state_capturer/routed_experts.py` / `RoutedExpertsCapturer` | 保存每层每个 token 的 top-k expert IDs，支持 DP / DeepEP 切片。 |
| 输出收集 | `python/sglang/srt/managers/scheduler_components/batch_result_processor.py` / `_maybe_collect_routed_experts()` | 请求结束时从 capturer 中取出 routed experts。 |
| Detokenizer 编码 | `python/sglang/srt/managers/detokenizer_manager.py` / `_b64_encode_per_request()`、`handle_batch_token_id_out()` | 把 routed experts tensor 编成 base64 放入响应。 |
| OpenAI 请求字段 | `python/sglang/srt/entrypoints/openai/protocol.py` / `return_routed_experts`、`routed_experts_start_len` | 用户请求是否返回 routed experts，以及从哪个 token 起返回。 |

---

## 2. Router 的分层心智模型

先给一个非常实用的分类：

```text
请求级 router:
  输入：完整用户请求
  输出：选择哪个服务实例 / backend
  例子：SmartRouter

实验级 router:
  输入：模拟请求 SimRequest
  输出：选择哪个模拟 GPU
  例子：RandomRouter / RoundRobinRouter / StickyRouter

连接级 route:
  输入：rank / room / dp 信息
  输出：Prefill 与 Decode 之间如何建立连接
  例子：PD bootstrap /route

模型级 router:
  输入：hidden states
  输出：top-k expert ids 和 top-k weights
  例子：MoE router

观测级 routed experts:
  输入：MoE router 的 topk_ids
  输出：响应里的 routed_experts metadata
  例子：RoutedExpertsCapturer
```

读源码时要先问：**这个 router 的输入输出是什么？它路由的是请求、GPU、rank，还是 token？**

---

## 3. SmartRouter：请求级路由

### 3.1 它解决什么问题

`SmartRouter` 位于：

```text
python/sglang/srt/entrypoints/ollama/smart_router.py
类：SmartRouter
```

它的目标很直接：

- 简单请求走本地 Ollama，低延迟、低成本；
- 复杂请求走远端 SGLang，使用更强模型；
- 如果目标 backend 失败，fallback 到另一个 backend。

它不是 SGLang 核心 serving pipeline 的必要组件，更像一个轻量级入口示例。

### 3.2 SmartRouter 架构图

```mermaid
flowchart TD
  A["用户请求 prompt / messages"] --> B{"force_remote / force_local?"}
  B -- "force_remote" --> Remote["Remote SGLang"]
  B -- "force_local" --> Local["Local Ollama"]
  B -- "未强制" --> Judge["judge_client.chat()<br/>分类 SIMPLE / COMPLEX"]

  Judge --> C{"分类结果"}
  C -- "SIMPLE" --> Local
  C -- "COMPLEX" --> Remote

  Local --> D{"调用成功?"}
  Remote --> D
  D -- "成功" --> Resp["返回 content/model/location/reason"]
  D -- "失败" --> Fallback["切换到另一个 backend"]
  Fallback --> Resp
```

### 3.3 初始化：创建三个 client

```text
python/sglang/srt/entrypoints/ollama/smart_router.py
函数：SmartRouter.__init__()
```

初始化时会创建：

- `local_client`：连接本地 Ollama。
- `remote_client`：连接远端 SGLang，使用 Ollama-compatible API。
- `judge_client`：负责分类请求复杂度，默认复用本地模型。

关键字段：

```text
local_host
remote_host
local_model
remote_model
judge_model
judge_host
```

### 3.4 分类：_classify_with_llm()

```text
python/sglang/srt/entrypoints/ollama/smart_router.py
函数：SmartRouter._classify_with_llm()
```

它做了四步：

1. 把用户 prompt 填入 `CLASSIFICATION_PROMPT`。
2. 调用 `judge_client.chat()`。
3. 要求 judge 只输出 `SIMPLE` 或 `COMPLEX`。
4. 如果 judge 失败，默认走本地。

简化伪代码：

```text
classification_prompt = CLASSIFICATION_PROMPT.format(prompt=prompt[:500])
response = judge_client.chat(model=judge_model, messages=[...], temperature=0)
result = response["message"]["content"].strip().upper()

if "COMPLEX" in result:
    return True, "Complex task"
else:
    return False, "Simple task"
```

这里的返回值是 `(use_remote, reason)`：

- `use_remote=True`：走远端 SGLang；
- `use_remote=False`：走本地 Ollama。

### 3.5 非流式请求：chat()

```text
python/sglang/srt/entrypoints/ollama/smart_router.py
函数：SmartRouter.chat()
```

流程：

```mermaid
sequenceDiagram
  participant User
  participant Router as SmartRouter
  participant Judge as Judge Model
  participant Local as Local Ollama
  participant Remote as Remote SGLang

  User->>Router: chat(prompt/messages)
  Router->>Router: 提取最后一条 user message
  alt force_remote
    Router->>Remote: client.chat()
  else force_local
    Router->>Local: client.chat()
  else normal
    Router->>Judge: classify prompt
    Judge-->>Router: SIMPLE / COMPLEX
    alt SIMPLE
      Router->>Local: client.chat()
    else COMPLEX
      Router->>Remote: client.chat()
    end
  end
  alt backend failed
    Router->>Router: fallback to the other backend
  end
  Router-->>User: content / model / location / reason
```

返回结构：

```text
{
  "content": response["message"]["content"],
  "model": model,
  "location": "Local Ollama" or "Remote SGLang",
  "reason": reason,
}
```

### 3.6 流式请求：chat_stream()

```text
python/sglang/srt/entrypoints/ollama/smart_router.py
函数：SmartRouter.chat_stream()
```

它和 `chat()` 的决策逻辑几乎一样，区别在最后一步：

```text
for chunk in client.chat(model=model, messages=messages, stream=True):
    yield chunk
```

所以 SmartRouter 的 streaming 并没有自己做 token 级调度，它只是把选中的 backend 的 streaming chunk 原样转发出去。

### 3.7 SmartRouter 的边界

SmartRouter 很适合教学，但它不是生产级全局负载均衡器。它缺少：

- backend 健康检查和权重；
- 队列长度感知；
- KV cache 命中率感知；
- model / adapter / tenant 维度路由；
- 并发限流；
- retry budget；
- Prometheus metrics。

所以它更像一个“请求分类器 + backend 选择器”的最小实现。

---

## 4. Schedule Simulator Router：调度实验中的请求分配

调度模拟器的 router 位于：

```text
python/sglang/srt/debug_utils/schedule_simulator/routers/
```

这一组实现不参与真实 serving，而是用来研究不同请求分配策略对 GPU 队列和吞吐的影响。

### 4.1 RouterPolicy 抽象

```text
python/sglang/srt/debug_utils/schedule_simulator/routers/base.py
类：RouterPolicy
函数：route(incoming_request: SimRequest) -> int
```

它只定义一个接口：

```text
输入：SimRequest
输出：gpu_id
```

### 4.2 SimRequest

```text
python/sglang/srt/debug_utils/schedule_simulator/request.py
类：SimRequest
```

核心字段：

```text
request_id
input_len
output_len
decoded_tokens
group_id
prefix_len
```

`group_id` 对 sticky router 很关键，它表示一组相关请求。比如多轮会话、共享 prefix 的请求、同一用户请求，都可以用同一个 group。

### 4.3 三种内置策略

```text
python/sglang/srt/debug_utils/schedule_simulator/routers/random_router.py
类：RandomRouter
策略：随机选择 GPU。

python/sglang/srt/debug_utils/schedule_simulator/routers/round_robin_router.py
类：RoundRobinRouter
策略：counter % num_gpus，轮询选择 GPU。

python/sglang/srt/debug_utils/schedule_simulator/routers/sticky_router.py
类：StickyRouter
策略：同一 group_id 固定到同一 GPU；没有 group_id 时随机。
```

对比：

| 策略 | 优点 | 缺点 |
| --- | --- | --- |
| RandomRouter | 简单，天然打散 | 不关心 prefix locality 和负载 |
| RoundRobinRouter | 分布均匀、可预测 | 不关心请求长度和历史状态 |
| StickyRouter | 适合保持会话或 prefix locality | 可能造成热点 GPU |

### 4.4 Simulator 如何使用 Router

```text
python/sglang/srt/debug_utils/schedule_simulator/simulator.py
函数：Simulator._route_requests()
```

核心逻辑：

```text
for req in incoming_requests:
    gpu_id = self.router.route(req)
    if gpu_id < self.num_gpus_per_engine:
        self.gpu_states[gpu_id].pending_requests.append(req)
```

完整模拟流程：

```mermaid
flowchart TD
  A["incoming SimRequest list"] --> B["_route_requests()"]
  B --> C["router.route(req) -> gpu_id"]
  C --> D["GPUState[gpu_id].pending_requests.append(req)"]
  D --> E["_schedule_all_gpus()"]
  E --> F["scheduler.schedule(gpu)"]
  F --> G["_execute_step()"]
  G --> H["gpu.execute_step()"]
  H --> I["record metrics / step records"]
```

这套 router 是理解真实 Scheduler 之前的实验台。它把“请求应该进入哪个 GPU 队列”这个问题单独抽出来，便于对比不同策略。

---

## 5. PD Bootstrap /route：分离部署中的连接路由

在 PD 分离中，Prefill 侧和 Decode 侧需要知道彼此的 rank、端口、page size、KV cache dtype 等信息。这里也出现了 route。

关键位置：

```text
python/sglang/srt/disaggregation/common/conn.py
函数：_setup_routes()
函数：_handle_route()
函数：_handle_route_put()
函数：_handle_route_get()
```

### 5.1 route API 注册

```text
self.app.router.add_route("*", "/route", self._handle_route)
self.app.router.add_post("/register_dp_rank", self._handle_register_dp_rank)
self.app.router.add_post("/query_dp_ranks", self._handle_query_dp_ranks)
self.app.router.add_get("/health", self._handle_health_check)
```

这不是模型请求的路由，而是 bootstrap server 的 HTTP route。

### 5.2 PUT /route：注册 rank 信息

`_handle_route_put()` 从请求里读取：

```text
attn_tp_size
attn_tp_rank
attn_cp_size
attn_cp_rank
attn_dp_size
attn_dp_rank
pp_size
pp_rank
system_dp_size
system_dp_rank
rank_ip
rank_port
page_size
kv_cache_dtype
```

这些信息会进入内部表：

```text
prefill_port_table
room_to_dp_rank
```

它服务的是第 7 讲提到的 bootstrap / prealloc / transfer 流程：Decode 侧需要根据 room、rank、并行维度，找到对应 Prefill 侧连接点。

### 5.3 与第 7 讲的关系

PD 分离里的 `/route` 可以理解成：

```text
不是“把用户请求路由到哪个模型”
而是“让不同分布式 rank 找到彼此”
```

它处在控制面，负责连接信息，不直接参与 token forward。

---

## 6. MoE Expert Router：模型内部的 token 路由

这一部分是本讲最重要的源码。

MoE router 的输入输出是：

```text
输入：
hidden_states: [num_tokens, hidden_dim]
router_weight: [num_experts, hidden_dim]

输出：
topk_weights: [num_tokens, topk]
topk_ids: [num_tokens, topk]
```

含义：

- `topk_ids[i]`：第 i 个 token 被路由到哪些 expert。
- `topk_weights[i]`：第 i 个 token 对应 expert 的 combine 权重。

### 6.1 MoE Router 总流程图

```mermaid
flowchart TD
  A["hidden_states<br/>[tokens, hidden_dim]"] --> B["router linear weight<br/>[experts, hidden_dim]"]
  B --> C["router logits = hidden @ weight.T"]
  A --> C
  C --> D{"moe_softcapping > 0?"}
  D -- "是" --> E["tanh-like softcap"]
  D -- "否" --> F["raw logits"]
  E --> G["add correction_bias optional"]
  F --> G
  G --> H["top-k selection"]
  H --> I["topk_ids"]
  H --> J["topk_weights"]
  I --> K["_post_process_topk_ids()"]
  J --> K
  K --> L["FusedMoE expert execution"]
```

### 6.2 FusedMoeRouter 类

```text
python/sglang/srt/layers/moe/router.py
类：FusedMoeRouter
```

它包装了：

- `router_linear`：gate/router 线性层；
- `topk`：每个 token 选几个 expert；
- `moe_softcapping`：router logits 的 softcap 参数。

入口：

```text
FusedMoeRouter.forward(x, residual)
```

分支：

```text
if x.is_cuda:
    return self.forward_cuda(x, residual)
else:
    return self.forward_vllm(x, residual)
```

当前文件中核心实现是 `forward_cuda()`：

```text
python/sglang/srt/layers/moe/router.py
函数：FusedMoeRouter.forward_cuda()

调用：
fused_moe_router_shim(
    moe_softcapping=self.moe_softcapping,
    hidden_states=x,
    gating_output=self.router_linear.weight,
    topk=self.topk,
    renormalize=False,
)
```

注意：这里传入的 `gating_output` 实际是 router linear 的权重。

### 6.3 fused_moe_router_shim()

```text
python/sglang/srt/layers/moe/router.py
函数：fused_moe_router_shim()
```

这个函数决定使用哪个 Triton kernel：

```text
if (bs >= 512 or num_experts > 8)
   and hidden_dim % BLOCK_SIZE_K == 0
   and not enable_deterministic_inference:
    使用 fused_moe_router_tensorcore()
else:
    使用 fused_moe_router_cudacore()
```

可以理解为：

- batch 大或 expert 多：用 tensorcore 路径，吞吐更好。
- batch 小、shape 不适合、要求 deterministic：用 cudacore 路径，逻辑更直接。

### 6.4 cudacore kernel

```text
python/sglang/srt/layers/moe/router.py
函数：fused_moe_router_cudacore_kernel()
```

它的粒度是：

```text
一个 Triton program 处理一个 token
```

核心步骤：

1. 根据 `pid` 取当前 token 的 hidden vector。
2. 加载所有 expert 的 router weight。
3. 对每个 expert 计算 dot product。
4. 对 logits 做 softcap。
5. 加 correction bias。
6. 选择 top1 / top2 / topk。
7. 写出 `topk_ids` 和 `topk_weights`。

伪代码：

```text
for token in tokens:
    x = hidden_states[token]
    logits = []
    for expert in experts:
        logits[expert] = dot(x, router_weight[expert])

    logits = softcap(logits)
    logits = logits + correction_bias
    ids, weights = topk_softmax(logits, topk)
```

这个 kernel 的优点是简单直接，适合较小 batch 或需要避免 tensorcore 路径的场景。

### 6.5 tensorcore kernel

```text
python/sglang/srt/layers/moe/router.py
函数：fused_moe_router_tensorcore_kernel()
```

它把 router 计算看成矩阵乘：

```text
A = hidden_states       [bs, hidden_dim]
B = router_weight       [num_experts, hidden_dim]
logits = A @ B.T        [bs, num_experts]
```

kernel 以 block 为单位处理多个 token 和多个 expert：

- `BLOCK_SIZE_M`：token 维度 block；
- `BLOCK_SIZE_N`：expert 维度 block；
- `BLOCK_SIZE_K`：hidden_dim 维度 block。

流程：

```text
1. 建立 A tile 和 B tile 指针
2. 循环 K 维度做 tl.dot()
3. 得到 logits block
4. softcap
5. correction bias
6. DP attention workaround：把 NaN 替换成极小值
7. top1 / top2
8. 写出 topk_ids / topk_weights
```

这个路径目前只支持 `topk <= 2`。因此 `fused_moe_router_tensorcore()` 里有：

```text
assert topk <= 2
```

### 6.6 softcap 是什么

router logits 可能过大，softcap 会把它压到一个受控范围：

```text
logits_scaled = logits / moe_softcapping
logits_softcapped = tanh(logits_scaled) * moe_softcapping
```

源码中用指数形式近似实现：

```text
exped = tl.exp(2 * logits_scaled)
logits_softcapped = (exped - 1) / (exped + 1) * moe_softcapping
```

作用是让 router logits 不至于过分尖锐，稳定 top-k 选择和权重分布。

---

## 7. MoE Router 在模型中的接入示例

以 `grok.py` 为例：

```text
python/sglang/srt/models/grok.py
代码段：custom_routing_function = functools.partial(fused_moe_router_shim, self.router_logit_softcapping)
```

模型里会创建：

```text
self.gate = ReplicatedLinear(...)
self.router_logit_softcapping = 30.0
custom_routing_function = functools.partial(
    fused_moe_router_shim, self.router_logit_softcapping
)
self.topk = TopK(
    top_k=top_k,
    renormalize=False,
    layer_id=layer_id,
    custom_routing_function=None if _is_npu else custom_routing_function,
)
self.experts = FusedMoE(...)
```

这里的结构是：

```mermaid
flowchart LR
  X["hidden_states"] --> Gate["gate / router linear"]
  Gate --> TopK["TopK<br/>custom_routing_function"]
  TopK --> IDs["topk_ids"]
  TopK --> Weights["topk_weights"]
  IDs --> FusedMoE["FusedMoE"]
  Weights --> FusedMoE
  X --> FusedMoE
```

也就是说，MoE router 不是一个孤立模块，它通常被 `TopK` 或模型自定义路径调用，输出再交给 `FusedMoE`。

---

## 8. TopK 后处理：路由结果还要再整理

router kernel 给出 `topk_ids` 和 `topk_weights` 后，SGLang 还会做后处理。

关键位置：

```text
python/sglang/srt/layers/moe/topk.py
函数：_post_process_topk_ids()
```

这段逻辑做几件事：

1. 如果启用了 routed experts 捕获，就调用 capturer 保存 `topk_ids`。
2. CUDA 路径下根据 expert location dispatch info 做 expert ID 映射。
3. DeepEP 场景下区分 routed experts 和 fused shared experts。
4. 如果存在 fused shared experts，要处理额外的 shared expert 列。

对应代码段：

```text
if (cap := get_global_experts_capturer()) is not None:
    cap.capture(layer_id=layer_id, topk_indices=topk_ids)
```

这一行把“模型内部 router 的结果”接到了“用户可观测 metadata”链路上。

---

## 9. RoutedExpertsCapturer：把 expert 路由结果返回给用户

如果请求设置：

```text
return_routed_experts = True
routed_experts_start_len = N
```

SGLang 可以把每个 token 在每层 MoE 中被路由到的 experts 返回。

### 9.1 请求字段入口

```text
python/sglang/srt/entrypoints/openai/protocol.py
字段：return_routed_experts
字段：routed_experts_start_len

python/sglang/srt/managers/io_struct.py
字段：return_routed_experts
字段：routed_experts_start_len
```

`routed_experts_start_len` 表示从哪个 token 位置开始返回 routed experts，避免返回过长 prompt 的全部路由信息。

### 9.2 Capturer 初始化

```text
python/sglang/srt/model_executor/model_runner.py
代码段：RoutedExpertsCapturer.create(...)
```

它会根据模型配置创建 buffer：

```text
num_layers = model_config.hf_text_config.num_hidden_layers
topk_size = model_config.hf_text_config.num_experts_per_tok
max_batch_size = max(chunked_prefill_size * dp_size, max_running_requests * dp_size)
```

### 9.3 Capturer 捕获流程

```text
python/sglang/srt/state_capturer/routed_experts.py
类：RoutedExpertsCapturer
函数：capture(layer_id, topk_indices)
```

普通场景：

```text
topk_indices -> BaseTopkCapturer.capture() -> device cache
```

DeepEP 场景：

```text
local_topk -> attn_tp_all_gather_into_tensor() -> full topk -> device cache
```

为什么 DeepEP 要 all-gather？

因为 DeepEP a2a 路径中，每个 attention TP rank 只看到被 scatter 后的一部分 token / expert 信息。为了让后续按请求取回完整 routed experts，捕获时要先把 attn TP 切片聚合回来。

### 9.4 请求结束时收集

```text
python/sglang/srt/managers/scheduler_components/batch_result_processor.py
函数：BatchResultProcessor._maybe_collect_routed_experts()
```

逻辑：

```text
if not req.return_routed_experts:
    return
capturer = get_global_experts_capturer()
if capturer is None:
    return
req.routed_experts = capturer.get_topk(
    req_pool_idx=req.req_pool_idx,
    seqlen=seqlen,
    req_to_token_pool=self.req_to_token_pool,
    start_len=req.routed_experts_start_len,
)
```

这里用 `req_pool_idx` 和 `req_to_token_pool` 把“请求维度”映射回“token cache 位置”，再从 capturer 的 top-k buffer 中取出该请求对应的 rows。

### 9.5 Detokenizer 编码输出

```text
python/sglang/srt/managers/detokenizer_manager.py
函数：DetokenizerManager._b64_encode_per_request()
函数：DetokenizerManager.handle_batch_token_id_out()
```

routed experts tensor 最后被转成 base64：

```text
pybase64.b64encode(item.numpy().tobytes()).decode("utf-8")
```

然后进入：

```text
BatchStrOutput.routed_experts
```

OpenAI 层再把它放到响应扩展字段中。

---

## 10. routed_experts 完整返回流程

```mermaid
sequenceDiagram
  participant Client
  participant API as OpenAI API
  participant TM as TokenizerManager
  participant Sch as Scheduler
  participant MR as ModelRunner
  participant TopK as MoE TopK / Router
  participant Cap as RoutedExpertsCapturer
  participant BRP as BatchResultProcessor
  participant Detok as DetokenizerManager

  Client->>API: request(return_routed_experts=True)
  API->>TM: GenerateReqInput
  TM->>Sch: TokenizedGenerateReqInput
  Sch->>Sch: Req.return_routed_experts=True
  Sch->>MR: ForwardBatch
  MR->>TopK: model forward
  TopK->>TopK: router selects topk_ids
  TopK->>Cap: capture(layer_id, topk_ids)
  MR-->>Sch: Batch result
  Sch->>BRP: process finished req
  BRP->>Cap: get_topk(req_pool_idx, seqlen, start_len)
  Cap-->>BRP: routed_experts tensor
  BRP-->>Detok: BatchTokenIDOutput.routed_experts
  Detok->>Detok: base64 encode per request
  Detok-->>TM: BatchStrOutput.routed_experts
  TM-->>API: meta_info.routed_experts
  API-->>Client: response extension
```

这条链路把模型内部的 MoE 路由决策暴露给外部用户，适合做：

- MoE expert 分布分析；
- 请求级 expert 负载诊断；
- DeepEP / EPLB 调试；
- 模型行为观测。

---

## 11. MoE Router 与 EP / DeepEP / EPLB 的关系

MoE router 输出的是 logical expert IDs，但执行层可能需要 physical expert IDs。

这里有几个概念：

```text
Logical expert:
  模型语义上的 expert 编号。

Physical expert:
  实际放在某个 rank / GPU 上的 expert 副本或位置。

EP:
  Expert Parallel，把 experts 分布到不同 rank。

DeepEP:
  更高性能的 expert parallel 通信路径，涉及 token scatter / gather。

EPLB:
  Expert Parallel Load Balancer，可能调整 logical expert 到 physical expert 的映射。
```

`_post_process_topk_ids()` 中的 expert location dispatch 逻辑，就是把 router 选出来的 expert IDs 调整到执行后端能理解的位置。

简化流程：

```text
router logits
-> topk logical expert ids
-> optional EPLB logical-to-physical remap
-> optional DeepEP scatter/gather
-> FusedMoE execution
```

所以 MoE router 不只是一层 `topk`，它和分布式 expert 布局紧密相关。

---

## 12. 四类 Router 的对照总结

| Router 类型 | 输入 | 输出 | 是否参与真实 forward | 核心文件 |
| --- | --- | --- | --- | --- |
| SmartRouter | 用户 prompt / messages | local 或 remote backend | 否，只在入口转发 | `entrypoints/ollama/smart_router.py` |
| Simulator Router | `SimRequest` | `gpu_id` | 否，调度实验工具 | `debug_utils/schedule_simulator/routers/` |
| PD `/route` | rank / room / dp/tp/pp 信息 | prefill rank 连接表 | 不直接 forward，服务 PD 控制面 | `disaggregation/common/conn.py` |
| MoE Expert Router | hidden states | topk expert IDs / weights | 是，模型 forward 内部路径 | `layers/moe/router.py` |
| RoutedExpertsCapturer | topk expert IDs | response metadata | 不改变计算，只做观测 | `state_capturer/routed_experts.py` |

---

## 13. 推荐阅读顺序

如果你要系统阅读 router 相关源码，建议按下面顺序：

1. `python/sglang/srt/entrypoints/ollama/smart_router.py` / `SmartRouter`  
   先理解最直观的请求级 router。

2. `python/sglang/srt/debug_utils/schedule_simulator/routers/base.py` / `RouterPolicy`  
   理解 router 抽象：输入请求，输出目标 GPU。

3. `python/sglang/srt/debug_utils/schedule_simulator/simulator.py` / `_route_requests()`  
   看 router 如何影响 pending queue。

4. `python/sglang/srt/disaggregation/common/conn.py` / `_setup_routes()`、`_handle_route_put()`  
   理解 PD 分离里的 route 是 rank 连接注册，不是普通请求调度。

5. `python/sglang/srt/layers/moe/router.py` / `fused_moe_router_shim()`  
   理解 MoE router 如何选择 cudacore / tensorcore kernel。

6. `python/sglang/srt/layers/moe/router.py` / `fused_moe_router_cudacore_kernel()`  
   先读更直观的逐 token kernel。

7. `python/sglang/srt/layers/moe/router.py` / `fused_moe_router_tensorcore_kernel()`  
   再读 block GEMM 风格的高性能 kernel。

8. `python/sglang/srt/layers/moe/topk.py` / `_post_process_topk_ids()`  
   看 top-k expert IDs 如何与 DeepEP、EPLB、capturer 接上。

9. `python/sglang/srt/state_capturer/routed_experts.py` / `RoutedExpertsCapturer`  
   看 routed experts 如何被保存和按请求取回。

10. `python/sglang/srt/managers/scheduler_components/batch_result_processor.py` / `_maybe_collect_routed_experts()`  
    看请求结束时如何收集路由结果。

---

## 14. 阅读任务

### 任务 1：解释 SmartRouter 的失败回退

阅读：

```text
python/sglang/srt/entrypoints/ollama/smart_router.py
函数：SmartRouter.chat()
```

回答：

```text
如果 judge 判断请求应该走 Remote SGLang，
但 remote_client.chat() 抛出异常，
SmartRouter 会如何处理？
返回的 reason 字段是什么？
```

### 任务 2：比较三种模拟 router

阅读：

```text
random_router.py
round_robin_router.py
sticky_router.py
```

回答：

```text
如果有 100 个请求属于同一个 group_id，
三种 router 会如何分配它们？
哪一种最可能保留 prefix locality？
哪一种最可能造成单 GPU 热点？
```

### 任务 3：追踪 MoE router 输出

从下面函数开始：

```text
python/sglang/srt/layers/moe/router.py
函数：fused_moe_router_shim()
```

追踪到：

```text
python/sglang/srt/layers/moe/topk.py
函数：_post_process_topk_ids()
```

回答：

```text
topk_ids 在进入 FusedMoE 执行前，可能经历哪些后处理？
为什么 DeepEP 和 fused shared experts 会让这个步骤变复杂？
```

### 任务 4：追踪 routed_experts 返回

从请求字段：

```text
return_routed_experts=True
```

追踪到响应字段：

```text
meta_info.routed_experts
```

至少经过：

```text
protocol.py
io_struct.py
topk.py
routed_experts.py
batch_result_processor.py
detokenizer_manager.py
tokenizer_manager.py
```

目标是讲清楚：模型内部的 top-k expert IDs 是如何变成响应里的 base64 字符串的。

---

## 15. 本讲心智模型

最后用一句话总结：

> SGLang 里的 router 不是一个模块，而是一组“选择器”：有的选择 backend，有的选择 GPU 队列，有的选择 PD rank，有的选择 MoE expert。

如果只记一个公式，可以记这个：

```text
请求级 router:
  request -> backend / engine

调度实验 router:
  SimRequest -> gpu_id

PD route:
  rank metadata -> connection table

MoE router:
  hidden_states -> topk_ids + topk_weights

routed experts capturer:
  topk_ids -> response metadata
```

读源码时先确认 router 的输入输出，再看它是否参与真实 forward。这样你会发现：

- SmartRouter 是入口层示例；
- Simulator router 是实验工具；
- PD `/route` 是分布式连接控制面；
- MoE router 才是模型执行路径中的性能关键点。

---

## 16. 下一讲预告

下一讲可以继续看 **Structured Output / Grammar 约束生成**：

- regex / JSON schema / grammar 如何进入请求；
- grammar backend 如何初始化；
- Scheduler 如何维护每个请求的 grammar 状态；
- sampling 阶段如何用 grammar mask 限制 token；
- grammar 约束如何影响 streaming、batching 和 speculative decoding。
