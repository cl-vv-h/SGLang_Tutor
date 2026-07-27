# 第十五讲：LayerCommunicator、HCCL 与层边界通信

本讲聚焦 SGLang 模型层内部最容易被忽略的一类代码：`LayerCommunicator`。它频繁出现在 GLM-4.7-Flash、DeepSeek-V2/V3/R1 类模型、GPT-OSS MoE、Bailing MoE 等 decoder layer 中，用来统一处理 residual、RMSNorm、TP/DP/CP/MoE 之间的数据分布转换和跨卡 collective。

先给出结论：

`LayerCommunicator` 是一个通用的层边界通信与 residual 状态机，不是 GLM 专属类，也不是 Ascend NPU 专属类。它的源码位于 `python/sglang/srt/layers/communicator.py`，不同模型只要采用类似的 decoder layer 结构，就可以把 attention、MLP/MoE、RMSNorm 和并行通信交给它编排。

但它也不是最底层的 HCCL wrapper。真正调用 HCCL 的路径通常是：

```text
LayerCommunicator
  -> communication_op.py 中的 tensor/attention/moe collective wrapper
  -> parallel_state.py 中的 GroupCoordinator
  -> device_communicators/npu_communicator.py 中的 NpuCommunicator
  -> torch.distributed collective
  -> torch_npu ProcessGroupHCCL / HCCL
```

因此，阅读 `LayerCommunicator` 时要同时看两条线：

1. **层语义线**：`hidden_states`、`residual`、RMSNorm、attention、MLP/MoE 之间怎样交接。
2. **通信执行线**：all-reduce、all-gather、reduce-scatter、quant all-reduce 怎样落到 HCCL 或 NPU tensor op。

## 1. 源码定位

| 文件 | 关键对象 | 职责 |
|---|---|---|
| `python/sglang/srt/layers/communicator.py` | `LayerCommunicator`、`LayerScatterModes`、`ScatterMode`、`Communicate*Fn` | 层边界状态机。根据输入/输出分布模式选择通信函数，并在边界处执行 RMSNorm、all-reduce、all-gather、reduce-scatter。 |
| `python/sglang/srt/models/glm4_moe_lite.py` | `Glm4MoeLiteDecoderLayer` | GLM-4.7-Flash 使用 `LayerCommunicator` 的典型样例。 |
| `python/sglang/srt/models/deepseek_v2.py` | `DeepseekV2DecoderLayer` | DeepSeek 系列和 MLA/MoE 路径中复用同一套 layer communicator 机制。 |
| `python/sglang/srt/models/gpt_oss.py` | `GptOssDecoderLayer` | MoE 模型复用示例。 |
| `python/sglang/srt/models/bailing_moe.py` | `BailingMoeDecoderLayer` | MoE 模型复用示例，并包含 last layer output capture 场景。 |
| `python/sglang/srt/layers/communicator_dsa_cp.py` | `DSACPLayerCommunicator` | DSA context parallel 场景下的派生实现。 |
| `python/sglang/srt/layers/dp_attention.py` | `attn_tp_all_gather_into_tensor`、`attn_tp_reduce_scatter_tensor`、`dp_gather_partial`、`dp_scatter` | DP attention、attention TP、attention CP 相关 token gather/scatter 工具。 |
| `python/sglang/srt/distributed/communication_op.py` | `attention_tensor_model_parallel_all_reduce`、`moe_tensor_model_parallel_all_reduce` | 把模型侧调用转成不同 process group 上的 collective。 |
| `python/sglang/srt/distributed/parallel_state.py` | `GroupCoordinator`、`initialize_model_parallel` | 创建 TP/ATTN_TP/ATTN_CP/MOE_TP/MOE_EP 等 group，并决定 collective 调用路径。 |
| `python/sglang/srt/distributed/device_communicators/npu_communicator.py` | `NpuCommunicator` | NPU 设备通信 wrapper，最终调用 `torch.distributed`，由 HCCL 执行。 |
| `python/sglang/srt/layers/layernorm.py` | `RMSNorm.forward_npu` | 在 NPU 上绑定 `torch_npu.npu_rms_norm` 和 `torch_npu.npu_add_rms_norm`。 |
| `python/sglang/srt/hardware_backend/npu/cmo.py` | `prepare_weight_cache` | 通信后可触发 NPU prefetch，用 `torch_npu.npu_prefetch` 提前搬运后续 matmul 权重。 |

## 2. 它是不是各模型通用类

是，但要限定“通用”的范围。

`LayerCommunicator` 通用于一类 decoder-only 或 MoE decoder layer：这些 layer 通常包含 attention block、FFN/MoE block、两个 RMSNorm、residual stream，以及 TP/DP/CP/EP 等并行策略。GLM-4.7-Flash 正好属于这一类，所以每个 `Glm4MoeLiteDecoderLayer` 都会创建一个 `LayerCommunicator`。

它不要求模型一定是 GLM，也不要求 attention 一定是 MLA。只要模型实现愿意把下面三段边界交给它，就可以复用：

```text
上一层输出 pair
  -> prepare_attn()
  -> attention
  -> prepare_mlp()
  -> MLP / MoE
  -> postprocess_layer()
  -> 下一层输入 pair
```

不过，它不是所有模型的唯一通信入口。更简单的模型可能直接在 layer forward 中调用 RMSNorm、attention、MLP 和 TP all-reduce；特殊 CP 或 DSA 路径可能使用 `DSACPLayerCommunicator`；DeepEP/FuseEP 的 expert dispatch/combine 也不是由它完成，而是由 MoE backend 自己处理。

可以把它和 `NpuCommunicator` 这样区分：

| 对象 | 层次 | 主要问题 | 是否理解模型结构 |
|---|---|---|---|
| `LayerCommunicator` | 模型层边界 | 什么时候 add residual，什么时候 RMSNorm，什么时候 gather/scatter/all-reduce，下一阶段需要什么 token 分布 | 是。它知道 attention 前、MLP 前、layer 输出这些边界语义 |
| `GroupCoordinator` | 分布式 group 封装 | 这个 collective 应该走哪个 process group，是否走 custom op、pynccl、NPU communicator 或 torch distributed | 否。它只关心 group 和 collective |
| `NpuCommunicator` | 设备通信 wrapper | NPU 上怎样执行 all-reduce、quant all-reduce、all-gather | 否。它只关心 tensor 和 HCCL group |

## 3. 为什么模型层需要这个类

先忽略 TP/DP/CP/EP，一层 decoder 的数学语义可以写成：

```text
输入: h_l [N,D]

a_l = Attention(RMSNorm_attn(h_l))      # [N,D]
u_l = h_l + a_l                         # [N,D]
f_l = MLP_or_MoE(RMSNorm_ffn(u_l))       # [N,D]
h_(l+1) = u_l + f_l                     # [N,D]
```

在 GLM-4.7-Flash 中，`D=2048`。prefill 时 `N=T`，表示本次 extend 处理的 token 数；decode 时 `N=B`，表示 batch 中每个请求的新 token 数。

上面的公式很干净，但真实推理服务要面对更多状态：

- attention 的 `o_proj` 可能是 row-parallel，本 rank 只有局部 partial，需要跨 TP ranks all-reduce；
- DP attention 或 context parallel 可能让每张卡只拿到一段 token，需要 all-gather 或 reduce-scatter；
- MoE 可能需要普通 TP 汇总，也可能由 DeepEP/FuseEP 做 token dispatch/combine；
- residual add 可以和 RMSNorm 合成一个 NPU op，避免单独执行 add；
- TBO/SBO 会把一个 layer 拆成多个可调度 op，要求通信边界可复用；
- NPU Graph capture 希望 forward 的执行路径尽量稳定。

`LayerCommunicator` 的价值就是把这些“层边界上的麻烦”收拢起来，让模型文件只表达高层结构：

```python
hidden_states, residual = self.layer_communicator.prepare_attn(...)
hidden_states = self.self_attn(...)
hidden_states, residual = self.layer_communicator.prepare_mlp(...)
hidden_states = self.mlp(...)
hidden_states, residual = self.layer_communicator.postprocess_layer(...)
```

## 4. `hidden_states` 与 `residual` 的 pair 语义

`LayerCommunicator` 的输入输出经常是 `(hidden_states, residual)`，这不是两个相互独立的模型分支，而是为了延迟物化 residual add。

在一层内部可以这样理解：

| 时刻 | `hidden_states` | `residual` |
|---|---|---|
| 进入当前层 | 上一层留下的增量或 embedding 输出 | 上一层留下的 residual；Layer 0 通常是 `None` |
| `prepare_attn()` 后 | attention 前 RMSNorm 结果 | attention block 的 residual 基底 |
| attention 后 | attention 产生的增量，可能是 TP partial | attention 前 residual |
| `prepare_mlp()` 后 | MLP/MoE 前 RMSNorm 结果 | `旧 residual + attention 增量` |
| MLP/MoE 后 | MLP/MoE 产生的增量，可能还需要通信 | attention residual block 输出 |
| `postprocess_layer()` 后 | 传给下一层的 hidden 增量 | 传给下一层的 residual |

当下一层再次进入 `prepare_attn()` 时，如果 `residual is not None`，NPU 路径会通过 `RMSNorm.forward_npu()` 调用：

```python
torch_npu.npu_add_rms_norm(residual, x, weight, eps)
```

其数学语义是：

```text
residual_out = residual + x        # [N,D]
normed       = RMSNorm(residual_out, weight)
```

返回值中 `normed` 会作为下一段计算的输入，`residual_out` 会继续沿 residual stream 传递。这个设计解释了为什么 profiler 中经常看不到单独的 residual add kernel：它被融合进 `npu_add_rms_norm` 了。

## 5. `ScatterMode`：描述 token 分布，不是描述 dtype

`LayerCommunicator` 的第一层抽象是 `ScatterMode`：

```python
class ScatterMode(Enum):
    SCATTERED = auto()
    TP_ATTN_FULL = auto()
    FULL = auto()
    MOE_FULL = auto()
```

它描述的是“当前 rank 拿到了哪些 token”，不是权重 shape，也不是 hidden size 是否完整。

| 模式 | 直观含义 | 常见出现位置 |
|---|---|---|
| `SCATTERED` | 每个 rank 只持有一部分 token，多个 rank 合起来才是完整 batch/token 集合。 | DP attention、CP、DeepEP/FuseEP、`enable_attn_tp_input_scattered`。 |
| `TP_ATTN_FULL` | attention TP group 内每个 rank 都拿到该 attention group 需要的完整 token。 | attention 输入通常需要这个模式，因为不同 TP rank 计算不同 heads，但 token 序列要一致。 |
| `FULL` | 普通 TP/MLP/MoE group 视角下的完整 token 集合。 | dense MLP、普通 fused MoE。 |
| `MOE_FULL` | MoE CP group 视角下的完整 token 集合。 | `attn_cp_size > moe_dp_size` 时，MoE 前跨 CP ranks all-gather。 |

`LayerScatterModes` 有五个字段：

```python
@dataclass
class LayerScatterModes:
    layer_input_mode: ScatterMode
    attn_mode: ScatterMode
    mlp_mode: ScatterMode
    middle_residual_mode: ScatterMode
    layer_output_mode: ScatterMode
```

它们分别描述五个边界：

| 字段 | 含义 | 影响的方法 |
|---|---|---|
| `layer_input_mode` | 本层刚收到的 pair 是什么 token 分布。 | `prepare_attn()` |
| `attn_mode` | attention 需要什么 token 分布。源码中通常固定为 `TP_ATTN_FULL`。 | `prepare_attn()`、attention backend |
| `mlp_mode` | MLP/MoE 需要什么 token 分布。 | `prepare_mlp()`、MoE backend |
| `middle_residual_mode` | attention residual block 输出后 residual 的分布。 | `prepare_mlp()`、`postprocess_layer()` |
| `layer_output_mode` | 当前层结束后交给下一层的 pair 分布。 | `postprocess_layer()` |

构造 `LayerScatterModes` 时，源码会根据这些信息推导：

- 当前层是不是 sparse MoE；
- 上一层和下一层是不是 sparse MoE；
- 是否启用 DeepEP/FuseEP 等 A2A MoE backend；
- 是否启用 MoE CP all-gather；
- `moe_dense_tp_size` 是否让 dense MLP 退化成 fully-DP；
- 当前是否是首层或末层。

在 GLM-4.7-Flash 的常见基线中，假设 `TP=4`、`DP=1`、`CP=1`、不开 DeepEP/FuseEP、`moe_dense_tp_size=None`，那么多数层的模式可以理解为：

| 层 | `layer_input_mode` | `attn_mode` | `mlp_mode` | `middle_residual_mode` | `layer_output_mode` |
|---|---|---|---|---|---|
| Layer 0 dense | `TP_ATTN_FULL` | `TP_ATTN_FULL` | `FULL` | `TP_ATTN_FULL` | `TP_ATTN_FULL` |
| Layer 1-46 MoE | `TP_ATTN_FULL` | `TP_ATTN_FULL` | `FULL` | `TP_ATTN_FULL` | `TP_ATTN_FULL` |

在这条基线里，`FULL` 和 `TP_ATTN_FULL` 的 group size 都等于 4，所以很多转换不会改变 tensor 的物理 shape。但保留两个模式名很重要，因为打开 DP attention、CP 或 MoE CP 后，二者就可能对应不同 group。

## 6. 构造期：把模式编译成三个函数

`LayerCommunicator.__init__()` 持有五类状态：

| 字段 | 来源 | 作用 |
|---|---|---|
| `layer_scatter_modes` | 模型 layer 构造阶段传入 | 描述本层各边界的数据分布。 |
| `input_layernorm` | 模型 layer 的 attention 前 RMSNorm | `prepare_attn()` 中使用。 |
| `post_attention_layernorm` | 模型 layer 的 MLP 前 RMSNorm | `prepare_mlp()` 中使用。 |
| `allow_reduce_scatter` | 模型显式传入 | 允许把某些 all-reduce 改成 reduce-scatter。 |
| `qkv_latent_func` | attention 对象提供 | MLA 等模型可提前缓存 Q/KV latent，减少重复计算或推迟 all-gather。 |

初始化时它会先创建 `CommunicateContext`：

```python
self._context = CommunicateContext.init_new()
```

`CommunicateContext` 会读取当前进程的分布式状态：

```text
attn_tp_rank / attn_tp_size
attn_dp_size
attn_cp_rank / attn_cp_size
tp_rank / tp_size
moe_cp_size
```

然后把不同 `ScatterMode` 映射成 group size：

```python
process_group_sizes = {
    ScatterMode.SCATTERED: 1,
    ScatterMode.TP_ATTN_FULL: attn_tp_size,
    ScatterMode.FULL: tp_size // attn_cp_size,
    ScatterMode.MOE_FULL: tp_size // (attn_cp_size // moe_cp_size),
}
```

接着 `_post_init_communicate()` 会选择三个函数：

```python
self._communicate_simple_fn = CommunicateSimpleFn.get_fn(...)
self._communicate_with_all_reduce_and_layer_norm_fn = CommunicateWithAllReduceAndLayerNormFn.get_fn(...)
self._communicate_summable_tensor_pair_fn = CommunicateSummableTensorPairFn.get_fn(...)
```

三者分别对应：

| 函数槽位 | 对应阶段 | 典型动作 |
|---|---|---|
| `_communicate_simple_fn` | `prepare_attn()` 末尾 | `SCATTERED -> TP_ATTN_FULL` 时 all-gather；同 group size 时 no-op。 |
| `_communicate_with_all_reduce_and_layer_norm_fn` | `prepare_mlp()` | attention TP all-reduce，residual add，post-attention RMSNorm，必要时 DP gather/scatter。 |
| `_communicate_summable_tensor_pair_fn` | `postprocess_layer()` | MLP/MoE 输出后 scatter、gather、reduce-scatter 或 no-op。 |

这一步很像把“模式组合”预编译成“执行函数”。forward 中不需要反复做复杂分支，只要调用预先选好的函数。

## 7. 执行期第一段：`prepare_attn()`

`prepare_attn()` 负责把上一层输出变成 attention 输入。

主线逻辑是：

```text
输入 pair: hidden_states, residual
  -> 如果启用 attn input scattered，先做 TP reduce-scatter
  -> 如果 residual is None，初始化 residual = hidden_states
  -> 否则做 residual add + input RMSNorm
  -> 必要时把 token 分布转成 TP_ATTN_FULL
  -> 如果提供 qkv_latent_func，注册 AttentionInputs
  -> 返回 attention 输入 hidden_states 和 residual
```

### 7.1 首层和普通层的差别

Layer 0 刚进入时通常是：

```text
hidden_states = embedding(input_ids)   # [N,2048]
residual = None
```

这时 `prepare_attn()` 会把 `residual` 设为原始 hidden，并对 hidden 做 input RMSNorm：

```text
residual = hidden_states
hidden_states = RMSNorm(hidden_states)
```

在 NPU 上，这会落到：

```text
RMSNorm.forward_npu
  -> torch_npu.npu_rms_norm
```

从 Layer 1 开始，经常是：

```text
hidden_states = 上一层 MLP/MoE 增量
residual = 上一层 attention residual block 输出
```

这时 `prepare_attn()` 会执行：

```text
residual_out = residual + hidden_states
hidden_states = RMSNorm(residual_out)
residual = residual_out
```

在 NPU 上通常落到：

```text
RMSNorm.forward_npu
  -> torch_npu.npu_add_rms_norm
```

### 7.2 `SCATTERED -> TP_ATTN_FULL` 的 all-gather

如果 `layer_input_mode=SCATTERED`、`attn_mode=TP_ATTN_FULL`，`CommunicateSimpleFn` 会选择 `_scattered_to_tp_attn_full()`。

它的典型调用链是：

```text
LayerCommunicator.prepare_attn
  -> CommunicateSimpleFn._scattered_to_tp_attn_full
  -> attn_tp_all_gather_into_tensor
  -> get_attention_tp_group().all_gather_into_tensor
  -> GroupCoordinator.all_gather_into_tensor
  -> torch.distributed.all_gather_into_tensor
  -> HCCL AllGather
```

输出 token 数会从本 rank 的局部 token 数变成 attention TP group 内的完整 token 数。hidden size `D` 不变。

### 7.3 `qkv_latent_func` 与 MLA 的延迟 all-gather

GLM-4.7-Flash 构造 `LayerCommunicator` 时传入：

```python
qkv_latent_func=self.self_attn.prepare_qkv_latent
```

`prepare_attn()` 末尾会创建 `AttentionInputs`，注册到全局 `AttnTpContext`：

```python
attn_inputs = AttentionInputs(hidden_states, forward_batch, self.qkv_latent_func)
get_attn_tp_context().set_attn_inputs(attn_inputs)
```

后续 NPU MLA prepare 可以通过：

```python
get_attn_tp_context().fetch_qkv_latent()
```

取回提前算好的 Q/KV latent。

这对 MLA 很重要。GLM 的主 hidden 是 `[N,2048]`，但 Q/KV latent 合计宽度是：

```text
q_lora_rank + kv_lora_rank + qk_rope_head_dim
= 768 + 512 + 64
= 1344
```

当启用 `enable_attn_tp_input_scattered` 且条件满足时，可以先在局部 token 上算 latent，再对 `[N,1344]` 做 all-gather，而不是对 `[N,2048]` 做 all-gather。这样减少通信字节数，但控制逻辑会更复杂。

## 8. 执行期第二段：`prepare_mlp()`

attention 执行完后，`prepare_mlp()` 负责进入 MLP/MoE 前的边界处理。

在 GLM-4.7-Flash 中，attention 构造时设置了：

```python
reduce_results=False
```

这意味着 attention 的 output projection 不在 attention 内部立刻做 TP all-reduce。每个 TP rank 先得到自己的 partial hidden，随后由 `LayerCommunicator.prepare_mlp()` 统一处理。

基线 TP=4、DP=1、CP=1 时，典型路径是：

```text
attention partial hidden [N,2048]
  -> attention TP all-reduce
  -> hidden [N,2048]
  -> residual add + post_attention RMSNorm
  -> mlp_input [N,2048]
```

对应源码路径：

```text
LayerCommunicator.prepare_mlp
  -> CommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual
  -> attention_tensor_model_parallel_all_reduce
  -> get_attn_tp_group().all_reduce
  -> GroupCoordinator.all_reduce
  -> NpuCommunicator.all_reduce
  -> torch.distributed.all_reduce
  -> HCCL AllReduce
  -> RMSNorm.forward_npu
  -> torch_npu.npu_add_rms_norm
```

这里有两个底层动作：

1. **HCCL AllReduce**：把 TP ranks 的 partial attention 输出求和，让每张卡得到完整 hidden。
2. **NPU Add RMSNorm**：把 attention 输出加回 residual，并得到 MLP/MoE 输入。

它们是不同类型的执行单元。前者是跨卡通信，后者是本卡 NPU 计算算子。

### 8.1 量化通信分支

如果当前不是 decode/idle，并且启用了 `enable_quant_communications`，`prepare_mlp()` 中的 attention TP all-reduce 可能走：

```text
attention_tensor_model_parallel_quant_all_reduce
  -> get_attn_tp_group().quant_all_reduce
  -> GroupCoordinator.quant_all_reduce
  -> NpuCommunicator.quant_all_reduce
```

`NpuCommunicator.quant_all_reduce()` 的实现不是直接调用一个“量化 all-reduce”HCCL API，而是拆成：

```text
x_q, scale = torch_npu.npu_dynamic_quant(x, dst_type=torch.int8)
dist.all_gather_into_tensor(output_tensor, x_q, group)
dist.all_gather_into_tensor(output_scale, scale, group)
output_tensor = output_tensor.to(x.dtype) * output_scale
output_tensor = output_tensor.reshape(world_size, *input_size)
return output_tensor.sum(dim=0)
```

也就是说：

- `torch_npu.npu_dynamic_quant` 在本卡 NPU 上把通信 payload 压到 int8；
- HCCL AllGather 传输 int8 payload 和 scale；
- 本卡再反量化并沿 world 维求和，得到 all-reduce 语义。

在 profiler 中，这条路径会看到 dynamic quant、HCCL AllGather、反量化/乘法/求和相关 kernel，而不是一个单独的 HCCL AllReduce。

### 8.2 DP attention 分支

当 `context.attn_dp_size != 1` 时，`prepare_mlp()` 不是简单 all-reduce。源码会根据 token padding 模式选择：

```text
dp_gather_partial
dp_scatter
dp_reduce_scatter_tensor
```

这些函数位于 `layers/dp_attention.py`。它们的目标是让 DP attention 下的局部 token、全局 token buffer 和 attention TP group 之间保持一致。

典型调用包括：

```text
dp_gather_partial
  -> _dp_gather
  -> all_gather 或 all_reduce 风格的 DP gather

dp_reduce_scatter_tensor
  -> get_tp_group().reduce_scatter_tensor
  -> 必要时 get_attention_tp_group().all_gather_into_tensor
```

这也是为什么 `LayerCommunicator` 不能只写成“attention 后 all-reduce 一下”：DP attention 会改变 token 维的分布，必须同时考虑 residual 的分布。

### 8.3 NPU CMO prefetch

`prepare_mlp()` 还可能收到 `cache` 参数。NPU 分支中，如果 `context.cache is not None`，在 attention all-reduce 后会调用：

```text
prepare_weight_cache(hidden_states, context.cache)
  -> torch_npu.npu_prefetch
```

`prepare_weight_cache()` 位于 `hardware_backend/npu/cmo.py`。它会在单独的 NPU stream 上预取后续 matmul 权重，目标是在通信或其它 kernel 执行期间重叠部分内存访问时间。

这不是模型数学语义的一部分，但会影响 NPU profiling 中的 stream 和时间线。

## 9. 执行期第三段：`postprocess_layer()`

MLP/MoE 执行完后，`postprocess_layer()` 负责把当前层输出整理成下一层需要的 pair。

源码主线是：

```python
return self._communicate_summable_tensor_pair_fn(
    hidden_states=hidden_states,
    residual=residual,
    forward_batch=forward_batch,
    context=self._context,
    allow_reduce_scatter=self.allow_reduce_scatter,
)
```

可选函数包括：

| 函数 | 典型模式变化 | 动作 |
|---|---|---|
| `_trivial` | 输入和输出 group size 相同 | 不通信，直接返回 pair。 |
| `_scatter_hidden_states` | `FULL + TP_ATTN_FULL -> TP_ATTN_FULL` | 把 MLP/MoE full 输出切回下一层 attention 所需 token 范围。可使用 reduce-scatter。 |
| `_gather` | `SCATTERED + SCATTERED -> TP_ATTN_FULL` | 先把 `hidden_states += residual`，再 attention TP all-gather。 |
| `_scatter` | `TP_ATTN_FULL -> SCATTERED` | 按 attention TP rank 切分 token。 |
| `_scatter_hidden_states_moe` | `MOE_FULL -> TP_ATTN_FULL` | MoE CP all-gather 后切回本 CP rank 的真实 token 段。 |

对普通 GLM TP=4 基线，MLP/MoE 自身通常已经完成必要的 TP 汇总，`layer_output_mode` 仍是 `TP_ATTN_FULL`，因此 `postprocess_layer()` 经常退化为 no-op。不要因为这里 no-op 就认为 `LayerCommunicator` 没做通信：attention 后的关键 all-reduce 已经在 `prepare_mlp()` 中完成。

### 9.1 `should_use_reduce_scatter()`

`LayerCommunicator.should_use_reduce_scatter()` 会告诉 MLP/MoE：

```text
本层 MLP/MoE 结束后是否可以直接 reduce-scatter，
而不是先 all-reduce 成完整 hidden 再由 postprocess_layer 切分。
```

它会在以下情况返回 `True`：

- 当前模型允许 `allow_reduce_scatter`；
- `postprocess_layer()` 选中的函数支持把 full 输出切回局部 token；
- DP padding 模式适合 reduce-scatter；
- DSA/MLA prefill CP 正在使用；
- `attn_tp_input_scattered` 生效且当前不是最后一层。

这个标志会传进模型的 `self.mlp(...)`。对 MoE 来说，它会影响 routed/shared partial 输出最后怎样跨 rank 汇总。

### 9.2 `should_fuse_mlp_allreduce_with_next_layer()`

这个函数用于把 MLP/MoE 后的 all-reduce 延迟到下一层 `prepare_attn()`，典型方式是在 tensor 上打标：

```python
hidden_states._sglang_needs_allreduce_fusion = True
```

下一层 `prepare_attn()` 看到这个标志后，会先完成 `moe_tensor_model_parallel_all_reduce(hidden_states)`，再做 input RMSNorm。

在 Ascend NPU BF16 常见基线中，这条融合路径通常不是主要路径，因为源码里的相关融合判断更多面向 FlashInfer/CUDA 或 Aiter/ROCm。但理解这个标志很重要：如果 profiler 中某一层 MLP 后面没有立刻看到 all-reduce，可能不是少通信，而是通信被推迟到下一层边界。

## 10. HCCL 调用链

### 10.1 分布式 backend 如何变成 HCCL

NPU 的默认分布式 backend 在 `parallel_state.py` 中定义：

```python
_DEVICE_TO_DISTRIBUTED_BACKEND = {
    ...
    "npu": "hccl" if not envs.SGLANG_ZBAL_LOCAL_MEM_SIZE.get() > 0 else "zbal",
}
```

`ModelRunner.init_torch_distributed()` 会调用：

```text
backend = get_default_distributed_backend(self.device)
init_distributed_environment(backend=backend, ...)
initialize_model_parallel(...)
initialize_dp_attention(...)
```

对普通 NPU 推理来说，`backend="hccl"`。随后 `init_distributed_environment()` 调用：

```python
torch.distributed.init_process_group(
    backend=backend,
    init_method=distributed_init_method,
    world_size=world_size,
    rank=rank,
    timeout=timeout,
    pg_options=pg_options,
)
```

NPU 上 `pg_options` 来自：

```python
torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
```

并可设置：

```python
options.hccl_config = {"hccl_buffer_size": hccl_buffer_size}
```

后续 TP、attention TP、attention CP、MoE TP、MoE EP 等子 group 都由 `initialize_model_parallel()` 创建。

### 10.2 `GroupCoordinator` 怎样选择 NPU communicator

`init_model_parallel_group()` 创建 `GroupCoordinator` 时会传入：

```python
use_npu_communicator=True
```

`GroupCoordinator.__init__()` 中，如果是 NPU 且 `world_size > 1`，就创建：

```python
self.npu_communicator = NpuCommunicator(group=self.device_group)
```

当上层调用：

```python
get_attn_tp_group().all_reduce(hidden_states)
```

会进入：

```text
GroupCoordinator.all_reduce
  -> 如果 npu_communicator 可用
  -> NpuCommunicator.all_reduce
  -> dist.all_reduce(x, group=self.group)
```

这里的 `dist.all_reduce` 在 HCCL process group 上执行，所以底层是 HCCL AllReduce。

### 10.3 常见 collective 到 HCCL 的映射

| 上层入口 | 中间 wrapper | NPU 执行 |
|---|---|---|
| `attention_tensor_model_parallel_all_reduce(x)` | `get_attn_tp_group().all_reduce(x)` | `NpuCommunicator.all_reduce -> dist.all_reduce -> HCCL AllReduce` |
| `moe_tensor_model_parallel_all_reduce(x)` | `get_moe_tp_group().all_reduce(x)` | `NpuCommunicator.all_reduce -> dist.all_reduce -> HCCL AllReduce` |
| `tensor_model_parallel_all_reduce(x)` | `get_tp_group().all_reduce(x)` | `NpuCommunicator.all_reduce -> dist.all_reduce -> HCCL AllReduce` |
| `attn_tp_all_gather_into_tensor(out, x)` | `get_attention_tp_group().all_gather_into_tensor(out, x)` | `GroupCoordinator._all_gather_into_tensor -> dist.all_gather_into_tensor -> HCCL AllGather` |
| `attn_tp_reduce_scatter_tensor(out, x)` | `get_attention_tp_group().reduce_scatter_tensor(out, x)` | `GroupCoordinator._reduce_scatter_tensor -> dist.reduce_scatter_tensor -> HCCL ReduceScatter` |
| `attention_tensor_model_parallel_quant_all_reduce(x)` | `get_attn_tp_group().quant_all_reduce(x)` | `torch_npu.npu_dynamic_quant + HCCL AllGather + NPU dequant/sum` |

注意：`LayerCommunicator` 不直接调用 `torch.distributed`，而是通过这些 wrapper。这样做的好处是同一段模型代码可以在 CUDA、ROCm、XPU、HPU、NPU 上复用不同 communicator。

## 11. NPU 计算算子绑定

`LayerCommunicator` 本身不做 attention matmul，也不做 MoE grouped matmul，但它会触发 norm、quant、prefetch 和一些 tensor 操作。

| 触发位置 | Python 调用 | NPU 绑定 | 数学或系统语义 |
|---|---|---|---|
| `prepare_attn()` 中 `residual is None` | `input_layernorm(hidden_states)` | `torch_npu.npu_rms_norm` | 对每个 token 的 `[D]` hidden 做 RMSNorm。 |
| `prepare_attn()` 中 `residual is not None` | `input_layernorm(hidden_states, residual)` | `torch_npu.npu_add_rms_norm` | 先 residual add，再 RMSNorm，并返回新的 residual。 |
| `prepare_mlp()` 后 attention residual | `post_attention_layernorm(hidden_states, residual)` | `torch_npu.npu_add_rms_norm` | 把 attention 增量加回 residual，再生成 MLP/MoE 输入。 |
| 量化通信 | `NpuCommunicator.quant_all_reduce` | `torch_npu.npu_dynamic_quant` | 通信前按 token 动态量化到 int8。 |
| CMO prefetch | `prepare_weight_cache` | `torch_npu.npu_prefetch` | 在独立 NPU stream 上预取后续 matmul 权重。 |
| pair 合并 | `hidden_states += residual`、`tensor_split`、`reshape` | ATen NPU tensor kernels | 本卡 tensor 操作，不是 HCCL collective。 |

因此 profiler 中分析 `LayerCommunicator` 时，至少要同时关注：

```text
HCCL AllReduce / AllGather / ReduceScatter
torch_npu.npu_rms_norm
torch_npu.npu_add_rms_norm
torch_npu.npu_dynamic_quant
torch_npu.npu_prefetch
普通 ATen NPU add/slice/reshape/copy kernel
```

## 12. 以 GLM-4.7-Flash TP=4 为例

以常见启动为例：

```bash
sglang serve \
  --model-path /home/{myspace}/models/GLM-4.7-Flash \
  --device npu \
  --tp-size 4 \
  --attention-backend ascend \
  --sampling-backend ascend \
  --disable-cuda-graph
```

假设不启用 DP attention、CP、DeepEP/FuseEP、量化通信，Layer 0 的主链可以读成：

```text
embedding output [N,2048]
  -> prepare_attn
       residual = hidden_states
       npu_rms_norm -> attention input [N,2048]
  -> MLA attention
       local attention partial [N,2048]
  -> prepare_mlp
       HCCL AllReduce across attention TP group
       npu_add_rms_norm -> dense MLP input [N,2048]
  -> dense MLP
       local/projected compute + necessary TP collective in MLP implementation
  -> postprocess_layer
       usually no-op in this baseline
  -> pair for Layer 1
```

Layer 1 到 Layer 46 的主链类似，只是 MLP 换成 sparse MoE：

```text
上一层 pair
  -> prepare_attn
       npu_add_rms_norm -> attention input [N,2048]
  -> MLA attention
       local attention partial [N,2048]
  -> prepare_mlp
       HCCL AllReduce
       npu_add_rms_norm -> MoE input [N,2048]
  -> sparse MoE
       router/top-k/expert grouped matmul/shared expert
       necessary TP collective or reduce-scatter in MoE implementation
  -> postprocess_layer
       usually no-op in this baseline
  -> pair for next layer
```

`LayerCommunicator` 在这条基线中最重要的动作是两类：

1. 每层 attention 前和 MLP/MoE 前的 RMSNorm，其中 residual 场景绑定到 `npu_add_rms_norm`。
2. attention 输出后的 TP all-reduce，其中跨卡通信绑定到 HCCL AllReduce。

## 13. 打开复杂并行后会发生什么

### 13.1 `enable_attn_tp_input_scattered`

满足条件时，`AttnTpContext.use_input_scattered()` 会在 extend/prefill 中让 attention 输入保持 scattered，一直到 Q/KV latent 阶段再 all-gather。

直观效果：

```text
默认:
  gather hidden [N,2048]
  -> compute q/kv latent

input_scattered:
  compute local q/kv latent [N_local,1344]
  -> gather latent [N,1344]
```

这可以降低通信宽度，但要求 attention prepare 使用 `get_attn_tp_context().fetch_qkv_latent()` 或 `fetch_hidden_states()` 获取正确数据。

### 13.2 DP attention

DP attention 会让 token 维在 DP ranks 之间分布。`LayerCommunicator` 必须在 attention、MLP/MoE 和 residual 之间插入：

```text
dp_gather_partial
dp_scatter
dp_reduce_scatter_tensor
```

这类路径中，`hidden_states.shape[0]` 可能包含 padding，也可能与真实 token 数不同。真实 token 范围由 `ForwardBatch` 和 DP metadata 决定。

### 13.3 MoE CP

当 `attn_cp_size > moe_dp_size` 时，MoE 前可能需要 `MOE_FULL`：

```text
attention CP local tokens
  -> moe_cp_all_gather_into_tensor
  -> MoE group sees full tokens
  -> MoE compute
  -> _scatter_hidden_states_moe
  -> slice back to local CP rank tokens
```

这解释了为什么 `postprocess_layer()` 中存在 `_scatter_hidden_states_moe()`：它不是普通 TP scatter，而是为 MoE CP all-gather 后恢复 token 范围。

### 13.4 DeepEP / FuseEP

如果启用 DeepEP、Mooncake 或 Ascend FuseEP 等 A2A MoE backend，`LayerScatterModes._compute_mlp_mode()` 会倾向于让 sparse layer 的 `mlp_mode=SCATTERED`。

含义是：

```text
LayerCommunicator 不再把所有 token gather 成普通 TP MoE 输入；
MoE backend 自己负责 token dispatch/combine；
LayerCommunicator 只处理进入和离开 MoE 组件的边界。
```

所以定位 DeepEP 问题时，不要只看 `LayerCommunicator`。它决定边界模式，但 expert token 的跨设备 dispatch/combine 在 DeepEP/FuseEP 组件中。

## 14. TBO/SBO 拆分执行中的 `op_comm_*`

GLM-4.7-Flash 的 `Glm4MoeLiteDecoderLayer` 除了普通 `forward()`，还定义了：

```python
op_comm_prepare_attn()
op_comm_prepare_mlp()
op_comm_postprocess_layer()
```

这三个函数不是另一套数学逻辑，而是把普通 forward 拆成可调度的小块：

```text
普通 forward:
  prepare_attn -> attention -> prepare_mlp -> mlp -> postprocess_layer

拆分 op:
  op_comm_prepare_attn
  attention op
  op_comm_prepare_mlp
  mlp/moe op
  op_comm_postprocess_layer
```

TBO/SBO 可以在不同 layer、不同 sub-batch 之间交错执行这些 op，但每个通信 op 仍然调用同一个 `LayerCommunicator` 方法。因此 profiling 中看到的 collective 和 norm 仍可按本讲的三段边界理解。

## 15. Profiling 与定位方法

### 15.1 先确认当前 layer 的模式

调试时可以先打印或断点观察：

```python
layer.layer_scatter_modes
layer.layer_communicator._context
layer.layer_communicator._communicate_simple_fn
layer.layer_communicator._communicate_with_all_reduce_and_layer_norm_fn
layer.layer_communicator._communicate_summable_tensor_pair_fn
```

重点看：

- `mlp_mode` 是 `FULL`、`SCATTERED` 还是 `MOE_FULL`；
- `attn_tp_size`、`attn_dp_size`、`attn_cp_size` 是否符合启动参数；
- 三个 `_communicate_*_fn` 是否选中了预期函数；
- `should_use_reduce_scatter()` 是否和 MLP/MoE 的通信策略一致；
- 是否有 `_sglang_needs_allreduce_fusion` 标志把通信推迟到下一层。

### 15.2 在 profiler 中识别它

如果当前是普通 GLM TP=4 基线，每层常见的 `LayerCommunicator` 相关事件包括：

```text
prepare_attn:
  npu_rms_norm 或 npu_add_rms_norm

prepare_mlp:
  HCCL AllReduce
  npu_add_rms_norm

postprocess_layer:
  多数层可能没有明显 collective
```

如果启用复杂并行，还可能看到：

```text
HCCL AllGather
HCCL ReduceScatter
npu_dynamic_quant
HCCL AllGather for quant payload
npu_prefetch
ATen NPU copy/slice/add
```

### 15.3 常见问题定位顺序

| 现象 | 优先检查 |
|---|---|
| 多卡精度和单卡差异明显 | attention partial 是否在 `prepare_mlp()` 里正确 all-reduce；MLP/MoE 是否根据 `use_reduce_scatter` 正确汇总；quant communication 是否启用。 |
| 某层后 shape 不一致 | `layer_scatter_modes` 五个字段，尤其是 `mlp_mode`、`middle_residual_mode`、`layer_output_mode`。 |
| profiler 中 all-reduce 少于预期 | 是否走了 allreduce fusion marker；是否 reduce-scatter 替代 all-reduce；是否 DeepEP/FuseEP 接管 MoE dispatch/combine。 |
| `npu_add_rms_norm` 输出异常 | 同时检查返回的 `out` 和 `residual_out`，不要只看 normed tensor。 |
| CP/MoE CP 下 token 数错乱 | 检查 `attn_cp_metadata.per_rank_actual_token`、padding 后 max tokens、`_scatter_hidden_states_moe()` 的 narrow 范围。 |
| 首 token 延迟高 | HCCL 初始化、首次 all-reduce、CMO stream 创建、NPU Graph 是否关闭。 |

## 16. 常见误解

### 16.1 `LayerCommunicator` 不是 attention backend

它不会计算 Q/K/V，不会调用 FIA，也不会执行 paged attention。它只负责 attention 前后的边界。真正 attention 计算仍在 `DeepseekV2AttentionMLA`、`RadixAttention`、`AscendAttnBackend` 等对象里完成。

### 16.2 `LayerCommunicator` 不是 `NpuCommunicator`

`LayerCommunicator` 知道 layer 语义；`NpuCommunicator` 只知道如何在 NPU process group 上做 collective。前者调用后者，中间隔着 `communication_op.py` 和 `GroupCoordinator`。

### 16.3 `disable_custom_all_reduce=True` 不表示 NPU 不通信

在 Ascend NPU 上，关闭 CUDA custom all-reduce 不等于关闭 HCCL。`GroupCoordinator` 仍会通过 `NpuCommunicator` 或 `torch.distributed` 执行 HCCL collective。

### 16.4 `FULL` 不等于“hidden size 没切”

`FULL` 描述 token 覆盖范围和 group 语义，不是说权重或 hidden feature 没有 TP 切分。GLM TP=4 时，许多 linear 权重仍是 TP 分片的，但 layer 边界上的 hidden tensor 可以在每个 rank 上持有完整 `[N,2048]` 语义。

### 16.5 `postprocess_layer()` no-op 不代表没有通信

普通 TP 基线中最关键的 attention all-reduce 发生在 `prepare_mlp()`。`postprocess_layer()` 只是层尾收口，很多情况下确实不需要额外通信。

## 17. 阅读检查题

1. 为什么 GLM-4.7-Flash 的 attention 设置 `reduce_results=False` 后，`prepare_mlp()` 必须承担 attention TP all-reduce？
2. `SCATTERED`、`TP_ATTN_FULL`、`FULL`、`MOE_FULL` 描述的是 token 分布还是 hidden feature 分片？
3. `npu_add_rms_norm` 同时返回哪两个对后续 layer 有意义的结果？
4. `LayerCommunicator.prepare_mlp()` 中的 HCCL AllReduce 和 `RMSNorm.forward_npu()` 中的 `torch_npu.npu_add_rms_norm` 分别属于通信还是计算？
5. 为什么 `enable_attn_tp_input_scattered` 可以把通信宽度从 `[N,2048]` 降到 MLA latent 的 `[N,1344]`？
6. DeepEP/FuseEP 打开后，为什么不能只在 `LayerCommunicator` 里找 expert dispatch/combine？
7. 如果 profiler 中某层 MLP 后没有立即出现 all-reduce，应该检查哪些延迟通信或替代通信分支？

