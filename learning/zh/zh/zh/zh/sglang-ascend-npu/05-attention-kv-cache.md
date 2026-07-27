# 05. Ascend Attention、KV Cache 与 HiCache

这一讲拆 SGLang-NPU 最核心的性能路径：attention backend 如何读写 KV cache，以及 NPU 专用 KV pool 为什么要和 `attention_backend=ascend` 配套使用。

## 主链路

```mermaid
flowchart LR
  A["ScheduleBatch"] --> B["ForwardBatch"]
  B --> C["ModelRunner.forward"]
  C --> D["model.forward"]
  D --> E["RadixAttention / attention layer"]
  E --> F["AscendAttnBackend"]
  F --> G["ForwardMetadata"]
  F --> H["NPUMHATokenToKVPool / NPUMLATokenToKVPool"]
  F --> I["torch_npu / torch.ops.npu"]
```

## 关键源码

| 主题 | 文件 |
|---|---|
| attention 注册 | `python/sglang/srt/layers/attention/attention_registry.py` |
| Ascend attention 主实现 | `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py` |
| MLA 预处理 | `python/sglang/srt/hardware_backend/npu/attention/mla_preprocess.py` |
| NPU KV pool | `python/sglang/srt/hardware_backend/npu/memory_pool_npu.py` |
| NPU allocator | `python/sglang/srt/hardware_backend/npu/allocator_npu.py` |
| KV pool 选择 | `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` |

## Attention backend 初始化

```mermaid
flowchart TD
  A["ModelRunner.init_attention_backend"] --> B["_get_attention_backend"]
  B --> C["server_args.get_attention_backends"]
  C --> D["_get_attention_backend_from_str('ascend')"]
  D --> E["ATTENTION_BACKENDS['ascend']"]
  E --> F["create_ascend_backend"]
  F --> G["AscendAttnBackend"]
  G --> H["attn_backend_wrapper"]
  H --> I{"hybrid/mambaish?"}
  I -->|"否"| J["直接使用 AscendAttnBackend"]
  I -->|"是"| K["AscendHybridLinearAttnBackend / AscendGDNAttnBackend"]
```

## `ForwardMetadata`

`ForwardBatch` 是 SGLang 通用的模型执行输入；`ForwardMetadata` 是 Ascend attention backend 的设备侧执行信息。

它承载的信息包括：

- `seq_lens`
- `extend_seq_lens`
- `positions`
- `slot_mapping`
- `block_tables`
- attention mask
- graph replay 所需的固定 shape metadata

理解方式：

```mermaid
flowchart LR
  A["ForwardBatch<br/>通用 batch 表示"] --> B["AscendAttnBackend.init_forward_metadata"]
  B --> C["ForwardMetadata<br/>Ascend kernel metadata"]
  C --> D["forward_extend"]
  C --> E["forward_decode"]
  C --> F["forward_decode_graph"]
```

## Prefill / Extend 路径

Prefill 处理 prompt token，写入 KV cache。

```mermaid
flowchart TD
  A["input_ids / positions"] --> B["ForwardBatch"]
  B --> C["init_forward_metadata"]
  C --> D["forward_extend"]
  D --> E{"模型类型"}
  E -->|"MHA"| F["MHA attention kernel"]
  E -->|"MLA"| G["NPUFusedMLAPreprocess + MLA kernel"]
  E -->|"fallback"| H["AscendTorchNativeAttnBackend"]
  F --> I["写 KV cache"]
  G --> I
  H --> I
```

重点：

- 长 prompt 会受 `chunked_prefill_size` 影响。
- MHA 和 MLA 的 KV layout 不同。
- MLA 可能经过 `NPUFusedMLAPreprocess` 做 RMSNorm、RoPE、KV cache 写入融合。

## Decode 路径

Decode 每轮通常为每个活跃请求生成 1 个 token。

```mermaid
flowchart TD
  A["running batch"] --> B["prepare_for_decode"]
  B --> C["alloc_decode: 每请求分配新 KV slot"]
  C --> D["ForwardBatch"]
  D --> E{"can replay graph?"}
  E -->|"是"| F["forward_decode_graph"]
  E -->|"否"| G["forward_decode"]
  F --> H["读取历史 KV + 写新 KV"]
  G --> H
  H --> I["logits -> sampler"]
```

Decode 的性能非常依赖：

- KV cache 读取效率。
- graph replay 是否命中。
- batch size 是否在 capture 范围内。
- attention backend 是否走 Ascend kernel，而不是 fallback。

## NPU KV Pool 类型

```mermaid
classDiagram
  class NPUMHATokenToKVPool {
    MHA/GQA KV cache
    +set_kv_buffer()
    +get_cpu_copy()
    +load_cpu_copy()
  }

  class NPUMLATokenToKVPool {
    MLA latent KV cache
    +get_key_buffer()
    +get_value_buffer()
    +get_index_k_buffer()
    +set_kv_buffer()
    +set_index_k_buffer()
  }

  class NPUPagedTokenToKVPoolAllocator {
    +alloc_extend()
    +alloc_decode()
    +free()
  }

  NPUPagedTokenToKVPoolAllocator --> NPUMHATokenToKVPool
  NPUPagedTokenToKVPoolAllocator --> NPUMLATokenToKVPool
```

选择逻辑：

| 模型类型 | KV pool |
|---|---|
| 普通 MHA/GQA | `NPUMHATokenToKVPool` |
| MLA | `NPUMLATokenToKVPool` |
| hybrid SWA | `SWAKVPool(token_to_kv_pool_class=NPUMHATokenToKVPool)` |

## Allocator

`NPUPagedTokenToKVPoolAllocator` 负责从 KV pool 中分配 slot。

| 方法 | 作用 |
|---|---|
| `alloc_extend` | prefill/extend 时为多个 token 分配 KV slot。 |
| `alloc_decode` | decode 时通常为每个请求分配 1 个新 token slot。 |
| `free` | 请求结束后释放 KV slot。 |

调度层看到的是 token 位置和 request index；attention kernel 需要的是可以访问 KV cache 的具体 slot metadata。allocator 就是中间桥梁。

## HiCache

HiCache 是分层 KV cache，可以把部分 KV 从 NPU device memory 扩展到 host 或 storage。

NPU 默认配置：

```text
hicache_io_backend = kernel_ascend
hicache_mem_layout = page_first_kv_split   # MLA
hicache_mem_layout = page_first_direct     # MHA
```

```mermaid
flowchart TB
  A["NPU KV cache"] --> B{"显存压力 / cache policy"}
  B --> C["保留热 KV"]
  B --> D["evict 到 host/storage"]
  D --> E["HiCache kernel_ascend"]
  E --> F["load back 到 NPU"]
  F --> A
```

初学建议：

- 先不要打开 HiCache。
- 先确认普通 KV cache 和 attention 路径稳定。
- 再用长上下文和 prefix cache 场景测试 HiCache。

## 常见错误直觉

| 现象 | 可能方向 |
|---|---|
| 服务能启动，first token 很慢 | graph 未命中、prefill 过大、attention fallback。 |
| 长 prompt OOM | `chunked_prefill_size`、KV pool size、`mem_fraction_static`。 |
| decode 抖动大 | graph capture size 不覆盖、batch shape 变化大。 |
| MLA 模型异常 | MLA KV layout、`NPUFusedMLAPreprocess`、模型配置。 |
| HiCache 打开后变慢 | IO backend、host 内存、load back 频繁。 |

## 阅读任务

1. 在 `attention_registry.py` 中找到 `"ascend"` 注册点。
2. 在 `ascend_backend.py` 中找到 `init_forward_metadata()`、`forward_extend()`、`forward_decode()`。
3. 在 `model_runner_kv_cache_mixin.py` 中找到 NPU KV pool 选择分支。
4. 在 `memory_pool_npu.py` 中比较 `NPUMHATokenToKVPool` 和 `NPUMLATokenToKVPool`。
