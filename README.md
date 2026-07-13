# SGLang Tutor

这个仓库用于学习和讲解 [SGLang](https://github.com/sgl-project/sglang) 的实现原理。仓库中保留了教学所需的 SGLang 源码快照与讲义材料，重点不是维护 SGLang 本身，而是围绕请求链路、调度、模型执行、KV Cache、Attention、PD 分离、Router、Ascend NPU 和 AI Infra 基础机制做中文源码阅读。

## 仓库内容

| 目录 | 内容 |
|---|---|
| [`learning/`](./learning/) | 本仓库核心教学材料，包含源码阅读、AI Infra 基础、Scheduler、ModelRunner、Ascend NPU 实践等专题。 |
| [`python/`](./python/) | 从 SGLang 原仓库保留的 Python runtime 源码，供教学文档引用和源码阅读使用。 |
| [`sgl-kernel/`](./sgl-kernel/) | SGLang runtime 会引用的 kernel 代码与 Python 包，例如 `sgl_kernel` 和 `torch.ops.sgl_kernel.*`。 |
| [`rust/sglang-grpc/`](./rust/sglang-grpc/) | `python/pyproject.toml` 中声明的 Rust gRPC 扩展源码。 |
| [`experimental/sgl-router/`](./experimental/sgl-router/) | 从 SGLang 上游补充的 Rust router 源码，用于讲解 KV-aware、OpenAI-compatible worker 路由器。 |
| [`proto/`](./proto/) | `rust/sglang-grpc` 构建时使用的 protobuf 定义。 |
| [`LICENSE`](./LICENSE) | 保留 SGLang 上游项目的 Apache-2.0 许可证信息。 |

当前仓库已经去掉了大量与教学无关的上游工程内容，例如完整 CI、Docker、benchmark、测试集、文档站和第三方依赖目录。需要运行、部署或参与 SGLang 开发时，请以 [SGLang 官方仓库](https://github.com/sgl-project/sglang) 为准。

## Learning 总览

[`learning/`](./learning/) 是这个仓库的主入口，建议按下面顺序阅读：

1. [`learning/ai-infra-basic/`](./learning/ai-infra-basic/)：先补 LLM serving 基础概念。
2. [`learning/sglang-source-reading/`](./learning/sglang-source-reading/)：再沿 SGLang 主链路读源码。
3. [`learning/scheduler-architecture/`](./learning/scheduler-architecture/)：深入 Scheduler 架构和中文注释源码。
4. [`learning/tp-worker-model-runner/`](./learning/tp-worker-model-runner/)：深入 TpModelWorker 与 ModelRunner 执行层。
5. [`learning/sglang-ascend-npu/`](./learning/sglang-ascend-npu/)：最后进入 Ascend NPU 部署、适配和优化专题。

## 1. SGLang 源码阅读主线

目录：[`learning/sglang-source-reading/`](./learning/sglang-source-reading/)

这个专题按 SGLang online serving 的真实执行链路展开，并已经按源码层次归档：全局组件、入口/路由、调度运行时、KV/cache、模型执行、layer/通信和高级特性。适合在读 `python/sglang/srt/` 源码时作为总导航。

| 层级 | 文件 | 摘要 |
|---|---|---|
| 入口 | [`README.md`](./learning/sglang-source-reading/README.md) | 源码阅读路线、使用方法、CodeGraph 校准说明。 |
| 全局总览 | [`00-overview/01-public-components-code-walkthrough.md`](./learning/sglang-source-reading/00-overview/01-public-components-code-walkthrough.md) | SGLang 高频公共组件、调用关系、关键函数定位和端到端 code walkthrough。 |
| 特性地图 | [`00-overview/00-feature-map.md`](./learning/sglang-source-reading/00-overview/00-feature-map.md) | SGLang 特性地图，解释 dLLM、HiCache、PD disaggregation、Speculative Decoding、LoRA 等概念。 |
| 入口/路由 | [`01-entry-routing/01-request-lifecycle.md`](./learning/sglang-source-reading/01-entry-routing/01-request-lifecycle.md) | 一次 `/v1/chat/completions` 请求从 HTTP 入口、tokenize、Scheduler、GPU forward 到返回响应的完整生命周期。 |
| 入口/路由 | [`01-entry-routing/09-router.md`](./learning/sglang-source-reading/01-entry-routing/09-router.md) | SGLang 中多层 router 的含义，包括 SmartRouter、PD bootstrap route、MoE expert router 和 Rust `sgl-router`。 |
| 入口/路由 | [`01-entry-routing/10-sgl-router-source-deep-dive.md`](./learning/sglang-source-reading/01-entry-routing/10-sgl-router-source-deep-dive.md) | Rust `sgl-router` 的 discovery、worker registry、KV-aware routing、PD 调度、Proxy/SSE 通信边界。 |
| 调度运行时 | [`02-scheduler-runtime/02-scheduler-core.md`](./learning/sglang-source-reading/02-scheduler-runtime/02-scheduler-core.md) | Scheduler 如何排队、组 prefill/decode batch，并支撑 continuous batching。 |
| 调度运行时 | [`02-scheduler-runtime/06-multiprocess-distributed.md`](./learning/sglang-source-reading/02-scheduler-runtime/06-multiprocess-distributed.md) | Engine 如何拉起 Tokenizer/Scheduler/Detokenizer，TP/PP/DP/DP attention 如何组织 rank 与通信。 |
| 缓存/内存 | [`03-cache-memory/03-kv-cache-radix-cache.md`](./learning/sglang-source-reading/03-cache-memory/03-kv-cache-radix-cache.md) | KV cache memory pool、Radix prefix cache、HiCache 与 Scheduler 的协作方式。 |
| 模型执行 | [`04-model-execution/04-model-runner-attention.md`](./learning/sglang-source-reading/04-model-execution/04-model-runner-attention.md) | `ForwardBatch`、`ModelRunner`、`RadixAttention` 与 attention backend 如何读写 KV cache。 |
| Layer/通信 | [`05-layer-communication/01-layer-communicator-and-common-layers.md`](./learning/sglang-source-reading/05-layer-communication/01-layer-communicator-and-common-layers.md) | DecoderLayer、LayerCommunicator、TP/EP/CP 通信、attention backend、linear/MoE kernel 的配合。 |
| 高级特性 | [`06-advanced-features/05-speculative-decoding.md`](./learning/sglang-source-reading/06-advanced-features/05-speculative-decoding.md) | Draft/target verify、`spec_info`、EAGLE/NGRAM、spec v1/v2 与接受 token 后处理。 |
| 高级特性 | [`06-advanced-features/07-disaggregation-pd.md`](./learning/sglang-source-reading/06-advanced-features/07-disaggregation-pd.md) | Prefill/Decode 分离部署、bootstrap/prealloc/transfer 队列、KV sender/receiver 与 transfer backend。 |
| 高级特性 | [`06-advanced-features/08-lora-serving.md`](./learning/sglang-source-reading/06-advanced-features/08-lora-serving.md) | LoRA adapter 注册、热加载/卸载、Scheduler 混批约束、LoRA memory pool 和 LoRA kernel。 |

## 2. AI Infra 基础专题

目录：[`learning/ai-infra-basic/`](./learning/ai-infra-basic/)

这个专题用于补齐阅读 SGLang 源码前后的基础知识。它不追求生产级完整实现，而是用讲义和小型 Python demo 拆开 LLM serving 常见机制。

| 顺序 | 专题 | 摘要 | 内容链接 |
|---|---|---|---|
| 1 | [`Inference_Basics`](./learning/ai-infra-basic/Inference_Basics/) | Transformer 推理、prefill/decode、batch inference、TTFT/ITL/TPS 等基础概念。 | [`README.md`](./learning/ai-infra-basic/Inference_Basics/README.md) |
| 2 | [`Schedule_Optimization`](./learning/ai-infra-basic/Schedule_Optimization/) | Prefill/decode 调度、continuous batching、chunked prefill、吞吐与延迟权衡。 | [`README.md`](./learning/ai-infra-basic/Schedule_Optimization/README.md), [`prefill_decode_demo.py`](./learning/ai-infra-basic/Schedule_Optimization/prefill_decode_demo.py), [`chunked_prefill_with_fakeLLM_tutorial.py`](./learning/ai-infra-basic/Schedule_Optimization/chunked_prefill_with_fakeLLM_tutorial.py) |
| 3 | [`KV_Cache_Memory`](./learning/ai-infra-basic/KV_Cache_Memory/) | KV Cache 布局、分页、prefix cache、显存估算和 cache eviction 的基础模型。 | [`README.md`](./learning/ai-infra-basic/KV_Cache_Memory/README.md) |
| 4 | [`Attention_Kernel`](./learning/ai-infra-basic/Attention_Kernel/) | FlashAttention、FlashDecoding、attention shape、KV 读写和 memory-bound/kernel 优化直觉。 | [`README.md`](./learning/ai-infra-basic/Attention_Kernel/README.md), [`flash_attention_tutorial.py`](./learning/ai-infra-basic/Attention_Kernel/flash_attention_tutorial.py), [`flash_decoding_tutorial.py`](./learning/ai-infra-basic/Attention_Kernel/flash_decoding_tutorial.py) |
| 5 | [`Execution_Graph`](./learning/ai-infra-basic/Execution_Graph/) | 计算图、CUDA/NPU Graph、torch.compile、静态 shape、capture/replay 和 decode replay 数据流。 | [`README.md`](./learning/ai-infra-basic/Execution_Graph/README.md), [`01-what-is-graph.md`](./learning/ai-infra-basic/Execution_Graph/01-what-is-graph.md), [`02-graph-execution-dataflow.md`](./learning/ai-infra-basic/Execution_Graph/02-graph-execution-dataflow.md) |
| 6 | [`Mamba_State_Space`](./learning/ai-infra-basic/Mamba_State_Space/) | Mamba/SSM 原理、Mamba state、MambaPool、scheduler strategy 与 radix cache 状态复用。 | [`README.md`](./learning/ai-infra-basic/Mamba_State_Space/README.md), [`01-mamba-and-sglang-state.md`](./learning/ai-infra-basic/Mamba_State_Space/01-mamba-and-sglang-state.md), [`02-mamba-principles.md`](./learning/ai-infra-basic/Mamba_State_Space/02-mamba-principles.md), [`03-mamba-radix-cache.md`](./learning/ai-infra-basic/Mamba_State_Space/03-mamba-radix-cache.md) |
| 7 | [`Parallel_Strategy`](./learning/ai-infra-basic/Parallel_Strategy/) | DP、TP、PP、SP/CP、EP 推理并行策略和通信模式。 | [`README.md`](./learning/ai-infra-basic/Parallel_Strategy/README.md), [`tutorial.md`](./learning/ai-infra-basic/Parallel_Strategy/tutorial.md), [`dp_inference_demo.py`](./learning/ai-infra-basic/Parallel_Strategy/dp_inference_demo.py), [`tp_inference_demo.py`](./learning/ai-infra-basic/Parallel_Strategy/tp_inference_demo.py), [`pp_inference_demo.py`](./learning/ai-infra-basic/Parallel_Strategy/pp_inference_demo.py), [`sp_inference_demo.py`](./learning/ai-infra-basic/Parallel_Strategy/sp_inference_demo.py), [`ep_moe_demo.py`](./learning/ai-infra-basic/Parallel_Strategy/ep_moe_demo.py) |
| 8 | [`KV_Transfer`](./learning/ai-infra-basic/KV_Transfer/) | PD 分离、KV sender/receiver、远程 KV cache、Mooncake/NIXL/Mori/Ascend transfer backend 的基础抽象。 | [`README.md`](./learning/ai-infra-basic/KV_Transfer/README.md) |
| 9 | [`Speculative_Decoding`](./learning/ai-infra-basic/Speculative_Decoding/) | Draft/target、verify、acceptance rate、EAGLE/NGRAM 与投机解码是否划算的判断方法。 | [`README.md`](./learning/ai-infra-basic/Speculative_Decoding/README.md) |
| 10 | [`Quantization`](./learning/ai-infra-basic/Quantization/) | FP16/BF16/FP8/INT8/INT4、weight-only、KV quant、AWQ/GPTQ/SmoothQuant 基础。 | [`README.md`](./learning/ai-infra-basic/Quantization/README.md) |
| 11 | [`LoRA`](./learning/ai-infra-basic/LoRA/) | LoRA、QLoRA、DoRA、AdaLoRA 的简化训练与模块替换 demo，以及多 LoRA serving 直觉。 | [`README.md`](./learning/ai-infra-basic/LoRA/README.md), [`lora_tutorial.py`](./learning/ai-infra-basic/LoRA/lora_tutorial.py), [`qlora_tutorial.py`](./learning/ai-infra-basic/LoRA/qlora_tutorial.py), [`dora_tutorial.py`](./learning/ai-infra-basic/LoRA/dora_tutorial.py), [`adalora_tutorial.py`](./learning/ai-infra-basic/LoRA/adalora_tutorial.py) |
| 12 | [`Benchmark_Profiling`](./learning/ai-infra-basic/Benchmark_Profiling/) | TTFT/ITL/TPS、压测、profiling、瓶颈定位和优化验证闭环。 | [`README.md`](./learning/ai-infra-basic/Benchmark_Profiling/README.md) |

## 3. Scheduler 架构专题

目录：[`learning/scheduler-architecture/`](./learning/scheduler-architecture/)

这个专题比源码阅读主线中的 Scheduler 一讲更细，适合想单独吃透 `python/sglang/srt/managers/scheduler.py` 的读者。

| 文件 | 摘要 |
|---|---|
| [`README.md`](./learning/scheduler-architecture/README.md) | Scheduler 专题入口、阅读顺序和学习主线。 |
| [`01-architecture.md`](./learning/scheduler-architecture/01-architecture.md) | Scheduler 的角色、模块边界、队列、batch 和 worker 调用关系。 |
| [`02-flowcharts.md`](./learning/scheduler-architecture/02-flowcharts.md) | 请求调度、batch 构造、forward、返回结果等流程图。 |
| [`03-annotated-code-walkthrough.md`](./learning/scheduler-architecture/03-annotated-code-walkthrough.md) | 按关键路径串起 Scheduler 源码的中文导读。 |
| [`04-function-map.md`](./learning/scheduler-architecture/04-function-map.md) | Scheduler 关键函数和代码段定位表。 |
| [`05-scheduler-annotated-cn.py`](./learning/scheduler-architecture/05-scheduler-annotated-cn.py) | `scheduler.py` 的中文块级注释版源码副本，只用于学习对照。 |

## 4. TP Worker 与 ModelRunner 专题

目录：[`learning/tp-worker-model-runner/`](./learning/tp-worker-model-runner/)

这个专题聚焦 Scheduler 之后的模型执行层，解释 `TpModelWorker` 如何接住 `ScheduleBatch`，以及 `ModelRunner` 如何管理模型权重、KV cache、attention backend、graph、sampling 和分布式资源。

| 文件 | 摘要 |
|---|---|
| [`README.md`](./learning/tp-worker-model-runner/README.md) | TP Worker 与 ModelRunner 专题入口。 |
| [`01-architecture.md`](./learning/tp-worker-model-runner/01-architecture.md) | `TpModelWorker` 与 `ModelRunner` 在整体 serving 架构中的位置和职责划分。 |
| [`02-flowcharts.md`](./learning/tp-worker-model-runner/02-flowcharts.md) | Scheduler 到 worker、worker 初始化、ModelRunner 初始化、forward、decode/extend、sampling 的流程图。 |
| [`03-function-map.md`](./learning/tp-worker-model-runner/03-function-map.md) | `tp_worker.py` 与 `model_runner.py` 的关键函数定位表。 |
| [`04-tp-worker-annotated-cn.py`](./learning/tp-worker-model-runner/04-tp-worker-annotated-cn.py) | `tp_worker.py` 中文精读注释版源码副本。 |
| [`05-model-runner-annotated-cn.py`](./learning/tp-worker-model-runner/05-model-runner-annotated-cn.py) | `model_runner.py` 中文精读注释版源码副本。 |

## 5. Ascend NPU 实践专题

目录：[`learning/sglang-ascend-npu/`](./learning/sglang-ascend-npu/)

这个专题拆解 SGLang 在 Ascend NPU 上的安装、启动、源码适配、graph、HCCL、attention、KV cache、PD 分离、LoRA/MoE、性能测试和精度定位。适合想在 910 系列 NPU 上部署或二次开发 SGLang 的读者。

| 讲次 | 文件 | 摘要 |
|---|---|---|
| 入口 | [`README.md`](./learning/sglang-ascend-npu/README.md) | Ascend NPU 学习总览、源码地图、架构图和实践流程。 |
| 第 0 讲 | [`00-background.md`](./learning/sglang-ascend-npu/00-background.md) | Ascend NPU 背景知识，梳理 CANN、torch_npu、HCCL、KV cache、attention、graph、PD 等前置概念。 |
| 第 1 讲 | [`01-environment-and-install.md`](./learning/sglang-ascend-npu/01-environment-and-install.md) | GNU/Linux + Ascend NPU 服务器上的环境检查、安装、Docker/源码部署和最小服务跑通。 |
| 第 2 讲 | [`02-ascend-npu-integration-map.md`](./learning/sglang-ascend-npu/02-ascend-npu-integration-map.md) | Ascend NPU 全量源码接入点、初始化流程、调用关系和知识图谱。 |
| 第 3 讲 | [`03-launch-and-minimal-serving.md`](./learning/sglang-ascend-npu/03-launch-and-minimal-serving.md) | 单卡服务启动、请求验证、日志解读和最小 serving 闭环。 |
| 第 4 讲 | [`04-npu-backend-args.md`](./learning/sglang-ascend-npu/04-npu-backend-args.md) | `set_default_server_args()` 逐项讲解，理解 NPU 上 attention backend、page size、graph、all-reduce 等默认值。 |
| 第 5 讲 | [`05-attention-kv-cache.md`](./learning/sglang-ascend-npu/05-attention-kv-cache.md) | Ascend attention、KV cache、HiCache、page/layout/dtype 对执行路径的影响。 |
| 第 6 讲 | [`06-npu-graph-compilation.md`](./learning/sglang-ascend-npu/06-npu-graph-compilation.md) | `NPUGraph`、piecewise graph、warmup、capture、replay、静态 shape 和 graph 命中。 |
| 第 7 讲 | [`07-distributed-hccl-tp.md`](./learning/sglang-ascend-npu/07-distributed-hccl-tp.md) | TP、多卡 rank、HCCL backend、communicator、ZBAL 和 NPU 通信排错。 |
| 第 8 讲 | [`08-ascend-pd-disaggregation.md`](./learning/sglang-ascend-npu/08-ascend-pd-disaggregation.md) | Ascend PD 分离、`AscendTransferEngine`、`sdma`/`device_rdma`、KV transfer 初始化差异。 |
| 第 9 讲 | [`09-lora-moe-feature-branches.md`](./learning/sglang-ascend-npu/09-lora-moe-feature-branches.md) | Ascend LoRA backend、MoE stream、fallback 分支和特性组合风险。 |
| 第 10 讲 | [`10-benchmark-debugging.md`](./learning/sglang-ascend-npu/10-benchmark-debugging.md) | NPU 压测方法、日志定位、性能问题排查顺序。 |
| 第 11 讲 | [`11-performance-optimization-work-map.md`](./learning/sglang-ascend-npu/11-performance-optimization-work-map.md) | 面向 SGLang 与 `sglang-kernel-npu` 开发者的推理优化方向分类。 |
| 第 12 讲 | [`12-npu-profiling-guide.md`](./learning/sglang-ascend-npu/12-npu-profiling-guide.md) | SGLang-NPU profiling 流程、NPU trace 解读、timeline 分析和性能归因模板。 |
| 第 13 讲 | [`13-run-models-by-scenario.md`](./learning/sglang-ascend-npu/13-run-models-by-scenario.md) | 单卡、多卡、PD、在线/离线模型、LoRA、MoE、量化、多模态等场景的启动模板。 |
| 第 14 讲 | [`14-performance-testing.md`](./learning/sglang-ascend-npu/14-performance-testing.md) | 性能测试流程、workload 设计、TTFT/P95/P99/TPS 分析和瓶颈归因。 |
| 第 15 讲 | [`15-accuracy-testing-and-debugging.md`](./learning/sglang-ascend-npu/15-accuracy-testing-and-debugging.md) | 精度测试、`ais_bench` 参考链路、token/logits diff 和按执行流程定位精度问题。 |

## 推荐阅读路线

如果你是第一次读这个仓库，建议按下面节奏走：

1. 先读 [`learning/ai-infra-basic/Inference_Basics/README.md`](./learning/ai-infra-basic/Inference_Basics/README.md)，建立 prefill/decode、KV cache、batching、吞吐和延迟的基础模型。
2. 再读 [`learning/sglang-source-reading/00-overview/01-public-components-code-walkthrough.md`](./learning/sglang-source-reading/00-overview/01-public-components-code-walkthrough.md)，建立公共组件和调用层次。
3. 接着读请求生命周期、Scheduler、KV Cache、ModelRunner 四讲：[`01-request-lifecycle.md`](./learning/sglang-source-reading/01-entry-routing/01-request-lifecycle.md)、[`02-scheduler-core.md`](./learning/sglang-source-reading/02-scheduler-runtime/02-scheduler-core.md)、[`03-kv-cache-radix-cache.md`](./learning/sglang-source-reading/03-cache-memory/03-kv-cache-radix-cache.md)、[`04-model-runner-attention.md`](./learning/sglang-source-reading/04-model-execution/04-model-runner-attention.md)。
4. 想深入执行层，就进入 [`learning/tp-worker-model-runner/`](./learning/tp-worker-model-runner/)；想深入调度层，就进入 [`learning/scheduler-architecture/`](./learning/scheduler-architecture/)。
5. 想学习 layer 级通信和 kernel 组织，读 [`01-layer-communicator-and-common-layers.md`](./learning/sglang-source-reading/05-layer-communication/01-layer-communicator-and-common-layers.md)。
6. 想学习高级 serving 优化，再读 [`05-speculative-decoding.md`](./learning/sglang-source-reading/06-advanced-features/05-speculative-decoding.md)、[`07-disaggregation-pd.md`](./learning/sglang-source-reading/06-advanced-features/07-disaggregation-pd.md)、[`08-lora-serving.md`](./learning/sglang-source-reading/06-advanced-features/08-lora-serving.md)、[`09-router.md`](./learning/sglang-source-reading/01-entry-routing/09-router.md)。
7. 如果目标是 Ascend NPU 部署和优化，按 [`learning/sglang-ascend-npu/README.md`](./learning/sglang-ascend-npu/README.md) 中的 0 到 15 讲顺序推进。

## 与 SGLang 原项目的关系

SGLang 的版权和许可证归原项目贡献者所有。本仓库只在教学目的下保留必要源码片段和阅读材料，并尽量使用相对路径引用源码，避免依赖某台机器上的本地路径或不稳定的代码行号。

如果你想运行、部署或参与开发 SGLang，请以官方仓库为准：

```text
https://github.com/sgl-project/sglang
```
