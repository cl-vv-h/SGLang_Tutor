# SGLang Tutor

[English](./README.md) | **简体中文**

SGLang Tutor 是一个面向 [SGLang](https://github.com/sgl-project/sglang) 实现原理与推理优化的双语教学仓库。仓库保留教学所需的源码快照，并围绕 serving 请求生命周期、调度、模型执行、KV Cache、Attention、PD 分离、路由、Ascend NPU 和 AI Infra 基础机制组织系统化课程。

> 这是教学仓库，不是 SGLang 的替代发行版。安装、部署、版本发布和上游开发请以 [SGLang 官方仓库](https://github.com/sgl-project/sglang) 为准。

## 选择语言

| 语言 | 教学入口 |
|---|---|
| 简体中文 | [打开中文课程目录](./learning/zh/) |
| English | [Open the English curriculum](./learning/en/) |

中英文目录约定和翻译维护流程见 [learning/LANGUAGE.md](./learning/LANGUAGE.md)。

## 学习地图

中英文课程在各自语言目录下采用完全一致的相对路径，因此可以快速找到同一专题的另一语言版本。

| 专题 | 中文 | English | 学习内容 |
|---|---|---|---|
| AI Infra 基础 | [中文](./learning/zh/ai-infra-basic/) | [English](./learning/en/ai-infra-basic/) | 推理基础、模型架构、调度、KV Cache、Attention Kernel、执行图、并行策略、KV 传输、投机解码、量化、LoRA 和性能分析 |
| SGLang 源码阅读 | [中文](./learning/zh/sglang-source-reading/) | [English](./learning/en/sglang-source-reading/) | 请求生命周期、Router、Scheduler runtime、缓存与内存、模型执行、层间通信和高级 serving 特性 |
| Scheduler 架构 | [中文](./learning/zh/scheduler-architecture/) | [English](./learning/en/scheduler-architecture/) | Scheduler 职责、队列、batch 构造、执行流程、函数地图和注释源码 |
| TP Worker 与 ModelRunner | [中文](./learning/zh/tp-worker-model-runner/) | [English](./learning/en/tp-worker-model-runner/) | Scheduler、`TpModelWorker`、`ModelRunner`、attention backend 和分布式资源之间的执行边界 |
| SGLang Ascend NPU | [中文](./learning/zh/sglang-ascend-npu/) | [English](./learning/en/sglang-ascend-npu/) | 环境部署、NPU 后端接入、Graph、HCCL、PD 分离、Profiling、性能和精度调试 |
| Ascend 算子基础设施 | [中文](./learning/zh/ascend-kernel-infra/) | [English](./learning/en/ascend-kernel-infra/) | Ascend 硬件与内存、CANN、Ascend C、Triton-Ascend、`torch_npu` 和 `sgl-kernel-npu` 算子路径 |

## 推荐学习路线

1. 从[推理基础](./learning/zh/ai-infra-basic/Inference_Basics/)开始，建立 prefill、decode、batching、延迟、吞吐和 KV Cache 的基本模型。
2. 阅读 [SGLang 公共组件总览](./learning/zh/sglang-source-reading/00-overview/01-public-components-code-walkthrough.md)。
3. 沿着[请求生命周期](./learning/zh/sglang-source-reading/01-entry-routing/01-request-lifecycle.md)、[Scheduler](./learning/zh/sglang-source-reading/02-scheduler-runtime/02-scheduler-core.md)、[KV Cache](./learning/zh/sglang-source-reading/03-cache-memory/03-kv-cache-radix-cache.md)和 [ModelRunner/Attention](./learning/zh/sglang-source-reading/04-model-execution/04-model-runner-attention.md)串起一次完整请求。
4. 继续学习独立的 [Scheduler](./learning/zh/scheduler-architecture/) 与 [TP Worker/ModelRunner](./learning/zh/tp-worker-model-runner/) 专题。
5. 选择高级方向：[投机解码](./learning/zh/sglang-source-reading/06-advanced-features/05-speculative-decoding.md)、[PD 分离](./learning/zh/sglang-source-reading/06-advanced-features/07-disaggregation-pd.md)、[LoRA serving](./learning/zh/sglang-source-reading/06-advanced-features/08-lora-serving.md)、[Router](./learning/zh/sglang-source-reading/01-entry-routing/09-router.md)或 [Ascend NPU](./learning/zh/sglang-ascend-npu/)。

## 仓库结构

| 路径 | 用途 |
|---|---|
| [`learning/`](./learning/) | 双语课程与语言导航 |
| [`python/`](./python/) | 教学文档引用的 SGLang Python runtime 源码快照 |
| [`sgl-kernel/`](./sgl-kernel/) | Runtime 教学引用的 kernel 代码与 Python binding |
| [`experimental/sgl-router/`](./experimental/sgl-router/) | Router 教学引用的 Rust 源码 |
| [`rust/sglang-grpc/`](./rust/sglang-grpc/) | Rust gRPC 扩展源码 |
| [`proto/`](./proto/) | gRPC 扩展使用的 protobuf 定义 |

## 上游项目与许可证

SGLang 源码版权归上游贡献者所有。本仓库在 [LICENSE](./LICENSE) 中保留 Apache License 2.0 信息，并仅将保留源码用于教学引用。
