# SGLang Tutor

**English** | [简体中文](./README.zh-CN.md)

SGLang Tutor is a bilingual learning repository for understanding the implementation and optimization of [SGLang](https://github.com/sgl-project/sglang). It combines a source snapshot used for code reading with structured lessons on the serving request lifecycle, scheduling, model execution, KV cache, attention, disaggregated serving, routing, Ascend NPU, and AI infrastructure fundamentals.

> This is an educational repository, not an alternative SGLang distribution. For installation, deployment, releases, and upstream development, use the [official SGLang repository](https://github.com/sgl-project/sglang).

## Choose a Language

| Language | Learning hub |
|---|---|
| English | [Open the English curriculum](./learning/en/) |
| 简体中文 | [打开中文课程目录](./learning/zh/) |

The bilingual directory convention and contribution workflow are documented in [learning/LANGUAGE.md](./learning/LANGUAGE.md).

## Learning Map

The English and Chinese curricula use identical paths below their language directory, so the same topic is easy to locate in either language.

| Topic | English | 中文 | What you will learn |
|---|---|---|---|
| AI Infra fundamentals | [English](./learning/en/ai-infra-basic/) | [中文](./learning/zh/ai-infra-basic/) | Inference basics, model architectures, scheduling, KV cache, attention kernels, execution graphs, parallelism, KV transfer, speculative decoding, quantization, LoRA, and profiling |
| SGLang source reading | [English](./learning/en/sglang-source-reading/) | [中文](./learning/zh/sglang-source-reading/) | Request lifecycle, router, scheduler runtime, cache and memory, model execution, layer communication, and advanced serving features |
| Scheduler architecture | [English](./learning/en/scheduler-architecture/) | [中文](./learning/zh/scheduler-architecture/) | Scheduler responsibilities, queues, batch formation, execution flow, function map, and annotated source |
| TP Worker and ModelRunner | [English](./learning/en/tp-worker-model-runner/) | [中文](./learning/zh/tp-worker-model-runner/) | The execution boundary between Scheduler, `TpModelWorker`, `ModelRunner`, attention backends, and distributed resources |
| SGLang on Ascend NPU | [English](./learning/en/sglang-ascend-npu/) | [中文](./learning/zh/sglang-ascend-npu/) | Environment setup, NPU backend integration, graph execution, HCCL, PD disaggregation, profiling, performance, and accuracy debugging |
| Ascend kernel infrastructure | [English](./learning/en/ascend-kernel-infra/) | [中文](./learning/zh/ascend-kernel-infra/) | Ascend hardware and memory, CANN, Ascend C, Triton-Ascend, `torch_npu`, and `sgl-kernel-npu` operator paths |

## Recommended Route

1. Start with [Inference Basics](./learning/en/ai-infra-basic/Inference_Basics/) to build a mental model of prefill, decode, batching, latency, throughput, and KV cache.
2. Read the [SGLang component overview](./learning/en/sglang-source-reading/00-overview/01-public-components-code-walkthrough.md).
3. Follow one request through the [request lifecycle](./learning/en/sglang-source-reading/01-entry-routing/01-request-lifecycle.md), [Scheduler](./learning/en/sglang-source-reading/02-scheduler-runtime/02-scheduler-core.md), [KV cache](./learning/en/sglang-source-reading/03-cache-memory/03-kv-cache-radix-cache.md), and [ModelRunner/attention](./learning/en/sglang-source-reading/04-model-execution/04-model-runner-attention.md).
4. Continue with the focused [Scheduler](./learning/en/scheduler-architecture/) and [TP Worker/ModelRunner](./learning/en/tp-worker-model-runner/) tracks.
5. Choose an advanced path: [speculative decoding](./learning/en/sglang-source-reading/06-advanced-features/05-speculative-decoding.md), [PD disaggregation](./learning/en/sglang-source-reading/06-advanced-features/07-disaggregation-pd.md), [LoRA serving](./learning/en/sglang-source-reading/06-advanced-features/08-lora-serving.md), [routing](./learning/en/sglang-source-reading/01-entry-routing/09-router.md), or [Ascend NPU](./learning/en/sglang-ascend-npu/).

## Repository Layout

| Path | Purpose |
|---|---|
| [`learning/`](./learning/) | Bilingual curriculum and language navigation |
| [`python/`](./python/) | SGLang Python runtime snapshot referenced by the lessons |
| [`sgl-kernel/`](./sgl-kernel/) | Kernel code and Python bindings referenced by runtime lessons |
| [`experimental/sgl-router/`](./experimental/sgl-router/) | Rust router source used by routing lessons |
| [`rust/sglang-grpc/`](./rust/sglang-grpc/) | Rust gRPC extension source |
| [`proto/`](./proto/) | Protobuf definitions used by the gRPC extension |

## Upstream and License

SGLang source code remains copyrighted by its upstream contributors. This repository keeps the Apache License 2.0 information in [LICENSE](./LICENSE) and uses the retained source only as teaching reference.
