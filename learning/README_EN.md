[中文](./README.md) | **English**

# SGLang Learning Hub

This directory contains Chinese-language educational materials for source code learning. The original SGLang code in the repository is used only as teaching reference; learning notes aim to locate source code through "file + function + key code snippets" approach, avoiding dependency on easily shifting absolute paths.

## Topic Directory

- [sglang-source-reading](./sglang-source-reading/): SGLang source code overview reading roadmap.
- [scheduler-architecture](./scheduler-architecture/): `Scheduler` architecture, request scheduling flow, flowcharts, and annotated code walkthrough with Chinese comments.
- [tp-worker-model-runner](./tp-worker-model-runner/): `TpModelWorker` and `ModelRunner` architecture, execution flow, function mapping, and Chinese-annotated source copies.
- [sglang-ascend-npu](./sglang-ascend-npu/): SGLang Ascend NPU practical overview, covering environment, NPU backend, attention, graph, HCCL, PD disaggregation, LoRA, and optimization roadmap.
- [ai-infra-basic](./ai-infra-basic/): AI Infra fundamentals, organized by inference basics, schedule optimization, KV Cache, Attention Kernel, execution graph, Mamba/SSM, parallel strategies, KV transfer, speculative decoding, quantization, LoRA, and benchmark/profiling.
- [ascend-kernel-infra](./ascend-kernel-infra/): Diving deeper into the Ascend NPU operator layer, learning the relationships and programming models of sgl-kernel-npu, Triton-Ascend, Ascend C, and torch_npu, along with performance optimization.
