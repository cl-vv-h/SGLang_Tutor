# SGLang 教学目录

这里存放面向源码学习的中文教学材料。仓库中的 SGLang 原版代码仅作为教学引用，学习笔记会尽量通过“文件 + 函数 + 关键代码段”的方式定位源码，避免依赖容易漂移的绝对路径。

## 专题目录

- [sglang-source-reading](./sglang-source-reading/)：SGLang 源码总览阅读路线。
- [scheduler-architecture](./scheduler-architecture/)：`Scheduler` 架构、请求调度流程、流程图与带中文注释的代码导读。
- [tp-worker-model-runner](./tp-worker-model-runner/)：`TpModelWorker` 与 `ModelRunner` 的架构、执行流程、函数定位和中文注释版源码副本。
- [sglang-ascend-npu](./sglang-ascend-npu/)：SGLang Ascend NPU 实践总览，覆盖环境、NPU 后端、attention、graph、HCCL、PD 分离、LoRA 与调优路线。
- [ai-infra-basic](./ai-infra-basic/)：AI Infra 基础专题，覆盖 Attention kernel、LoRA/QLoRA/DoRA/AdaLoRA、推理并行策略、Prefill/Decode 和 Chunked Prefill 调度优化。
