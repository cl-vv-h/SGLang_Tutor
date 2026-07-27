# TP Worker 与 ModelRunner 教学目录

本目录用于阅读 SGLang 中 `tp_worker.py` 与 `model_runner.py` 的实现。两者分别位于：

- `python/sglang/srt/managers/tp_worker.py`
- `python/sglang/srt/model_executor/model_runner.py`

这里不会修改原始源码，而是提供教学文档与带中文注释的源码副本，方便逐段对照阅读。

## 阅读顺序

1. `01-architecture.md`：先理解 `TpModelWorker` 与 `ModelRunner` 在 Scheduler、TP/PP、KV cache、attention backend 之间的位置。
2. `02-flowcharts.md`：用流程图串起初始化、请求执行、prefill/decode、采样和 graph replay。
3. `03-function-map.md`：按函数/代码段定位关键逻辑，便于回到源码中查找。
4. `04-tp-worker-annotated-cn.py`：`tp_worker.py` 的中文注释版副本。
5. `05-model-runner-annotated-cn.py`：`model_runner.py` 的中文注释版副本。

## 核心结论

`TpModelWorker` 是 Scheduler 与模型执行层之间的适配器。它理解当前进程的 TP rank、PP rank、draft/target worker 身份，并把调度好的 `ScheduleBatch` 转成 `ForwardBatch`。

`ModelRunner` 是真正的模型执行核心。它负责分布式初始化、模型加载、KV cache 内存池、attention backend、CUDA graph/piecewise graph、前向分发、logits 预处理和 sampling。
