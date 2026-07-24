[中文](./README.md) | [English](./README_EN.md)

# TP Worker & ModelRunner Learning Directory

This directory is for reading the implementations of `tp_worker.py` and `model_runner.py` in SGLang. They are located at:

- `python/sglang/srt/managers/tp_worker.py`
- `python/sglang/srt/model_executor/model_runner.py`

No original source code is modified here; instead, this directory provides educational documentation and Chinese-annotated source code copies for convenient side-by-side reading.

## Reading Order

1. `01-architecture.md`: First understand where `TpModelWorker` and `ModelRunner` sit among Scheduler, TP/PP, KV cache, and attention backend.
2. `02-flowcharts.md`: Use flowcharts to connect initialization, request execution, prefill/decode, sampling, and graph replay.
3. `03-function-map.md`: Locate key logic by function/code segment for easy cross-referencing with source code.
4. `04-tp-worker-annotated-cn.py`: Chinese-annotated copy of `tp_worker.py`.
5. `05-model-runner-annotated-cn.py`: Chinese-annotated copy of `model_runner.py`.

## Core Insights

`TpModelWorker` is the adapter between the Scheduler and the model execution layer. It understands the current process's TP rank, PP rank, draft/target worker identity, and converts a scheduled `ScheduleBatch` into a `ForwardBatch`.

`ModelRunner` is the true model execution core. It handles distributed initialization, model loading, KV cache memory pool, attention backend, CUDA graph/piecewise graph, forward dispatch, logits preprocessing, and sampling.
