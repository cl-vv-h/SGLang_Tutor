# Parallel Strategy

[简体中文](../../../zh/ai-infra-basic/Parallel_Strategy/README.md) | **English**

This topic covers the parallelization strategies used in LLM inference: Data Parallelism (DP), Tensor Parallelism (TP), Pipeline Parallelism (PP), Sequence Parallelism (SP), and Expert Parallelism (EP).

## Strategies Overview

| Strategy | How It Works | Best For | Communication |
|---|---|---|---|
| DP | Replicate model, split batch across devices | Increasing throughput | All-reduce gradients (training only) |
| TP | Split weight matrices column/row-wise across devices | Models too large for single GPU | All-reduce / all-gather per layer |
| PP | Split model layers across devices, micro-batch pipeline | Very deep models | Send/recv activations between stages |
| SP/CP | Split sequence length across devices | Long context inference | All-to-all for attention |
| EP | Place different experts on different devices | Large MoE models | All-to-all for token dispatch/combine |

## Files

| File | Content |
|---|---|
| [tutorial.md](./tutorial.md) | Comprehensive parallel strategy tutorial |
| [01-dp-attention-principles.md](./01-dp-attention-principles.md) | DP Attention: request ownership, attention subgroups, FFN/MoE layouts, partial/replicated values, padding, IDLE, and KV ownership |
| [dp_attention_layout_demo.py](./dp_attention_layout_demo.py) | Standard-library CPU simulator for SUM_LEN/MAX_LEN gathering, replica deduplication, and reduce-scatter output returns |
| [dp_inference_demo.py](./dp_inference_demo.py) | Data Parallelism demo |
| [tp_inference_demo.py](./tp_inference_demo.py) | Tensor Parallelism demo |
| [pp_inference_demo.py](./pp_inference_demo.py) | Pipeline Parallelism demo |
| [sp_inference_demo.py](./sp_inference_demo.py) | Sequence/Context Parallelism demo |
| [ep_moe_demo.py](./ep_moe_demo.py) | Expert Parallelism demo for MoE |

## SGLang Integration

For `--enable-dp-attention`, read [DP Attention principles](./01-dp-attention-principles.md), run the layout simulator, then follow the [end-to-end source walkthrough](../../sglang-source-reading/05-layer-communication/03-dp-attention-code-walkthrough.md). DP Attention combines request parallelism, attention TP, and FFN/MoE parallelism inside one model world. Its device count and communication differ from ordinary complete-model DP replicas.

- TP/PP ranks are initialized in `ParallelState` during `Scheduler.__init__`
- Communication backends (NCCL/HCCL) selected based on device type
- `LayerCommunicator` abstracts TP/EP/CP communication patterns
- EP uses `deep_ep::Buffer` for dispatch/combine in MoE layers

## Communication Foundations

After sharding strategies, read [Communication](../Communication/README.md): tensor semantics for AllReduce/AllGather/ReduceScatter/AllToAllV, followed by Gloo, NCCL, HCCL, DeepEP, and ZBAL implementation boundaries.
