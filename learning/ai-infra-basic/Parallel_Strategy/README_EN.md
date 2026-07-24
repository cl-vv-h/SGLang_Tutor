[中文](./README.md) | [English](./README_EN.md)

# Parallel Strategy

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
| [dp_inference_demo.py](./dp_inference_demo.py) | Data Parallelism demo |
| [tp_inference_demo.py](./tp_inference_demo.py) | Tensor Parallelism demo |
| [pp_inference_demo.py](./pp_inference_demo.py) | Pipeline Parallelism demo |
| [sp_inference_demo.py](./sp_inference_demo.py) | Sequence/Context Parallelism demo |
| [ep_moe_demo.py](./ep_moe_demo.py) | Expert Parallelism demo for MoE |

## SGLang Integration

- TP/PP ranks are initialized in `ParallelState` during `Scheduler.__init__`
- Communication backends (NCCL/HCCL) selected based on device type
- `LayerCommunicator` abstracts TP/EP/CP communication patterns
- EP uses `deep_ep::Buffer` for dispatch/combine in MoE layers
