[中文](./README.md) | [English](./README_EN.md)

# Schedule Optimization

This topic explains the key scheduling optimizations that make LLM serving systems efficient: continuous batching, chunked prefill, and the fundamental trade-offs between throughput and latency.

## Core Concepts

| Concept | Description |
|---|---|
| Continuous Batching | New requests can join a running batch without waiting for existing requests to finish |
| Chunked Prefill | Long prompts are split into multiple prefill chunks, interleaved with decode steps to avoid blocking |
| Prefill/Decode Trade-off | Prefill is compute-bound, decode is memory-bound — scheduling must balance both |
| Priority Scheduling | Higher-priority or shorter requests can preempt lower-priority ones |

## Files

| File | Content |
|---|---|
| [prefill_decode_demo.py](./prefill_decode_demo.py) | Demo showing prefill vs decode characteristics |
| [chunked_prefill_with_fakeLLM_tutorial.py](./chunked_prefill_with_fakeLLM_tutorial.py) | Tutorial demonstrating chunked prefill with a fake LLM |

## SGLang Integration

- `Scheduler.get_next_batch_to_run()` implements the prefill-first, decode-fallback policy
- `PrefillAdder` enforces token and request budgets for chunked prefill
- `SchedulePolicy.calc_priority()` determines request ordering
- `update_running_batch()` handles decode memory pressure via retraction
