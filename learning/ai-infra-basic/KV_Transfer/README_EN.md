[中文](./README.md) | [English](./README_EN.md)

# KV Transfer (PD Disaggregation)

Prefill/Decode disaggregation: splitting LLM serving into separate prefill and decode workers, with KV Cache transferred between them.

## Core Idea

Instead of running prefill and decode on the same GPU, disaggregated serving assigns them to different machines:

- **Prefill worker**: Handles the compute-heavy prompt processing, then sends KV Cache to decode workers
- **Decode worker**: Receives KV Cache and handles the memory-bound token generation

## Key Concepts

| Concept | Description |
|---|---|
| Bootstrap | Initial handshake where prefill and decode workers establish transfer channels |
| Prealloc | Decode worker pre-allocates KV Cache slots before receiving data |
| KV Transfer | Actual KV tensor transfer via RDMA, SDMA, or other protocols |
| Transfer Engine | Abstraction layer for different transfer backends (Mooncake, Ascend, NIXL) |

## SGLang Integration

- `disaggregation_mode` determines worker role (NULL/PREFILL/DECODE)
- `AscendTransferEngine` handles Ascend NPU-specific KV transfer
- Bootstrap/prealloc queues manage the lifecycle of disaggregated requests
