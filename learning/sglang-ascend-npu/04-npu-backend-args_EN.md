[中文](./04-npu-backend-args.md) | [English](./04-npu-backend-args_EN.md)

# 04. NPU Backend Args & Default Overrides

## Core Source

`python/sglang/srt/hardware_backend/npu/utils.py`

## Key Functions

| Function | Purpose |
|---|---|
| `init_npu_backend()` | Initialize NPU runtime: device detection, env setup, ZBAL init |
| `set_default_server_args(args)` | Override server args with NPU-appropriate defaults |
| `npu_format_cast(tensor, format)` | Cast tensor to Ascend-specific formats (e.g., FRACTAL_NZ) |
| `init_zbal(...)` | Initialize Zero-Balance memory optimization |
| `lazy_init_zbal_gva_mem(...)` | Lazy init of ZBAL global virtual address memory |

## Default Parameter Overrides

| Parameter | GPU Default | NPU Default | Rationale |
|---|---|---|---|
| `attention_backend` | `flashinfer` | `ascend` | NPU uses Ascend-specific attention kernels |
| `prefill_attention_backend` | Same as attention | `ascend` | Prefill needs NPU-compatible kernels & metadata |
| `decode_attention_backend` | Same as attention | `ascend` | Decode low-latency path needs fixed backend |
| `page_size` | `16` | `128` | Better alignment for NPU memory access patterns |
| `chunked_prefill_size` | Dynamic | Based on NPU HBM | Adjust for different memory capacity |
| `cuda_graph_max_bs` | Dynamic | Adjusted for NPU | NPUGraph uses different memory model |
| `disable_custom_all_reduce` | `False` | `True` | NPU doesn't support CUDA custom all-reduce |
| `speculative_attention_mode` | Default | `false` | Spec decode may use different attention paths |

## HiCache NPU Configuration

| Setting | Value | Notes |
|---|---|---|
| HiCache backend | `kernel_ascend` | Ascend-specific kernel for hierarchical cache I/O |
| KV layout | FRACTAL_NZ compatible | Matches Ascend's optimal matrix format |

## Reading Path

1. Find `init_npu_backend()` — understand runtime initialization
2. Find `set_default_server_args()` — see all NPU overrides
3. Trace `attention_backend = "ascend"` — how backend is selected
4. Check `page_size = 128` — why different from GPU default
5. Understand `disable_custom_all_reduce = True` — NPU communication difference
