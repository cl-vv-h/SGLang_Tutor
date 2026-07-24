[中文](./02-server-args-and-npu-defaults.md) | [English](./02-server-args-and-npu-defaults_EN.md)

# Foundation 02: Server Args & NPU Defaults

## set_default_server_args() in Detail

```python
def set_default_server_args(args):
    """
    Override ServerArgs with NPU-appropriate defaults.
    Called early in Scheduler initialization when is_npu() is True.
    """
    # Attention backend
    if args.attention_backend is None:
        args.attention_backend = "ascend"
        args.prefill_attention_backend = "ascend"
        args.decode_attention_backend = "ascend"
    
    # KV Cache page size (128 for better NPU alignment)
    if args.page_size is None:
        args.page_size = 128
    
    # Chunked prefill: adjust for NPU memory capacity
    if args.chunked_prefill_size is None:
        args.chunked_prefill_size = get_npu_chunked_size(args)
    
    # Graph batch size: adjust for NPU memory
    if args.cuda_graph_max_bs is None:
        args.cuda_graph_max_bs = get_npu_graph_max_bs(args)
    
    # Disable CUDA-specific features
    args.disable_custom_all_reduce = True
    
    # HiCache NPU backend
    if args.enable_hierarchical_cache:
        args.hicache_io_backend = "kernel_ascend"
```

## NPU-Specific Server Args Reference

| Arg | GPU Default | NPU Default | Why |
|---|---|---|---|
| `attention_backend` | `flashinfer` | `ascend` | Ascend uses own kernels |
| `page_size` | 16 | 128 | NPU memory alignment |
| `chunked_prefill_size` | 4096-8192 | NPU-memory-based | Different HBM capacity |
| `cuda_graph_max_bs` | Various | NPU-memory-based | NPUGraph memory model |
| `disable_custom_all_reduce` | False | True | No CUDA all-reduce on NPU |
| `speculative_attention_mode` | False | False | Spec decode paths differ |
