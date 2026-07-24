[中文](./03-request-lifecycle-npu-branch-points.md) | [English](./03-request-lifecycle-npu-branch-points_EN.md)

# Foundation 03: Request Lifecycle NPU Branch Points

## Annotated Lifecycle with NPU Branches

```mermaid
flowchart TD
  A["HTTP Request"] --> B["TokenizerManager"]
  B --> C["Scheduler.process_input_requests()"]
  C --> D["handle_generate_request()"]
  D --> E["_add_request_to_queue()"]
  E --> F["get_next_batch_to_run()"]
  
  F --> G{"is_npu()?"}
  G -->|"Yes"| H["NPU: page_size=128,<br/>chunked_prefill scaled"]
  G -->|"No"| I["GPU: page_size=16,<br/>standard prefill"]
  
  H --> J["run_batch()"]
  I --> J
  
  J --> K["TpModelWorker<br/>→ ModelRunner"]
  K --> L{"is_npu()?"}
  L -->|"Yes"| M["Ascend Attention<br/>FRACTAL_NZ KV Cache<br/>NPUGraph replay"]
  L -->|"No"| N["FlashInfer/Triton<br/>Standard KV Cache<br/>CUDAGraph replay"]
  
  M --> O["process_batch_result()"]
  N --> O
  O --> P["Detokenizer → HTTP Response"]
```

## Key NPU Branch Points

| Stage | What's NPU-Specific | Source |
|---|---|---|
| Startup | `is_npu()` detection, `init_npu_backend()` | `utils.py` |
| Config | `set_default_server_args()` overrides | `utils.py` |
| Scheduling | `page_size=128`, adjusted chunk sizes | `scheduler.py` |
| Attention | `AscendAttnBackend`, FRACTAL_NZ | `layers/attention/ascend/` |
| Graph | `NPUGraph` instead of `CUDAGraph` | `npu_piecewise_backend.py` |
| Communication | `hccl` instead of `nccl` | `parallel_state.py` |
| Kernels | `torch.ops.npu`, `sgl_kernel_npu` | Various |
| Transfer | `AscendTransferEngine` | `disaggregation/ascend/` |

## NPU Branch Detection Pattern

```python
# Pattern 1: Conditional import
if is_npu():
    from sglang.srt.hardware_backend.npu import init_npu_backend

# Pattern 2: Conditional execution
if is_npu():
    backend = "hccl"
else:
    backend = "nccl"

# Pattern 3: Feature toggle
self.use_npu_graph = is_npu() and args.enable_graph
```
