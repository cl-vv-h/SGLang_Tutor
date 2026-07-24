[中文](./02-flowcharts.md) | [English](./02-flowcharts_EN.md)

# Detailed Flowcharts

## 1. From Scheduler to Model Execution

```mermaid
flowchart TB
    Scheduler["Scheduler selects runnable requests"] --> ScheduleBatch["Construct / Update ScheduleBatch"]
    ScheduleBatch --> WorkerCall["TpModelWorker.forward_batch_generation(batch)"]
    WorkerCall --> SetConsumer["Set HiCache consumer<br/>Optional"]
    SetConsumer --> FB["ForwardBatch.init_new(batch, model_runner)"]
    FB --> DLLM{"dLLM worker?"}
    DLLM -- Yes --> DLLMPath["_forward_batch_generation_dllm()"]
    DLLM -- No --> PP{"Is current PP rank the last stage?"}
    PP -- No --> PPForward["ModelRunner.forward()<br/>Returns PPProxyTensors"]
    PPForward --> PPResult["GenerationBatchResult<br/>Only carries hidden state proxy"]
    PP -- Yes --> MRForward["ModelRunner.forward()"]
    MRForward --> Verify{"is_verify?"}
    Verify -- Yes --> ReturnLogits["Directly return logits/hidden states<br/>No sampling"]
    Verify -- No --> Sampling{"prefill-only?"}
    Sampling -- No --> Sample["ModelRunner.sample(...)"]
    Sampling -- Yes --> Logprob["compute_logprobs_only or dummy token"]
    Sample --> Result["GenerationBatchResult"]
    Logprob --> Result
```

Key code segments:

- `TpModelWorker.forward_batch_generation()`: Generation entry point.
- `ForwardBatch.init_new(batch, self.model_runner)`: Conversion from scheduler batch to model batch.
- `self.pp_group.is_last_rank`: Distinguishes PP intermediate stages from the last stage.
- `self.model_runner.sample(...)`: Only samples when it's the last stage and needs token generation.

## 2. TpModelWorker Initialization

```mermaid
flowchart TB
    Start["TpModelWorker.__init__"] --> Save["Save server_args, rank, device, port, etc."]
    Save --> Config["_init_model_config()"]
    Config --> Runner["_init_model_runner()"]
    Runner --> MultiEagle{"multi-layer EAGLE?"}
    MultiEagle -- Yes --> CreateMany["Create multiple draft ModelRunners"]
    MultiEagle -- No --> Continue["Continue"]
    CreateMany --> Continue
    Continue --> DLLM["_init_dllm_algorithm()"]
    DLLM --> Tokenizer["get_tokenizer(...) / get_processor(...)"]
    Tokenizer --> Groups["get_pp_group() / get_world_group()"]
    Groups --> Capacity["Read max_total_num_tokens / max_running_requests"]
    Capacity --> Seed["Sync random seed"]
    Seed --> Ready["worker ready"]
```
