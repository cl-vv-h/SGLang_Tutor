[中文](./00-reading-method-and-branch-search.md) | [English](./00-reading-method-and-branch-search_EN.md)

# Foundation 00: Reading Method & Branch Search

## Source Reading Method

### Search Strategy

```text
1. Identify the feature → Find `is_npu()` check → Follow the branch
2. grep for "ascend" or "npu" in relevant directories
3. Check `hardware_backend/npu/` for NPU-specific modules
4. Look for `torch.ops.npu` calls in execution paths
5. Search for `disable=_is_npu` patterns for fallback identification
```

### Branch Identification

| Pattern | Meaning | Example |
|---|---|---|
| `if is_npu():` | NPU-specific code path | `if is_npu(): backend = "hccl"` |
| `attention_backend="ascend"` | Using Ascend attention | `self.attention_backend = "ascend"` |
| `torch.npu.*` | Direct NPU API call | `torch.npu.NPUGraph()` |
| `torch.ops.npu.*` | NPU custom operator | `torch.ops.npu.sgmv_shrink()` |
| `device="npu"` | Device type check | `if device == "npu":` |
| `# NPU:` comment | Documentation hint | `# NPU: use FRACTAL_NZ format` |

## Git Search Commands

```bash
# Find all NPU-related files
git grep -l "is_npu" -- "*.py"

# Find Ascend attention backend
git grep "ascend" -- "srt/layers/attention/"

# Find NPU graph usage
git grep "NPUGraph" -- "*.py"

# Find torch_npu imports
git grep "import torch_npu" -- "*.py"
```

## Reading Order

1. Start at entry points: `is_npu()`, `init_npu_backend()`
2. Follow configuration: `set_default_server_args()`
3. Trace execution: `ModelRunner.forward()` → attention/graph/communication
4. Dive into features: LoRA, MoE, HiCache, PD transfer
5. Profile and optimize: kernels, communication, memory
