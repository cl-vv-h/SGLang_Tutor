[中文](./01-environment-and-install.md) | [English](./01-environment-and-install_EN.md)

# 01. Environment & Installation

## NPU Python Package Configuration

SGLang uses `python/pyproject_npu.toml` for NPU-specific dependencies:

```toml
[project.optional-dependencies]
srt_npu = ["torch_npu", "sgl_kernel_npu"]
all_npu = ["sglang[srt_npu]", "transformers", ...]
dev_npu = ["sglang[all_npu]", "pytest", ...]
```

## Environment Checklist

| Component | Purpose | Verification |
|---|---|---|
| CANN Toolkit | Ascend runtime, ACL, HCCL | `npu-smi info` |
| `torch_npu` | PyTorch NPU adapter | `python -c "import torch_npu; print(torch_npu.npu.is_available())"` |
| `sgl_kernel_npu` | SGLang NPU kernels | `pip show sgl_kernel_npu` |
| HCCL | Distributed communication | HCCL env vars configured |
| `memfabric_hybrid` | PD transfer (optional) | Check package availability |

## Minimal Installation

```bash
# Install CANN toolkit (version matched to torch_npu)
# Install torch_npu
pip install torch-npu  # Version must match CANN

# Install SGLang with NPU extras
pip install -e "python/[srt_npu]"

# Verify
python -c "
import torch
import torch_npu
print(f'torch_npu available: {torch_npu.npu.is_available()}')
print(f'Device count: {torch_npu.npu.device_count()}')
"
```

## Key Compatibility Matrix

| Component | Version Range | Notes |
|---|---|---|
| CANN | 8.0.RC2 - 9.0.0 | Must match torch_npu version |
| torch_npu | 2.x | Check compatibility with PyTorch version |
| sgl_kernel_npu | Latest | Pin to tested commit |
| Triton-Ascend | 3.2.1+ | Optional, for custom kernels |
