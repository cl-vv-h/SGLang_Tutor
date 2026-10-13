# 03. Gloo, NCCL, RCCL, HCCL, and ProcessGroup

**English** | [简体中文](../../../zh/ai-infra-basic/Communication/03-collective-libraries-gloo-nccl-hccl.md)

## 1. One AllReduce, Three Layers

```text
model: Y = sum(partial_Y_from_each_TP_rank)
  -> framework: torch.distributed.all_reduce(Y, group=tp_group)
  -> backend/runtime: Gloo / NCCL / HCCL / another implementation
  -> devices and links
```

The model defines the mathematics, ProcessGroup defines participants and the interface, and the backend implements it. SGLang may bypass ordinary `torch.distributed` collectives through PyNccl, custom AllReduce, or other optimized paths. A default `nccl` backend therefore does not prove every call traverses ProcessGroupNCCL.

## 2. Positioning the Main Libraries

| Library | Platform and Role in This Course | Typical Use | Selection Boundary |
|---|---|---|---|
| Gloo | CPU collectives and process coordination | CPU learning exercises, small tensors, host coordination | not a universal substitute for high-throughput NVIDIA/Ascend device communication |
| NCCL | NVIDIA GPU collectives / P2P | TP, training gradients, device data exchange | algorithms vary with topology, size, and version |
| RCCL | AMD GPU collectives | multi-GPU ROCm training and inference | PyTorch ROCm commonly retains the `nccl` backend name; the string alone does not identify hardware |
| HCCL | Huawei Collective Communication Library in this chapter | Ascend NPU TP/EP communication | integrated through compatible CANN, drivers, and `torch_npu` |
| MPI / UCX and others | HPC process communication ecosystem / transport abstraction | cluster applications and lower layers of some backends | MPI is a standard and implementation ecosystem; UCX occupies a different layer |

Do not describe Gloo as inherently CPU-only: device and primitive support depend on the project and PyTorch build. CPU is chosen here for an accessible learning path. [Gloo](https://github.com/pytorch/gloo), [PyTorch distributed](https://docs.pytorch.org/docs/stable/distributed.html), [RCCL documentation](https://rocm.docs.amd.com/projects/rccl/en/latest/)

See the [Ascend HCCL introduction](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/commlib/hcclug/docs/en/user_guide/hccl_intro.md) and [torch_npu adapter](https://github.com/Ascend/pytorch). Other vendors may use the same acronym; HCCL here always means the Ascend implementation.

## 3. Why SGLang Creates CPU and Device Groups

The ordinary initialization branch in [GroupCoordinator](../../../../python/sglang/srt/distributed/parallel_state.py) creates:

```python
device_group = torch.distributed.new_group(ranks, backend=torch_distributed_backend)
cpu_group = torch.distributed.new_group(ranks, backend="gloo")
```

This excerpt omits options/timeouts. The CPU group supports host coordination; the device group supports device tensors. They can have identical members without being the same communication object. There is also a specialized elastic Mooncake branch, so the ordinary path does not describe every mode.

When creating multiple groups, ranks must obey the framework's group-creation ordering requirements. Collectives must satisfy the API's group, operation-order, dtype, and shape/split contracts. An OOM or exception on one rank can appear as a communication timeout on others; a timeout does not necessarily indicate a NIC fault.

## 4. Minimal Gloo Experiment: A Real Two-Process Sum

Save the following as `gloo_demo.py` and run it in a PyTorch environment with distributed/Gloo support. It uses CPU tensors and needs neither CUDA nor Ascend.

```python
from datetime import timedelta

import torch
import torch.distributed as dist


def main():
    dist.init_process_group("gloo", timeout=timedelta(seconds=30))
    try:
        rank = dist.get_rank()
        world = dist.get_world_size()
        x = torch.tensor([float(rank + 1)])
        gathered = [torch.empty_like(x) for _ in range(world)]
        dist.all_gather(gathered, x)
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        expected = world * (world + 1) / 2
        assert x.item() == expected
        assert [t.item() for t in gathered] == list(range(1, world + 1))
        if rank == 0:
            print("AllGather:", [t.item() for t in gathered])
            print("AllReduce:", x.item())
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

```bash
torchrun --standalone --nproc-per-node=2 gloo_demo.py
```

Expected: AllGather is `[1.0, 2.0]`, and AllReduce is `3.0`. Changing only the backend string to `hccl` or `nccl` is insufficient: importing the adapter, binding the local device, using device tensors, and satisfying hardware/software dependencies are also necessary. This CPU example is not NPU validation.

## 5. Asynchronous Handles and Streams

`async_op=True` returns a work handle; it does not establish useful overlap. Establish dependencies before consumption and preserve buffer lifetimes. For device backends, the precise relationship between `wait()`, current-stream ordering, and host blocking depends on the API. Timing only the Python launch is insufficient.

Not every collective is in-place. For example, `all_reduce` updates its input tensor, whereas `all_gather` needs output storage. Verify the mathematical result and layout before discussing fusion, graph capture, streams, or kernel selection.

## 6. Reading Checks

1. Why are groups with identical members not interchangeable API arguments?
2. Does successful Gloo initialization prove HCCL/NCCL device traffic works?
3. Does waiting for a CPU-group message imply that traffic counts toward device AllReduce bandwidth?

Next: [MoE, DeepEP, and ZBAL](./04-moe-deepep-and-zbal.md).
