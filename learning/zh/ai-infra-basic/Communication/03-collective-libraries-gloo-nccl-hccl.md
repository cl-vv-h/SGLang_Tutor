# 03. Gloo、NCCL、RCCL、HCCL 与 ProcessGroup

**简体中文** | [English](../../../en/ai-infra-basic/Communication/03-collective-libraries-gloo-nccl-hccl.md)

## 1. 一个 AllReduce，三个层次

```text
model: Y = sum(partial_Y_from_each_TP_rank)
  -> framework: torch.distributed.all_reduce(Y, group=tp_group)
  -> backend/runtime: Gloo / NCCL / HCCL / another implementation
  -> devices and links
```

模型定义数学语义，ProcessGroup 定义参与者和接口，backend 负责实现。SGLang 还可能绕过普通 `torch.distributed` collective，选择 PyNccl、自定义 AllReduce 或其他优化分支，所以“默认 backend=nccl”并不能证明每次调用都经过 ProcessGroupNCCL。

## 2. 主要库怎么定位

| 库 | 本课关注的平台与角色 | 典型场景 | 选型边界 |
|---|---|---|---|
| Gloo | CPU collective 与进程协调 | CPU 教学实验、小张量、主机侧协调 | 不要当成 NVIDIA/Ascend 高吞吐设备通信的统一替代 |
| NCCL | NVIDIA GPU collective / P2P | TP、训练梯度、设备数据交换 | 算法随拓扑、大小与版本变化 |
| RCCL | AMD GPU collective | ROCm 上的多 GPU 训练和推理 | PyTorch ROCm 常沿用 `nccl` backend 名称；不能仅凭字符串判断硬件 |
| HCCL | 本文特指 Huawei Collective Communication Library | Ascend NPU 的 TP/EP 等通信 | 通过匹配的 CANN、驱动与 `torch_npu` 集成 |
| MPI / UCX 等 | HPC 进程通信生态 / 传输抽象 | 集群程序、某些 backend 的下层 | MPI 是接口标准及其实现生态，UCX 是另一层工具 |

Gloo 项目不应被简单描述为“只可能支持 CPU”：具体设备与原语支持要看项目和 PyTorch 构建。这里选择 CPU 是为了可复现的学习路径。[Gloo 项目](https://github.com/pytorch/gloo)、[PyTorch distributed 文档](https://docs.pytorch.org/docs/stable/distributed.html)、[RCCL 文档](https://rocm.docs.amd.com/projects/rccl/en/latest/)

HCCL 的定位见 [Ascend HCCL 介绍](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/commlib/hcclug/docs/en/user_guide/hccl_intro.md)，适配入口见 [torch_npu](https://github.com/Ascend/pytorch)。其他厂商也可能使用同名缩写，本文的 HCCL 始终指 Ascend 这一实现。

## 3. 为什么 SGLang 同时创建 CPU group 与 device group

[GroupCoordinator](../../../../python/sglang/srt/distributed/parallel_state.py) 的普通初始化分支创建：

```python
device_group = torch.distributed.new_group(ranks, backend=torch_distributed_backend)
cpu_group = torch.distributed.new_group(ranks, backend="gloo")
```

这是去掉 options/timeout 后的源码摘意。CPU group 用于主机侧协调；device group 用于设备 tensor。它们可以具有相同成员，但不是同一个通信对象。源码还有 elastic Mooncake 专用分支，不能把这段普通路径推广到所有模式。

创建多个 group 时，各 rank 必须遵守框架要求的创建顺序；进入 collective 后，参与组、操作顺序、dtype、shape/splits 都要满足对应 API 契约。某个 rank 的 OOM/异常可能表现为其他 rank 通信超时，因此超时不一定说明网卡故障。

## 4. 最小 Gloo 实验：看见真实的两进程求和

把下面代码保存为 `gloo_demo.py`，在安装了 distributed/Gloo 支持的 PyTorch 环境中运行。使用 CPU，不需要 CUDA 或 Ascend。

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

预期：AllGather 为 `[1.0, 2.0]`，AllReduce 为 `3.0`。不要仅把 backend 字符串改为 `hccl` 或 `nccl` 就认为实验已迁移：还需导入对应适配器、绑定 local device、使用设备 tensor，并满足硬件/软件依赖。本文不把这段 CPU 示例当成 NPU 验证。

## 5. 异步句柄与 stream

`async_op=True` 返回工作句柄，并不证明有有效重叠。需要在消费前建立依赖，并遵守 buffer 生命周期。对于设备 backend，`wait()` 与当前 stream、host 阻塞之间的具体保证依 API 而定；测时不能只量 Python launch 时间。

也不要默认所有 collective 都是 in-place。例如 `all_reduce` 更新输入 tensor，而 `all_gather` 需要输出容器。优化通信前先确认数学输出与布局，再讨论 fusion、graph capture、stream 和 kernel 选择。

## 6. 阅读检查

1. 两个 group 的成员一样，为什么也不能混着传给 API？
2. Gloo 初始化成功，能证明 HCCL/NCCL 的设备数据通路正常吗？
3. CPU group 在等消息，是否意味着该通信被计入设备 AllReduce 的带宽？

下一讲：[MoE、DeepEP 与 ZBAL](./04-moe-deepep-and-zbal.md)。
