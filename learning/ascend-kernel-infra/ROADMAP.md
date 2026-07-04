# Ascend Kernel Infra 路线图

本文件只维护课程主线、前置关系和待深化主题，不记录按日期的流水账。

## 当前主线

| 主题 | 前置知识 | 当前状态 | 对应文件 |
|---|---|---|---|
| 软件栈关系：先分清 SGLang、`sgl-kernel-npu`、`torch_npu`、Triton-Ascend、Ascend C | 无 | 已完成 | [`01-stack-and-relationships.md`](./01-stack-and-relationships.md) |
| CANN 全栈与边界：Driver/Firmware、Runtime、AscendCL、算子库、编译器、Tiling、Platform、HCCL | 01 | 已完成 | [`02-cann-stack-and-boundaries.md`](./02-cann-stack-and-boundaries.md) |
| Kernel 第一性原理：operator、kernel、Host/Device、SPMD、grid、tile | 01-02 | 已完成 | [`foundations/01-kernel-first-principles.md`](./foundations/01-kernel-first-principles.md) |
| Ascend 硬件与存储层级：AI Core、AIC/AIV、Cube、Vector、GM/L0/L1/UB | foundations/01 | 已完成 | [`foundations/02-ascend-hardware.md`](./foundations/02-ascend-hardware.md) |
| 搬运、计算、同步与流水：TPipe、TQue、double buffer、算术强度 | foundations/01-02 | 已完成 | [`foundations/03-memory-pipeline-and-sync.md`](./foundations/03-memory-pipeline-and-sync.md) |
| Triton-Ascend 基础到调优 | 01-02，foundations/01-03 | 已完成 | [`triton-ascend/`](./triton-ascend/) |
| Ascend C 基础到调优 | 01-02，foundations/01-03 | 已完成 | [`ascend-c/`](./ascend-c/) |
| `torch_npu`、ACLNN 与 custom op 注册边界 | 01-02，`sgl-kernel-npu/01`，Ascend C 02 | 已完成 | [`torch_npu/01-dispatch-aclnn-and-custom-op-boundaries.md`](./torch_npu/01-dispatch-aclnn-and-custom-op-boundaries.md) |
| `sgl-kernel-npu` 工程入口与两个真实算子 | 01-02，Triton-Ascend 01-03，Ascend C 01-03 | 已完成 | [`sgl-kernel-npu/`](./sgl-kernel-npu/) |

## 下一优先级

| 待深化主题 | 前置知识 | 当前状态 | 对应文件 |
|---|---|---|---|
| Platform 与 Tiling 协议专项：硬件查询、tiling data、workspace、blockDim 的统一读法 | 02，foundations/02-03，Ascend C 02 | 待写 | 计划新增：`ascend-c/04-platform-tiling-and-workspace-contracts.md` |
| Triton-Ascend 编译产物专项：TTIR、Linalg IR、AscendNPU IR、driver、cache | 02，Triton-Ascend 01-03 | 待写 | 计划新增：`triton-ascend/04-ttir-mlir-driver-and-cache.md` |
| `sgl-kernel-npu` 再精读一个“现成算子 + custom glue”混合路径案例 | `sgl-kernel-npu/01`，02 | 待选题 | 计划新增：`sgl-kernel-npu/04-*.md` |
| DeepEP / MoE / HCCL kernel 路径 | 02，`sgl-kernel-npu/01`，分布式与通信基础 | 待写 | 计划新增：`sgl-kernel-npu/05-deepep-hccl-and-moe-kernel-path.md` |

## 维护规则

- 新增章节前，先检查它是否填补了主线中的真实缺口，而不是重复已有总结。
- 如果某一主题开始膨胀到同时覆盖多个新概念，就拆章，不把所有细节堆进一篇。
- 每次补新章节时，都要更新本文件中的“前置知识、当前状态、对应文件”。
