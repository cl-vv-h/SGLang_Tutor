# AI Infra 通信：从张量语义到 GPU/NPU 数据路径

**简体中文** | [English](../../../en/ai-infra-basic/Communication/README.md)

多卡推理的通信至少包含三件不同的事：TP 合并局部计算结果，MoE 把 token 送到专家再送回来，PD 分离把 KV 交给另一个实例。先分清传什么、传给谁、何时可用，再选通信库。

本专题接在 [Parallel Strategy](../Parallel_Strategy/README.md) 后面，和 [KV Transfer](../KV_Transfer/README.md) 配合阅读。原有 Ascend 源码课继续负责具体实现；这里补充跨平台的概念、公式、比较和实验方法。

## 阅读顺序

| 文件 | 学完要能回答 |
|---|---|
| [01-communication-primitives-and-costs.md](./01-communication-primitives-and-costs.md) | AllReduce、AllGather、ReduceScatter、AllToAllV 到底改变了什么？怎样估算字节量和延迟？ |
| [02-links-rdma-and-memory.md](./02-links-rdma-and-memory.md) | NVLink/HCCS、PCIe、IB/RoCE、RDMA、注册内存与完成通知分别在哪一层？ |
| [03-collective-libraries-gloo-nccl-hccl.md](./03-collective-libraries-gloo-nccl-hccl.md) | Gloo、NCCL、RCCL、HCCL 与 `torch.distributed` 的关系是什么？ |
| [04-moe-deepep-and-zbal.md](./04-moe-deepep-and-zbal.md) | DeepEP dispatch/combine 做了哪些事？SGLang NPU 的 ZBAL 怎样接入？ |
| [05-kv-transfer-mooncake-and-nixl.md](./05-kv-transfer-mooncake-and-nixl.md) | Mooncake TE、Store、EP/PG、NIXL、Ascend TransferEngine 如何区分？ |
| [06-sglang-paths-and-performance.md](./06-sglang-paths-and-performance.md) | 如何从模型调用追到通信实现，再定位慢 rank、链路和重叠问题？ |
| [07-zbal-zero-buffer-source-dataflow.md](./07-zbal-zero-buffer-source-dataflow.md) | 第 04 讲的源码进阶：GVA/分配器、zero buffer、省掉与保留的复制、HCCL 委托、MoE 实际地址数据流 |
| [communication_semantics_demo.py](./communication_semantics_demo.py) | 用纯 Python 验证规约、拼接、分片与 MoE 回排，不需要加速卡 |

## 技术地图

这些名字不是同一层的互斥选项。表中“作用”是本课程关注的主要用途，不是项目全部能力。

| 层次 / 用途 | 技术 | 首先记住 |
|---|---|---|
| 框架接口 | `torch.distributed`、ProcessGroup | 定义 rank、通信组、collective 与异步工作句柄 |
| CPU collective / 协调 | Gloo | CPU 张量通信；SGLang 普通分支也用它建立 CPU 协调组 |
| 加速器 collective | NCCL / RCCL / HCCL | 分别面向 NVIDIA GPU、AMD GPU、Ascend NPU |
| 专用 MoE 通信 | DeepEP、Ascend DeepEP 适配 | 路由、dispatch、expert 布局、combine 的联合优化 |
| NPU 内存与通信 | ZBAL / MemFabric | 分配器与通信算子配合；SGLang 同时有 backend 与 DeepEP adapter 接入 |
| 异步数据传输 | Mooncake Transfer Engine、NIXL | 传已注册内存中的数据块，例如 KV、权重或 embedding |
| 分布式缓存 | Mooncake Store | 在传输之上管理对象、位置和缓存生命周期 |
| 底层编程工具 | UCX、NVSHMEM、MSCCL++ | 传输抽象、单边访问、设备端通信等不同工具 |
| 物理链路 / 网络 | PCIe、NVLink/NVSwitch、HCCS、InfiniBand、RoCE | 决定拓扑、可达性、延迟与可用带宽 |

例如，同一个服务可以用 Gloo 协调 CPU 进程，用 HCCL 做 TP collective，用 NPU DeepEP/ZBAL 做 MoE dispatch，再用 MemFabric TransferEngine 传 PD 的 KV。实际是否组合、怎样组合，必须看所安装版本和调用分支。

## 贯穿教程的三个问题

1. **语义**：这是求和、拼接、路由重排，还是远端内存复制？
2. **完成条件**：Python 函数返回、设备操作完成、消费者可读、请求可调度，分别由谁保证？
3. **成本**：通信组有多少 rank？每个 rank 多少有效字节？是否包含 padding、量化元信息和等待慢 rank？

## 运行与阅读约定

在仓库根目录执行：

```bash
python3 learning/zh/ai-infra-basic/Communication/communication_semantics_demo.py
```

demo 是单进程语义模拟和成本估算，**不测量真实网络性能**。第 03 讲另有可保存运行的两进程 Gloo 示例，需要安装支持 distributed/Gloo 的 PyTorch。

资料核对日期：**2026-10-13**。SGLang 调用路径以本仓库 `f249f7b` 源码为基线，文内相对源码链接便于随仓库阅读。上游通信项目迭代快，尤其 DeepEP 的 `Buffer/Config` 与新版 `EPBuffer` 不应混用；硬件矩阵和安装版本以链接的官方说明为准。

ZBAL 专项补充于 **2026-10-14** 核对：建议读完第 04 讲后接着读第 07 讲，跟随固定 MemFabric commit 查看实际 allocator、ProcessGroup 和 Ascend C kernel。

## 后续源码课

- [LayerCommunicator、HCCL 与层边界通信](../../sglang-ascend-npu/source-code-walkthrough/15-distributed-hccl-and-communication.md)
- [sgl-kernel-npu：DeepEP、HCCL 与 MoE token 路径](../../ascend-kernel-infra/sgl-kernel-npu/05-deepep-hccl-and-moe-kernel-path.md)
- [DeepEP low-latency 与 A2 分层路径](../../ascend-kernel-infra/sgl-kernel-npu/06-deepep-low-latency-and-layered-a2-path.md)

各讲在具体结论附近列出官方文档或源码依据。建议先读第 01 讲，再按 TP、MoE、PD 的实际问题选择后续章节。
