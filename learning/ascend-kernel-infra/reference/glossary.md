# 参考：Ascend Kernel Infra 术语表

本表给出本课程中的工作定义。遇到具体硬件规格和 API 行为时，仍以目标 CANN/Triton-Ascend 版本文档为准。

## A. 框架与运行时

| 术语 | 解释 |
|---|---|
| SGLang | 面向 LLM/多模态模型的高性能 serving 框架，负责请求、batch、模型执行和缓存等编排 |
| sgl-kernel-npu | SGLang 官方 Ascend NPU kernel 仓库/包，混合 Triton、Ascend C、C++ 和通信实现 |
| PyTorch/ATen | 上层 tensor 与算子语义、dispatcher 体系 |
| torch_npu | PyTorch 的 Ascend NPU 设备后端和扩展 |
| Driver / Firmware | 让操作系统、运行时和 NPU 硬件真正连通并可执行任务的底层软件与固件层 |
| CANN | Ascend 基础软件栈，包括编译器、算子库、runtime、通信等 |
| CANN Runtime | CANN 中负责 device、memory、stream、kernel/model launch 的执行层 |
| AscendCL | CANN 中的运行时/应用接口，管理 device、memory、stream、kernel/model execution 等 |
| HCCL | Ascend 多卡集合通信库，角色近似 CUDA 生态的 NCCL |
| Host | 通常指 CPU 进程及 Host 侧代码，准备参数和发起 device 工作 |
| Device | Ascend NPU 及其 device 侧执行环境 |
| Runtime | 加载 kernel、管理 stream/memory 并向设备提交任务的软件层 |
| Dispatcher | PyTorch 根据 operator、device/dtype 等选择 backend 实现的机制 |
| PrivateUse1 | PyTorch 为外部设备 backend 保留的 dispatch key，torch_npu 用于 NPU 接入 |

## B. 算子与编译

| 术语 | 解释 |
|---|---|
| Operator / 算子 | 数学语义加上 shape/dtype/layout、Host、注册等完整软件接口 |
| Kernel | 在 NPU device 上执行的一段计算程序 |
| Custom Op | 框架默认没有、由扩展自行注册实现的算子 |
| Fusion / 融合 | 把多个计算阶段放进更少 kernel，减少 launch 和 GM 中间读写 |
| DSL | 领域专用语言；Triton 是面向并行 kernel 的 Python DSL |
| JIT | Just-In-Time，运行时按参数编译 kernel；Triton 常用此方式 |
| AOT | Ahead-Of-Time，部署前编译；Ascend C shared library 常走此路线 |
| IR | Intermediate Representation，编译器在前端与机器代码之间使用的中间表示 |
| TTIR | Triton IR，Triton 编译链中的核心中间表示之一 |
| Lowering | 把高层语义逐步转换成更接近目标硬件的 IR/指令 |
| Meta-parameter | 控制 tile、stages 等编译策略的参数，常由 `tl.constexpr` 承载 |
| Autotune | 对一组候选 meta-parameter 做测量并选择更优配置 |
| Schema | Custom op 的函数签名、参数、返回值、mutation/alias 契约 |
| Binding | 把 Python/PyTorch 调用连接到 C++/device 实现的接口代码 |
| ACLNN | CANN 提供的一类高层算子接口/算子库入口，常用于直接复用现成 NPU 算子能力 |
| Operator Library / 算子库 | 已由 CANN 预先提供的标准或融合算子实现集合，优先用于复用而不是重复写 kernel |
| Tiling | 根据具体 shape、dtype 和硬件资源计算 blockDim、tile、workspace 等切分参数的过程/协议 |
| Platform | 向编译器和 Host 侧暴露硬件核数、存储层级、架构能力等事实的查询与抽象层 |

## C. 并行编程模型

| 术语 | 解释 |
|---|---|
| SPMD | Single Program, Multiple Data；多个实例执行同一程序但处理不同数据 |
| Program | Triton 的一个并行 kernel 实例，通常处理一个 tile |
| Program ID / pid | 当前 Triton program 在 grid 某个轴上的编号 |
| Grid | 一次 Triton launch 创建的 program instance 逻辑空间，最多三维 |
| BlockDim | Ascend C launch 的核/逻辑实例数量 |
| Block Index | 当前 Ascend C 实例编号，常由 `GetBlockIdx()` 获取 |
| Tile | 从大 tensor 切下、一次在某核上处理的数据块 |
| Block Tensor | Triton program 内的一块 N 维值或指针集合 |
| Persistent Kernel | 让有限 program 持续循环处理多个逻辑 tile，减少超大 grid 调度 |
| Tail / 尾块 | 总长度不能整除 tile 或核数时剩余的数据部分 |
| Mask | 标记 block tensor 中哪些 lane 的 load/store/compute 有效 |
| Stride | 某维索引增加 1 时，线性内存地址跨过的元素数 |
| Layout/Format | Tensor 元素在物理内存中的组织方式，如 ND、NZ |

## D. Ascend 硬件

| 术语 | 解释 |
|---|---|
| AI Core | Ascend NPU 中执行矩阵、向量密集计算的核心资源/架构抽象 |
| Cube | 面向矩阵乘加的高吞吐计算单元 |
| Vector | 面向逐元素、归约、数学函数等向量计算的单元 |
| Scalar | 核内负责地址、循环、分支、参数和指令发射的控制计算单元 |
| Cube Core / AIC | 分离模式下专注矩阵计算的核 |
| Vector Core / AIV | 分离模式下专注向量计算的核 |
| MTE | Memory Transfer Engine，负责不同存储层级的数据搬运 |
| FixPipe | Cube 输出等数据通路与随路转换相关单元，具体能力依架构 |
| GM | Global Memory，核外全局设备内存的逻辑称呼 |
| HBM | 设备上的高带宽外部内存；在本课程中常作为 GM 的物理背景理解 |
| L2 Cache | 多核共享的 GM 访问缓存 |
| L1 Buffer | 较大的片上中转/复用存储，常服务 Cube 数据 |
| L0A/L0B | Cube A/B 输入操作数的近端存储 |
| L0C | Cube 累加结果存储 |
| UB | Unified Buffer，Vector 输入输出和临时数据的主要片上存储 |

## E. Ascend C 数据与资源

| 术语 | 解释 |
|---|---|
| GM_ADDR | Ascend C kernel 的 Global Memory 地址参数类型 |
| GlobalTensor | 对 GM 数据的类型化 device 侧视图，不代表数据已搬入片上 |
| LocalTensor | 位于片上 Local Memory 的 tensor 抽象，供 Vector/Cube API 使用 |
| TPosition | LocalTensor/Queue 的逻辑存储位置，如 VECIN、A1、A2 |
| TPipe | 管理 Pipe/Queue 范式中的片上 buffer 与 event 资源 |
| TQue | 管理 LocalTensor buffer，并表达流水生产者/消费者同步 |
| TBuf | 不需要通过 queue 跨阶段传递的片上临时 buffer |
| AllocTensor | 从 queue/buffer 取得一块 LocalTensor 资源 |
| FreeTensor | 释放 LocalTensor 对应资源以便复用 |
| EnQue | 生产者把已就绪 LocalTensor 放入队列 |
| DeQue | 消费者取得已就绪 LocalTensor |
| DataCopy | 在 Global/Local 或不同 Local 位置之间搬运数据的 API 类别 |
| Tiling Data | Host 计算并传给 Device 的切分参数协议 |
| Workspace | 算子运行所需的额外临时全局内存 |

## F. 流水与同步

| 术语 | 解释 |
|---|---|
| CopyIn | 将当前 tile 从 GM 搬入 Local Memory |
| Compute | 使用 Vector/Cube 等单元处理 LocalTensor |
| CopyOut | 将结果从 Local Memory 搬回 GM |
| Pipeline / 流水 | 让不同 tile 的搬入、计算、搬出阶段在不同硬件通路重叠 |
| Double Buffer | 使用 ping/pong 两组 buffer，让下一 tile 搬入与当前 tile 计算重叠 |
| Queue Depth | 同一 TQue 可连续入队而未出队的次数，不等于 buffer number |
| Event | 表达异步指令通路之间依赖的同步资源 |
| Barrier | 让指定范围内多个执行者都到达某点后再继续的同步机制 |
| Stream | Host 向 device 提交异步任务的有序队列 |
| Record Stream | 告知 allocator 某 tensor storage 正被异步 stream 使用，避免提前回收 |
| Cross-core Sync | 多核之间的数据就绪或阶段同步 |
| CV Fusion | Cube 与 Vector 阶段在同一融合算子内协作执行 |

## G. 性能与正确性

| 术语 | 解释 |
|---|---|
| Alignment / 对齐 | 地址或搬运长度满足硬件粒度要求，如 32B 对齐 |
| Padding | 补充无效元素使 shape/字节满足对齐或基本块要求 |
| Arithmetic Intensity | 每搬运一个字节完成多少计算，用于判断计算或带宽倾向 |
| Memory-bound | 性能主要受数据搬运带宽/延迟限制 |
| Compute-bound | 性能主要受计算单元吞吐限制 |
| Occupancy/利用率 | 物理计算资源有多少时间在做有效工作；不同工具定义可能不同 |
| Load Balance | 多核工作量是否均匀 |
| Warmup | 正式计时前运行若干次，排除 JIT、缓存和初始化影响 |
| Microbenchmark | 只测单算子/kernel 的性能测试 |
| End-to-end Benchmark | 在 SGLang 请求与模型执行场景下测整体性能 |
| Reference | 简单独立的正确实现，用来比较 custom kernel 输出 |
| Numerical Tolerance | 浮点比较允许的 `atol/rtol` 误差范围 |
| Profiling / Trace | 记录 Host、launch、kernel、搬运和等待时间线以定位瓶颈 |
