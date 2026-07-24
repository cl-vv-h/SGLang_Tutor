**中文** | [English](./glossary_EN.md)

# 参考：Ascend Kernel Infra 术语表

本表给出本课程中的工作定义。遇到具体硬件规格和 API 行为时，仍以目标 CANN/Triton-Ascend 版本文档为准。

若一个术语涉及具体变量类型、shape 或 pointer arithmetic，请同时查[代码阅读手册](./code-reading-and-types.md)。

## A. 框架与运行时

| 术语 | 解释 |
|---|---|
| SGLang | 面向 LLM/多模态模型的高性能 serving 框架，负责请求、batch、模型执行和缓存等编排 |
| sgl-kernel-npu | SGLang 官方 Ascend NPU kernel 仓库/包，混合 Triton、Ascend C、C++ 和通信实现 |
| DeepEP-Ascend | `sgl-kernel-npu` 里的 MoE 专用通信子模块，站在 HCCL 之上，把 token 路由、dispatch、combine 和部分 fused expert 计算封装成统一接口 |
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
| Cache Manager | Triton runtime 中按哈希保存编译产物、launcher stub 或辅助 `.so` 的缓存管理器；命中后可直接复用，不必重新编译 |
| Dump Manager | Triton runtime 中负责把调试用 IR、launcher 源码或二进制写到 dump 目录的管理器 |
| Dispatcher | PyTorch 根据 operator、device/dtype 等选择 backend 实现的机制 |
| Dispatch Key | PyTorch dispatcher 用来选择后端实现的路由标签；Ascend NPU 通过 `torch_npu` 接入时常见内部标签是 `PrivateUse1` |
| Host-side Dispatch | 本课程中指 C++ Host wrapper 在被 PyTorch dispatcher 调到以后，继续按 dtype、shape、硬件资源、workspace 等条件选择具体 kernel 变体和 launch 参数；它不是独立进程调度 |
| PrivateUse1 | PyTorch 为外部设备 backend 保留的 dispatch key，torch_npu 用于 NPU 接入 |
| OpCommand | `torch_npu` framework 层用于打包一次 NPU 算子调用、stream 和 custom handler 的命令对象 |

## B. 算子与编译

| 术语 | 解释 |
|---|---|
| Operator / 算子 | 数学语义加上 shape/dtype/layout、Host、注册等完整软件接口 |
| Kernel | 在 NPU device 上执行的一段计算程序 |
| Custom Op | 框架默认没有、由扩展自行注册实现的算子 |
| Wrapper / 包装层 | 高层调用与底层 kernel contract 之间的边界适配器；通常负责检查 shape/dtype/stride/device，分配输出或 workspace，计算 grid/blockDim/tiling，选择后端，并调用 Triton kernel、`torch.ops`、ACLNN 或 Ascend C launch |
| Python Wrapper | 运行在 Python/Host 侧的 wrapper，接收 `torch.Tensor`、Python 标量或 module 对象，常见职责是输出分配、dtype/layout 整理、grid 计算、后端分流和调用 `kernel[grid]` 或 `torch.ops.*` |
| Host Wrapper | 运行在 C++ Host 侧的 wrapper，接收 dispatcher 传来的 `at::Tensor` 与标量，常见职责是 `TORCH_CHECK`、workspace/stream/tiling 准备，并最终调用 `EXEC_KERNEL_CMD`、`EXEC_NPU_CMD` 或 ACLNN |
| Fusion / 融合 | 把多个计算阶段放进更少 kernel，减少 launch 和 GM 中间读写 |
| Mega Kernel | 一种更激进的融合形态：把多个原本可拆成独立 launch 的算法阶段放进同一次 device kernel launch 中执行，以减少 Host/runtime 边界并增强 device 内部调度控制；它不等于所有中间结果都留在片上，也不等于 persistent kernel |
| Device Stage | 同一个 device kernel 内部具有明确输入、输出和依赖边界的阶段；它不是独立 PyTorch operator，也不是单独 launch，而是 mega kernel 内的一段设备侧工作 |
| MoE / Mixture-of-Experts | 先让 router 为每个 token 选 top-k 个 expert，再把 token 发给这些 expert 计算的模型结构；和 dense FFN 不同，它天然包含 token 分流与回流 |
| FLA / Flash Linear Attention | 一类把长序列注意力改写成“分块 + 状态递推”形式的算法/实现家族，避免显式构造完整 `T x T` attention 矩阵 |
| Gated Delta Rule | FLA 家族中的一种更新规则，用 `g` 控制遗忘/衰减，用 `beta` 控制当前 token 对状态的写入强度 |
| GDN | `sgl-kernel-npu` 当前源码中用于 FLA chunk gated delta rule 相关 mega op 的内部命名缩写，例如 `mega_chunk_gdn`、`GDN_D`、`GDN_C`；本课程不把它当作独立框架名 |
| PTO | 当前 `sgl-kernel-npu` 源码通过 `<pto/pto-inst.hpp>` 使用的 device 侧 tile/instruction helper 层，可先理解成更贴近 Ascend 硬件通路的低层 DSL/helper；本课程只按源码使用方式解释，不承诺其跨版本 API 稳定性 |
| KKT / `kkt` stage | 在 `mega_chunk_gdn` 中指 chunk 内 `K` 与 `K^T` 相关的 scaled dot/Gram-like 矩阵阶段，不是优化理论里的 Karush-Kuhn-Tucker 条件 |
| `solve_tril` | 下三角矩阵求解/求逆相关阶段；`tril` 表示 lower triangular，下三角 |
| WY / `wy_fast` | FLA chunk 递推中的紧凑中间因子构造阶段；在当前源码中主要把 `A_inv`、`k/v/beta/g` 合成后续 `chunk_h` 可复用的 `w/u` 张量 |
| DSL | 领域专用语言；Triton 是面向并行 kernel 的 Python DSL |
| JIT | Just-In-Time，运行时按参数编译 kernel；Triton 常用此方式 |
| AOT | Ahead-Of-Time，部署前编译；Ascend C shared library 常走此路线 |
| IR | Intermediate Representation，编译器在前端与机器代码之间使用的中间表示 |
| TTIR | Triton IR，Triton 编译链中的核心中间表示之一 |
| `ttadapter` / `kernel.ttadapter.mlir` | Triton-Ascend 源码里给“TTIR 经过一轮 Ascend 后端 pass 后的文本中间产物”起的文件名；它更像阶段标签，不是另一门独立语言 |
| Linalg IR | MLIR 中表达张量、循环和线性代数操作的一层 IR；Triton-Ascend 会把大量 Triton 专属操作降到这里，再交给 BiSheng 等后端工具 |
| MLIR Bytecode / `mlirbc` | MLIR 的二进制序列化格式；语义上和文本 `.mlir` 对应，只是更紧凑、更适合在工具链之间传递 |
| LLIR | 比 Linalg IR 更靠近后端代码生成的一层低级中间表示；在本课程里主要把它当成“进入最终 binary 生成前的最后几层 IR” |
| Lowering | 把高层语义逐步转换成更接近目标硬件的 IR/指令 |
| Meta-parameter | 控制 tile、stages 等编译策略的参数，常由 `tl.constexpr` 承载 |
| Autotune | 对一组候选 meta-parameter 做测量并选择更优配置 |
| Schema | Custom op 的函数签名、参数、返回值、mutation/alias 契约 |
| Binding | 把 Python/PyTorch 调用连接到 C++/device 实现的接口代码 |
| ACLNN | CANN 提供的一类高层算子接口/算子库入口，常用于直接复用现成 NPU 算子能力 |
| aclOpExecutor | ACLNN 两段式接口返回的执行句柄，封装本次算子的计算流程，后续与 workspace、stream 一起提交执行 |
| Operator Library / 算子库 | 已由 CANN 预先提供的标准或融合算子实现集合，优先用于复用而不是重复写 kernel |
| Tiling | 根据具体 shape、dtype 和硬件资源计算 blockDim、tile、workspace 等切分参数的过程/协议 |
| Tiling Strategy / Tiling 策略 | 开发者或算子库作者写在 Host 侧的切分规则；运行时 Host 把本次 shape、dtype、layout 与 Platform 查询结果代入策略，得到具体 `blockDim/tileLength/workspace/tiling key` 等参数 |
| Platform | 向编译器和 Host 侧暴露硬件核数、存储层级、架构能力等事实的查询与抽象层 |
| PlatformAscendCManager | Ascend C Host 侧常见的平台查询单例接口，可读取 AIV/AIC 核数、UB/L1/L0 容量和库侧 workspace 需求等信息 |
| Tiling Key | Host 编码出的 kernel 变体选择值，用来区分不同 transpose、dtype、format、split-K 等实现路径，不等于 tile 大小本身 |

## C. 并行编程模型

| 术语 | 解释 |
|---|---|
| SPMD | Single Program, Multiple Data；多个实例执行同一程序但处理不同数据 |
| Expert Parallel / EP | 按 expert 维度切分模型并把不同 expert 分布到不同卡上的并行方式；它和按张量维度切分的 TP 不同 |
| Top-k Routing | router 为每个 token 选出 top-k 个 expert 的路由过程；输出通常是 `topk_idx` 和 `topk_weights` |
| Dispatch | 把 token 按 top-k 路由结果统计、重排并发送到正确 expert 所在卡与本地缓冲区的过程，不等于单纯 `all_to_all` |
| Combine | expert 计算结束后，把结果按原 token 顺序回排并按 top-k 权重聚合的过程，是 dispatch 的回程 |
| Low-Latency Mode | DeepEP 为小 batch 推理准备的另一套 dispatch/combine 协议；目标更偏向固定小批次延迟，而不是 normal mode 那种大吞吐 prefill/training 路径 |
| `num_max_dispatch_tokens_per_rank` / 对齐 token 上界 | low-latency 路径按“各 rank 本轮 token 数的最大值”预留缓冲和协议边界；它不是当前 rank 的真实有效 token 数 |
| Packed Receive Buffer / `packed_recv_x` | low-latency dispatch 输出给本地 expert 计算的紧凑接收 buffer；顺序按 expert-friendly 布局组织，不等于原 token 顺序 |
| Program | Triton 的一个并行 kernel 实例，通常处理一个 tile |
| Program ID / pid | 当前 Triton program 在 grid 某个轴上的编号 |
| Grid | 一次 Triton launch 创建的 program instance 逻辑空间，最多三维；逻辑 grid 不天然等于设备同时并行的物理核数 |
| BlockDim | Ascend C launch 的核/逻辑实例数量 |
| Block Index | 当前 Ascend C 实例编号，常由 `GetBlockIdx()` 获取 |
| Tile | 从大 tensor 切下、一次在某核上处理的数据块 |
| Logical Tile / Logical Block | 从数学输出空间切出来的一份逻辑工作单元；它说明“总共有多少块工作”，不等于“这次真的启动了多少物理核” |
| `num_matrices` | `mega_chunk_gdn` 里表示本次 device 侧需要处理的逻辑矩阵数量，当前 Python wrapper 计算为 `num_chunks * num_value_heads`；它不是 token 数，也不是物理 core 数 |
| `H` / NumValueHeads | 本课程在 FLA mega kernel 上下文中常用 `H` 表示 value heads 数，即 `v/g/beta` 使用的 head 数 |
| `Hg` / NumKeyHeads | 本课程在 FLA mega kernel 上下文中常用 `Hg` 表示 query/key heads 数，即 `q/k` 使用的 head 数；GQA 场景下 `H` 和 `Hg` 可以不同 |
| Block Tensor | Triton program 内的一块 N 维值或指针集合 |
| `tl.tensor` | Triton Python 前端表示 IR value 的核心类，保存 IR handle、完整 type、静态 shape 与标量 dtype；它不是 `torch.Tensor` 数据容器 |
| `tl.constexpr` | JIT 编译时已知的值，用来决定 block shape、分支、循环展开与 kernel specialization；不同取值可能产生不同缓存变体 |
| Value Block | 元素为数值的 Triton block tensor，例如 `fp16[128]`；通常由 `tl.load`、算术或 `tl.zeros` 产生 |
| Pointer Block | 元素为地址的 Triton block tensor，例如 `pointer<fp16>[16,32]`；表示一块地址网格，构造它不等于读取内存 |
| Pointer Type | Triton 中保存 pointee `element_ty` 与 address space 的标量类型；合法 pointer arithmetic 是 pointer 加整数元素 offset，pointer 不能与 pointer 相加 |
| Element Offset | 以元素而非字节计数的地址偏移；Triton pointer addition、PyTorch stride 与 Ascend C `GlobalTensor[index]` 常使用这种单位 |
| Byte Size | 以字节计数的容量；Ascend C `InitBuffer`、workspace、UB budget 常使用此单位，通常需要 `elements * sizeof(T)` |
| Broadcast / 广播 | 把 scalar 或含长度 1 维度的 block 扩展到兼容 shape；Triton 二元运算会在 semantic 层完成兼容性检查并生成 broadcast/splat IR |
| Persistent Kernel | 让有限 program 持续循环处理多个逻辑 tile，减少超大 grid 调度 |
| Auto-blockify | Triton-Ascend 的大 grid 优化机制：编译期把 kernel 包进内层循环，运行期把 launch block 数钳到物理核数，用较少物理 block 覆盖更多逻辑 block |
| Tail / 尾块 | 总长度不能整除 tile 或核数时剩余的数据部分 |
| Mask | 标记 block tensor 中哪些 lane 的 load/store/compute 有效 |
| Stride | 某维索引增加 1 时，线性内存地址跨过的元素数 |
| Indices / 索引张量 | 保存位置编号的整型张量；具体可能是 token id、batch 行号、cache slot、expert id、top-k 返回下标等，必须结合 shape 和 gather/scatter 维度判断。在 `apply_token_bitmask` 中特指 batch row indices |
| Token ID / Vocabulary Index | token 在词表中的整数编号，范围通常是 `[0, vocab_size)`；在 logits 中常对应 vocab 维度的位置，不等于 batch 行号 |
| Layout/Format | Tensor 元素在物理内存中的组织方式，如 ND、NZ |
| Packed B=1 Layout | 把多条样本沿 token 维拼成一条长序列，对外保留 `B=1` 壳子，再用 `cu_seqlens` 恢复每条样本边界的布局约定 |
| `cu_seqlens` | cumulative sequence lengths，记录 packed 变长输入每条样本起止位置的前缀和数组 |

## D. Ascend 硬件

| 术语 | 解释 |
|---|---|
| AI Core | Ascend NPU 中执行矩阵、向量密集计算的核心资源/架构抽象 |
| Cube | 面向矩阵乘加的高吞吐计算单元 |
| Vector | 面向逐元素、归约、数学函数等向量计算的单元 |
| Scalar | 核内负责地址、循环、分支、参数和指令发射的控制计算单元 |
| Cube Core / AIC | 分离模式下专注矩阵计算的核 |
| Vector Core / AIV | 分离模式下专注向量计算的核 |
| Physical Core Count | 当前设备该类 kernel 可并行使用的真实物理核数量；对 Cube 主导 kernel 常看 `num_aicore`，对纯 Vector kernel 常看 `num_vectorcore` |
| MTE | Memory Transfer Engine，负责不同存储层级的数据搬运 |
| FixPipe | Cube 输出等数据通路与随路转换相关单元，具体能力依架构 |
| GM | Global Memory，核外全局设备内存的逻辑称呼 |
| HBM | 设备上的高带宽外部内存；在本课程中常作为 GM 的物理背景理解 |
| L2 Cache | 多核共享的 GM 访问缓存 |
| L1 Buffer | 较大的片上中转/复用存储，常服务 Cube 数据 |
| A1/B1 | Ascend C `TPosition` 中常见的 Cube A/B 操作数在 L1 阶段的逻辑位置；它描述数据角色，不应被硬编码理解为所有架构上的固定物理分区 |
| L1A/L1B | 资料或口语中有时用来描述 L1 中服务 A/B 操作数的区域或角色；不同硬件映射可能不同，初学时优先用 A1/B1 这种逻辑位置理解 |
| L0A/L0B | Cube A/B 输入操作数的近端存储 |
| A2/B2 | Ascend C `TPosition` 中常见的 Cube A/B 操作数在 L0 阶段的逻辑位置，通常对应 L0A/L0B 角色 |
| L0C | Cube 累加结果存储 |
| CO1/CO2 | Cube 输出或累加结果相关的逻辑位置/阶段命名，常与 L0C、输出格式转换和写回路径相关；具体含义需看目标架构和 API 文档 |
| UB | Unified Buffer，Vector 输入输出和临时数据的主要片上存储 |

## E. Ascend C 数据与资源

| 术语 | 解释 |
|---|---|
| GM_ADDR | Ascend C kernel 的 Global Memory 地址参数类型 |
| GlobalTensor | 对 GM 数据的类型化 device 侧视图，不代表数据已搬入片上 |
| LocalTensor | 位于片上 Local Memory 的 tensor 抽象，供 Vector/Cube API 使用 |
| Typed View / 类型化视图 | 保存地址、元素类型和可访问区域的轻量对象；`SetGlobalBuffer`、`operator[]` 或 `AllocTensor` 取得 view 不等于已经执行 DataCopy |
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
| Tiling Tensor | Host 把 tiling 结构体序列化后复制到 device 的 byte tensor，用于让 Device kernel 按约定 ABI 读取本次执行计划 |
| Workspace | 算子运行所需的额外临时全局内存 |
| Lib API Workspace | 由 `GetLibApiWorkSpaceSize()` 之类接口返回的库或 launch 框架所需 device scratch 大小，和算法自己的业务输入输出区分开 |
| Recurrent State / Final State | chunked 线性注意力或递推算子在 chunk 之间传递的历史摘要；`initial_state` 是输入状态，`final_state` 是本轮处理后输出给下一轮的状态 |
| `mask_lower` | 不含对角线的下三角 mask，常用于表示严格过去依赖，避免当前位置以同一种方式参与自身更新 |
| `mask_full` | 含对角线的下三角 mask，常用于输出阶段允许当前位置看到自身的因果结构 |
| `minus_identity` | 对角线为 `-1` 的单位阵变体；在 `mega_chunk_gdn` 中作为三角求解/逆相关 stage 的辅助矩阵传入 device |

## F. 流水与同步

| 术语 | 解释 |
|---|---|
| CopyIn | 将当前 tile 从 GM 搬入 Local Memory |
| Compute | 使用 Vector/Cube 等单元处理 LocalTensor |
| CopyOut | 将结果从 Local Memory 搬回 GM |
| All-to-all / AllToAllV | 各 rank 彼此交换不同数量数据包的通信模式；MoE token 在 expert parallel 下常用它跨卡分发与回收 |
| A2 Layered Path | 910B/A2 上按“同机 HCCS + 跨机 RDMA”分层组织的 DeepEP 通信路径；它需要比普通 per-rank count 更丰富的中间路由辅助张量 |
| Pipeline / 流水 | 让不同 tile 的搬入、计算、搬出阶段在不同硬件通路重叠 |
| Double Buffer | 使用 ping/pong 两组 buffer，让下一 tile 搬入与当前 tile 计算重叠 |
| Queue Depth | 同一 TQue 可连续入队而未出队的次数，不等于 buffer number |
| Event | 表达异步指令通路之间依赖的同步资源 |
| Barrier | 让指定范围内多个执行者都到达某点后再继续的同步机制 |
| Stream | Host 向 device 提交异步任务的有序队列 |
| Task Queue | Triton-Ascend launcher/runtime 的异步提交模式；开启后 Host 提交 launch 后可先返回，但这不等于 device kernel 内部变成 persistent 调度 |
| Record Stream | 告知 allocator 某 tensor storage 正被异步 stream 使用，避免提前回收 |
| Cross-core Sync | 多核之间的数据就绪或阶段同步 |
| CV Fusion | Cube 与 Vector 阶段在同一融合算子内协作执行 |
| `TLOAD` | PTO 风格 device helper 中常见的搬入动作，可理解为把 GM 或较低层数据搬进片上 tile；类似 CopyIn，但参数携带 PTO tile/layout 语义 |
| `TSTORE` | PTO 风格 device helper 中常见的写回动作，可理解为把片上 tile 结果写回 GM；类似 CopyOut |
| `TASSIGN` | PTO 风格 device helper 中把 tile view 绑定到某段片上 buffer 或逻辑地址的动作；它不是普通数学赋值 |
| `TTRANS` | PTO 风格 device helper 中的 tile transpose 动作，用于在片上改变 tile 行列布局 |
| `pipe_barrier` | Ascend device 侧 pipeline barrier，用于确保当前 core 内指定指令通路到达安全点后再继续 |
| `set_flag` / `wait_flag` | 同一 core 内不同通路之间的事件同步，用于表达搬运、计算、写回等生产消费依赖 |
| `wait_flag_dev` | device 侧跨 core 或 AIV/AIC 协作等待信号，用于保证多个执行者在阶段边界上对齐 |

## G. 性能与正确性

| 术语 | 解释 |
|---|---|
| Alignment / 对齐 | 地址或搬运长度满足硬件粒度要求，如 32B 对齐 |
| Launcher Stub | Host 侧临时生成的小段 C++/Python 扩展胶水，用来把 Triton/PyTorch 调用参数整理成底层 runtime 能执行的 launch 形式 |
| `aclrtlaunch_*` Stub | Ascend C 构建链路为某个 device kernel 生成或声明的 Host 侧 launch 入口，通常接收 `blockDim`、`aclrtStream`、设备地址和标量参数，再把 launch 请求交给 CANN Runtime |
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
| RDMA Size Hint / `get_low_latency_rdma_size_hint` | DeepEP 暴露的 low-latency buffer 预留提示接口；在当前 `d5630df` 实现里它直接返回 `num_max_dispatch_tokens_per_rank`，不要仅凭名字脑补 byte 计算公式 |
