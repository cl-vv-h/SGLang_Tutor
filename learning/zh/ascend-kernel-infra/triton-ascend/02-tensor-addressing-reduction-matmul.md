# Triton-Ascend 02：地址、广播、归约与矩阵分块

Vector Add 只处理一维连续数据。本章把相同模型扩展到二维 tensor、RMSNorm 和矩阵乘，掌握读 `sgl-kernel-npu` Triton 源码所需的核心语法。

本章沿用[代码阅读手册](../reference/code-reading-and-types.md)的记法：`int32[BM]` 是整数 value block，`pointer<fp16>[BM,BN]` 是地址 block。二者都是编译期前端类 `tl.tensor`，区别在其 `dtype` 和静态 `shape`。

## 1. 二维地址来自 Shape 与 Stride

对 `X[M,N]`，元素 `(i,j)` 的线性地址是：

```text
element_offset(i, j) = i * stride_xm + j * stride_xn
element_pointer(i, j) = x_ptr + element_offset(i, j)
```

这里用 `text` 表示数学寻址关系，而不冒充一段变量齐全的 Python 程序。`i/j/stride_xm/stride_xn` 都是整数；`element_offset` 的单位是元素；`x_ptr` 是 `pointer<x_dtype>`。pointer 加整数 offset 生成新地址，不读取数据。

Contiguous 行主序通常有：

```text
stride_xm = N
stride_xn = 1
```

Transpose view 可能 shape 相同但 stride 不同。因此 wrapper 要么把 stride 传入 kernel，要么明确要求 contiguous。

## 2. 用 `[:, None]` 和 `[None, :]` 构造地址矩阵

下面是变量完整声明的 Triton kernel。它把输入二维 tile 原样复制到输出，用来单独观察寻址与类型传播：

```python
import triton
import triton.language as tl


@triton.jit
def copy_2d_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    values = tl.load(x_ptrs, mask=mask, other=0.0)
    tl.store(out_ptrs, values, mask=mask)
```

形状变化：

```text
offs_m[:, None] -> [BLOCK_M, 1]
offs_n[None, :] -> [1, BLOCK_N]
broadcast result -> [BLOCK_M, BLOCK_N]
```

这不是创建一个 Python 嵌套列表，而是在 Triton IR 中表达一个 block tensor 地址网格。假设输入是 FP16、`BLOCK_M=16`、`BLOCK_N=32`，每个名字的类型是：

| 变量 | 类型 | shape | 本质 |
|---|---|---:|---|
| `x_ptr/out_ptr` | `tl.tensor<pointer<fp16>>` | `[]` | 标量设备指针 IR value |
| `M/N/stride_*` | 整数 `tl.tensor` | `[]` | launch 时传入的运行时标量 |
| `BLOCK_M/BLOCK_N` | `tl.constexpr` | 不适用 | JIT 编译期 meta-parameter |
| `pid_m/pid_n` | `tl.tensor<int32>` | `[]` | 逻辑 program 坐标 |
| `offs_m` | `tl.tensor<int32>` | `[16]` | 全局行坐标 block |
| `offs_n` | `tl.tensor<int32>` | `[32]` | 全局列坐标 block |
| `offs_m[:,None]` | `tl.tensor<int32>` | `[16,1]` | 插入单例列轴；不访问内存 |
| `offs_n[None,:]` | `tl.tensor<int32>` | `[1,32]` | 插入单例行轴；不访问内存 |
| `x_ptrs/out_ptrs` | `tl.tensor<pointer<fp16>>` | `[16,32]` | 广播后的 512 个地址 |
| `mask` | `tl.tensor<int1>` | `[16,32]` | 512 个逐地址有效位 |
| `values` | `tl.tensor<fp16>` | `[16,32]` | `tl.load` 后才得到的数据 |

关键链路是：`offs_m[:,None] * stride_xm` 先把标量 stride 广播成 `[16,1]`；再与 `[1,32]` 的列 offset 相加，广播为 `[16,32]`；最后标量 `x_ptr` 也被 splat 到 `[16,32]`，每个位置生成一次 `addptr`。具体实现见 [`broadcast_impl_value`](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/python/triton/language/semantic.py#L767-L817) 和 [`semantic.add`](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/python/triton/language/semantic.py#L226-L255)。

## 3. 二维 Mask

上一个完整 kernel 中，每个维度都有自己的边界，最终通过逻辑与组合。为看清中间类型，可以把同一行无损拆开：

```python
row_mask = offs_m < M
col_mask = offs_n < N
mask = row_mask[:, None] & col_mask[None, :]
```

`row_mask` 是 `int1[BLOCK_M]`，`col_mask` 是 `int1[BLOCK_N]`；插入维度并广播后，`mask` 是 `int1[BLOCK_M,BLOCK_N]`。[`tl.load`](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/python/triton/language/core.py#L2077-L2111) 规定 mask/other 广播到 pointer block 的 shape，`other=0.0` 再转换为 pointer 的元素 dtype。可读性本身就是正确性工具。

## 4. Reduction：从一块数据归约成较小结果

RMSNorm 对一行 `x[D]` 的核心是：

\[
rstd = \frac{1}{\sqrt{\frac{1}{D}\sum_i x_i^2 + \epsilon}}
\]

下面给出变量闭合的单行 RMSNorm kernel；为突出归约，省略 weight/bias，但没有省略任何地址或 mask：

```python
@triton.jit
def rmsnorm_row_kernel(
    x_ptr,
    y_ptr,
    D,
    eps,
    BLOCK_D: tl.constexpr,
):
    row_id = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < D
    row_start = row_id * D

    x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)
    mean_square = tl.sum(x_f32 * x_f32, axis=0) / D
    rstd = tl.rsqrt(mean_square + eps)
    y = x_f32 * rstd
    tl.store(y_ptr + row_start + offsets, y, mask=mask)
```

若输入为 FP16，`x` 是 `fp16[BLOCK_D]`，`x_f32/y` 是 `fp32[BLOCK_D]`；`mean_square/rstd` 是零维 `fp32[]`，与 `x_f32` 相乘时会广播回 `[BLOCK_D]`。`row_start` 是整数标量，不是 pointer；直到它与 `x_ptr` 相加才生成 pointer block。

三个重要点：

1. `tl.sum(..., axis=0)` 是 block 内归约；
2. 常把 FP16/BF16 输入转为 FP32 累加，提高数值稳定性；
3. padding 元素必须用不会污染归约的 `other` 值。

## 5. Fusion 为什么有价值

若分成三个 kernel：

```text
square -> mean/reduce -> rsqrt/mul
```

中间结果可能多次写回和读取 GM。融合后，一行数据加载一次，在 kernel 内完成平方、归约、缩放并写回。

代价是：

- kernel 更复杂；
- Local Memory/寄存器占用增加；
- 编译时间与 shape 限制可能增加；
- 过度融合可能降低并行度。

## 6. 矩阵乘的三个 Tile

`C[M,N] = A[M,K] @ B[K,N]`。下面给出一个假设 A/B/C 均为 FP16、FP32 累加的完整教学 kernel；它使用真实 Triton 语法，不用 `mask=...` 隐藏边界逻辑：

```python
@triton.jit
def matmul_fp16_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BM, BN), dtype=tl.float32)

    for k_tile in range(0, tl.cdiv(K, BK)):
        k_offsets = k_tile * BK + offs_k
        a_mask = (offs_m[:, None] < M) & (k_offsets[None, :] < K)
        b_mask = (k_offsets[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BK * stride_ak
        b_ptrs += BK * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)
```

若 `BM=16, BN=32, BK=64`，`a_ptrs/a` 分别是 `pointer<fp16>[16,64]` 与 `fp16[16,64]`，`b_ptrs/b` 是 `pointer<fp16>[64,32]` 与 `fp16[64,32]`，`acc` 是 `fp32[16,32]`，`c_ptrs` 是 `pointer<fp16>[16,32]`。`a_ptrs += BK * stride_ak` 是整个 pointer block 加同一个整数偏移，只更新下一轮的地址表达式，不搬运数据。

## 7. Grid 如何覆盖 C

Host wrapper 在已经从 `a.shape/b.shape` 取得 `M/N/K` 后，可构造逻辑二维 grid：

```python
def make_matmul_grid(M: int, N: int):
    return lambda meta: (
        triton.cdiv(M, meta["BM"]),
        triton.cdiv(N, meta["BN"]),
    )


grid = make_matmul_grid(M=4096, N=4096)
```

此处 `grid` 是 Python callable，`meta` 是 Python mapping，返回 Python `tuple[int,int]`；它们都不进入 device IR。`BM/BN` 只在 `matmul_fp16_kernel[grid](..., BM=..., BN=..., BK=...)` launch 时成为 `tl.constexpr`。

也可以把 `(pid_m,pid_n)` 压成一维 `pid`，再解码：

```python
pid = tl.program_id(0)
num_pid_n = tl.cdiv(N, BN)
pid_m = pid // num_pid_n
pid_n = pid % num_pid_n
```

二维更直观，一维更方便自定义 program 排序、L2 复用或持久化调度。两者表达的是同一个逻辑 tile 空间。

## 8. `tl.dot` 与 Cube

`tl.dot` 表达 block matrix multiplication。Triton-Ascend compiler 会结合 dtype、shape、layout 和 target 选择面向 Ascend 的降低路径。

不要把 `tl.dot` 理解成“必然一条硬件指令”。它是较高层语义，后端仍需处理：

- Cube 合法 tile；
- 数据搬入 L1/L0；
- 格式转换；
- 累加 dtype；
- pipeline 与 FixPipe；
- 尾块和对齐。

## 9. Vector、Cube 与 CV Fusion

| 类型 | 典型 Triton 代码 | 主要硬件角色 |
|---|---|---|
| Vector | elementwise、`tl.sum`、`tl.max` | AIV/Vector |
| Cube | 主要计算是 `tl.dot` | AIC/Cube |
| CV Fusion | `tl.dot` 前后还有较重 Vector 逻辑 | AIC 与 AIV 协作 |

例如 `MatMul + Bias + Activation` 可能是 CV 融合候选。收益来自不写回中间矩阵，但需要处理 AIC/AIV 数据交换、负载比例和同步。

## 10. Triton 中为什么没有 `TPosition::A1`，它怎样使用 UB/L1/L0

先给结论：

> 普通 Triton kernel 不需要、也通常不能像 Ascend C 那样给每个 `LocalTensor` 显式写 `TPosition::A1`、`VECIN`。开发者写 tile 级数据流，Triton-Ascend compiler 在 lowering、bufferization 和后端 codegen 中选择 UB、L1、L0A/L0B/L0C 等物理地址空间并生成搬运。但开发者仍必须在容量、tile、对齐和性能层面关心这些存储。

### 10.1 两种编程模型的控制边界

| 问题 | Ascend C | Triton-Ascend |
|---|---|---|
| GM 输入 view | `GlobalTensor<T>` | pointer 参数 + `tl.load` |
| 本地存储角色 | `LocalTensor<T>`、`TQue<TPosition::...>` | `tl.tensor`/SSA block value；源码通常不声明物理位置 |
| Cube 输入通路 | 可显式写 A1/B1、A2/B2，或交给高阶 Matmul API | 写 `tl.dot` 与 tile shape，backend 选择 Cube lowering 和存储 |
| Vector 临时数据 | 显式 VECIN/VECCALC/VECOUT、Queue/TBuf | 编译器通常把 block 临时值规划到 UB 等资源 |
| 搬运/同步 | `DataCopy`、Queue、Event/Barrier 等显式 API | 由 `tl.load/store` 语义和 compiler pass 生成大部分搬运/同步 |
| 资源失败 | 开发者的 buffer/通路配置错误 | 常表现为 UB overflow、非法 layout/tile 或 backend lowering 失败 |

`tl.tensor` 也不能直接等同于“一块已经分配好的 UB LocalTensor”。它首先是 Triton IR 中的 SSA value：compiler 可能让它驻留本地存储、融合掉、重算、切成内部循环，或根据后续 `tl.dot`/Vector 使用选择不同数据通路。

### 10.2 编译器并没有忽略硬件位置

Triton 源码层不写 `TPosition`，不代表 backend 只有一个模糊的 “local memory”。固定版本的 [`ascend_ir.cc`](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/third_party/ascend/ascend_ir.cc#L461-L466) 明确向编译器 Python binding 暴露：

```text
AddressSpace::L1
AddressSpace::UB
AddressSpace::L0A
AddressSpace::L0B
AddressSpace::L0C
```

同时，Triton tensor/memory op 会经过 TTIR 到 Linalg/Ascend IR 的 lowering，`triton::DotOp` 也会被转换成矩阵乘语义。控制权实际发生了转移：

```text
Ascend C:
  开发者明确声明 A1/A2/B1/B2/CO1 等逻辑位置

Triton:
  开发者声明 load -> tile compute/tl.dot -> store
  compiler/backend 决定物理 address space、内部 tile 和搬运
```

硬件存储没有消失，只是不再由普通 Triton 源码逐 buffer 指定。

### 10.3 Vector Add 与 MatMul 分别会怎样理解

Vector Add：

```python
x = tl.load(x_ptr + offsets, mask=mask)
y = tl.load(y_ptr + offsets, mask=mask)
z = x + y
tl.store(out_ptr + offsets, z, mask=mask)
```

概念数据通路通常是：

```text
GM pointers
  -> backend 生成 GM 到 UB 的搬运
  -> Vector 在 UB/相关本地资源上完成加法
  -> backend 生成写回 GM
```

开发者没有写 `VECIN`，但 `BLOCK_SIZE`、dtype、同时存活的 `x/y/z`、mask 和 multibuffer 会共同决定 UB 压力。

MatMul：

```python
a = tl.load(a_ptrs, mask=...)
b = tl.load(b_ptrs, mask=...)
acc = tl.dot(a, b, acc)
```

概念数据通路是：

```text
GM A/B tiles
  -> backend 为 Cube 规划 L1 与 L0A/L0B 等输入层次
  -> Cube 产生/累加 L0C 结果
  -> 必要的 Vector 后处理与输出写回
```

开发者没有显式写 `A1/A2/B1/B2/CO1`；`tl.dot`、tile shape、dtype、layout 与 target backend 给了 compiler 足够的意图。`C1/C2` 是 bias 输入位置，只有生成的矩阵路径确实融合 bias 时才有对应资源需求。A/B/C/CO 的完整含义见[硬件基础第 5.2 节](../foundations/02-ascend-hardware.md#52-a1b1c1a2b2c2-中的字母和数字到底表示什么)。

### 10.4 Triton 开发者仍然必须关心什么

不手写 `TPosition` 不等于可以不懂硬件。至少需要关心：

1. **tile 容量**：`BLOCK_M/N/K`、accumulator、mask、临时值和多缓冲能否放入目标 UB/L1/L0；
2. **计算单元**：`tl.dot` 是否形成合适的 Cube matmul，逐元素/归约是否主要落到 Vector；
3. **对齐与 layout**：尾轴、stride、dtype 和矩阵基本块是否满足高效搬运与 Cube 要求；
4. **值的 live range**：融合过多中间 block 会让它们同时存活，导致 UB/L0C 压力；
5. **compiler options**：`num_stages`、`multibuffer`、CV fusion 选项会改变本地 buffer 与流水；
6. **验证工具**：通过 IR dump、编译日志、UB memory report 和 profiler 检查实际 lowering，而不是仅凭 Python 源码猜测。

Triton-Ascend 还提供少量更硬件化的扩展，例如某些 `gather_out_to_ub`/`scatter_ub_to_out` 语义和 compile hint。它们允许高级开发者向 backend 表达更多意图，但仍不是在 Python kernel 中直接创建 `TQue<TPosition::A1>`。

因此两种技术栈的差别可以压缩成：

> Ascend C 要你更直接地管理“数据放在哪、什么时候搬”；Triton 要你准确描述“按什么 tile 读、算、写”，再对 compiler 自动选择的存储与流水负责验证和调优。

## 11. Tile 选择的约束

对矩阵乘，`BM/BN/BK` 同时影响：

- A/B tile 的搬运字节；
- C accumulator 大小；
- Cube 基本块利用率；
- L1/L0/UB 占用；
- 多核并行 tile 数；
- K 循环次数；
- 尾块比例。

没有脱离 shape 分布的“最佳 BLOCK”。LLM decode 常有很小的 M，prefill 则有较大 M；同一个 config 不一定适合二者。

## 12. 从简单到复杂的练习顺序

1. 1D vector add：掌握 pid、offset、mask；
2. 2D add：掌握 stride 和广播地址；
3. row sum：掌握 reduction；
4. RMSNorm：掌握 FP32 累加和 fusion；
5. matmul：掌握 BM/BN/BK 与 K-loop；
6. fused matmul epilogue：理解 CV fusion 价值；
7. attention：组合 matmul、softmax、mask 与 streaming reduction。

## 13. 本章检查点与参考答案

### 1. 为什么 shape 相同的两个 tensor 可能需要不同地址计算？

**答案：**shape 只规定逻辑维度大小，不规定底层元素排列；stride、storage offset 和 layout 才决定地址。

两个 `[M,N]` tensor 中，一个可能是 contiguous，地址公式为 `i*N+j`；另一个可能来自 slice 或 transpose 后再 view，行与列的 stride 不同，甚至首元素还有非零 storage offset。如果 kernel 写死 contiguous 公式，它只能正确处理第一种布局。

因此 custom kernel 必须明确契约：要么 wrapper 检查 `is_contiguous()` 并在必要时复制，要么把每维 stride 传入 kernel。支持任意 stride 更通用，但地址计算和访存模式也可能更复杂、性能更差。

### 2. `offs_m[:,None] + offs_n[None,:]` 在做什么？

**答案：**它利用广播把两个一维坐标轴组合成二维坐标网格。

`offs_m` 的 shape 是 `[BM]`，加上新维后成为 `[BM,1]`；`offs_n[None,:]` 是 `[1,BN]`。广播结果为 `[BM,BN]`：每一行使用一个 m 坐标，每一列使用一个 n 坐标。

真正的二维地址通常还会乘 stride：`base + offs_m[:,None]*stride_m + offs_n[None,:]*stride_n`。这一次构造出整个 tile 的 pointer block，使后续 `tl.load`、mask 和计算都以二维 block tensor 进行。

### 3. Reduction padding 为什么常用 `other=0`？最大值归约也能无脑用 0 吗？

**答案：**padding 值必须是该归约运算的**单位元**或至少不影响有效结果的值；0 只适合部分运算。

Sum 的单位元是 0，因此无效 lane 取 0 不改变总和。乘积归约通常用 1。Max 若用 0，当所有有效元素都为负数时，padding 0 会错误成为最大值；应使用该 dtype 可表示的 `-inf` 或足够小的值。Min 则对应 `+inf`。

RMSNorm 的平方和可以用 `other=0`，但分母仍必须除以真实维度而不是 padding 后的 block 长度。可见 mask、other 和归约公式需要一起审查。

### 4. `tl.dot` 为什么不能简单等同于一条 Cube 指令？

**答案：**`tl.dot` 是 block matrix multiplication 的高层 IR 语义，而硬件执行需要一整套数据准备、分块和指令序列。

后端要检查 dtype 与 tile 合法性，将 A/B 从 GM 搬入合适的 L1/L0 位置，必要时转换 layout，沿 K 维执行多轮 Cube 矩阵乘加，在 accumulator 中累积，再通过输出通路转换和写回。一个较大的 `tl.dot` 通常降低为多条搬运、同步和矩阵指令，而不是“一行 DSL 对应一条机器指令”。

这个区别很重要：`tl.dot` 语义正确不代表 tile 一定高效。BM/BN/BK、对齐、layout、K 尾块和流水仍决定实际 Cube 利用率。

### 5. Decode 和 prefill 为什么可能选择不同 matmul tile？

**答案：**两阶段的矩阵形状和优化目标不同。

Prefill 一次处理大量 prompt token，matmul 的 M 通常较大，有足够输出 tile 填满多核，适合较大的 BM/BN 来提高数据复用和吞吐。Decode 每个活跃序列通常只产生一个新 token，M 很小；过大的 BM 会产生大量 padding 和无效计算，或让可并行 tile 数不足。

Decode 更关注单步延迟、小 M 利用率和减少 launch，prefill 更关注大矩阵吞吐。即便 K/N 相同，也常需不同 config、persistent 策略或直接选择不同 vendor kernel。最佳选择必须覆盖真实 batch 和并发分布，而不是只测一个方形矩阵。

### 6. Triton kernel 不写 `TPosition`，是否意味着无需理解 UB、L1 和 L0？

**答案：**不是。普通 Triton 源码把物理位置选择交给 compiler，但 tile、dtype、layout、同时存活的中间值和流水选项仍决定能否合法、高效地使用这些资源。

Vector block 通常需要 UB；`tl.dot` 的 A/B tile 与 accumulator 会触发 Cube 的 L1、L0A/L0B、L0C 规划。若 `BLOCK` 太大或融合产生太多 live values，即使源码没有任何 `TPosition`，仍会出现 UB overflow、layout 不支持或性能下降。

正确工作方式是：源码层用 `tl.load/tl.dot/tl.store` 表达数据流；编译阶段用 IR、memory report 和错误信息核对 address-space/buffer 规划；运行阶段用 profiler 观察 MTE、Vector、Cube 与等待时间。只有 Ascend C 才通常要求开发者在 API 层显式写 A1/A2 等逻辑位置。

## 官方源码与文档

- [Triton-Ascend MatMul Tutorial](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/third_party/ascend/tutorials/03-matrix-multiplication.py)
- [Triton-Ascend LayerNorm Tutorial](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/third_party/ascend/tutorials/05-layer-norm.py)
- [Triton-Ascend Cube 算子开发指南](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/docs/zh/programming_guide/cube_operator.md)
- [Triton-Ascend CV 融合开发指南](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/docs/zh/programming_guide/cv_fusion_operator.md)
