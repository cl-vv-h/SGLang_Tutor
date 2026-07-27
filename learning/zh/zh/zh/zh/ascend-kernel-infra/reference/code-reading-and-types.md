# 代码阅读手册：变量类型、形状、地址与源码实现

这份手册解决一个贯穿本专题的问题：同一个变量不能只问“它是什么类型”，还要同时问它属于哪一层、元素是什么类型、有没有静态形状、位于哪种存储空间，以及它代表值还是地址。

建议在阅读任何 kernel 前都写出下面四列：

| 维度 | 要问的问题 | 例子 |
|---|---|---|
| 宿主语言类型 | Python/C++ 表面上是什么对象？ | `torch.Tensor`、`tl.tensor`、`AscendC::LocalTensor<half>` |
| 编译时或运行时 | 值在 JIT 编译时已知，还是 launch 后才知道？ | `BLOCK: tl.constexpr` 是编译期；`n_elements` 是运行期 |
| 元素类型与形状 | 一个标量还是一块值？每个元素是什么 dtype？ | `int32[BLOCK_M]`、`pointer<fp16>[BLOCK_M, BLOCK_N]` |
| 地址空间与语义 | 它是数据、地址，还是管理资源的句柄？ | GM 指针、UB tensor view、`TQue` 队列对象 |

只写“tensor”或“pointer”通常不够。本文使用 `int32[128]` 表示“元素类型为 `int32`、静态 shape 为 `[128]` 的 Triton block tensor”，使用 `pointer<fp16>[16, 32]` 表示“包含 `16×32` 个 FP16 元素地址的 pointer block”。这是一种便于教学的记法，不是 Python 的类型注解语法。

## 1. 先分清四个执行时刻

一段自定义算子代码会经历四个不同阶段：

1. **Python eager 阶段**：`torch.Tensor`、Python `int` 和 wrapper 真正在 CPU 上执行。
2. **JIT/编译阶段**：Triton 追踪被 `@triton.jit` 修饰的函数，创建 `tl.tensor` 与 IR；Ascend C 则由 C++ 编译器和 CANN 工具链编译模板、宏和设备函数。
3. **launch 阶段**：runtime 把设备地址、运行时标量、grid/blockDim 等实参交给 NPU。
4. **device 执行阶段**：AI Core 执行搬运、计算、同步与写回。

因此，Triton kernel 源码看起来像 Python，不代表 `offs_m` 是 NumPy 数组；`GlobalTensor<T>` 看起来像 C++ 容器，也不代表其中的数据存放在 CPU 堆内存。

## 2. Triton 的核心类型不是 `torch.Tensor`

### 2.1 `tl.tensor` 是编译器前端值

Triton 源码中的 [`class tensor`](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/python/triton/language/core.py#L841-L883) 保存三类关键信息：

- `handle`：指向当前 Triton IR value 的句柄；
- `type`：完整 Triton 类型，可能是标量类型，也可能是 `block_type`；
- `shape` 与 `dtype`：静态 block shape，以及每个元素的标量类型。

所以 `tl.tensor` 不是装着真实 NPU 数据的 Python 容器。它更像编译器在构图时使用的“带类型 IR 表达式”。`x + y` 会调用 `tensor.__add__`，继续生成新的 IR value，而不是让 Python 逐元素执行加法。

### 2.2 标量也是零维 `tl.tensor`

在 kernel 内，下面三个变量虽然都写成普通名字，但不是同一种东西：

| 变量 | 教学类型 | 已知时刻 | 含义 |
|---|---|---|---|
| `BLOCK: tl.constexpr` | `tl.constexpr` 包装的 Python 编译期值 | JIT 编译时 | 决定静态 shape、循环展开和 kernel 变体 |
| `pid = tl.program_id(0)` | 标量 `tl.tensor`，通常为 `int32[]` | device 执行时 | 当前逻辑 program 在 grid 第 0 轴的编号 |
| `n_elements` | 标量 `tl.tensor`，整数位宽由 launch 参数推导 | device 执行时 | 本次输入的真实元素数 |

[`tl.constexpr`](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/python/triton/language/core.py#L176-L246) 是编译期包装类型；[`tl.program_id`](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/python/triton/language/core.py#L1599-L1614) 则转交 semantic builder 创建运行时 IR 值。两者都能参与 `+`，但一个可在编译期折叠，另一个必须由设备执行。

### 2.3 `tl.arange` 返回整数 block tensor

```python
lane = tl.arange(0, BLOCK)
```

当 `BLOCK=128` 时，`lane` 的类型是 `tl.tensor`，完整类型可记作 `int32[128]`。它不是 Python 的 `range`，也不会让 Python 循环 128 次。

实现上，[`semantic.arange`](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/python/triton/language/semantic.py#L625-L644) 构造 `block_type(int32, [128])`，再通过 `create_make_range` 生成 IR。`128` 必须在编译时可知，因为 block shape 是静态类型的一部分。

### 2.4 Pointer 参数如何获得类型

例如 kernel 签名 `def add_kernel(x_ptr, n_elements, BLOCK: tl.constexpr)` 中，pointer 与运行时整数形参没有写 Python 类型注解，只有编译期参数显式标出 `tl.constexpr`。

launch 时传入 NPU 上的 `torch.Tensor`，Triton runtime 会把 `x_ptr` 专门化为指向该 tensor 元素 dtype 的设备指针。若输入是 FP16，可以把 kernel 内的 `x_ptr` 记作：

```text
标量 tl.tensor，dtype = pointer<fp16>，shape = []
```

Triton 的 [`pointer_type`](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/python/triton/language/core.py#L654-L691) 明确保存 `element_ty` 与 `address_space`。它仍然被包装在 `tl.tensor` 中，因为地址本身也是一个 IR value。

## 3. 为什么 pointer 可以和 offsets 相加

先看一维情况：

```python
offsets = pid * BLOCK + tl.arange(0, BLOCK)
x_ptrs = x_ptr + offsets
x = tl.load(x_ptrs)
```

假设 `x` 的元素 dtype 是 FP16、`BLOCK=128`，类型传播是：

| 表达式 | 结果类型 | 发生了什么 |
|---|---|---|
| `pid` | `int32[]` | 运行时标量 |
| `BLOCK` | 编译期整数 | 参与常量折叠 |
| `tl.arange(0, BLOCK)` | `int32[128]` | 产生 128 个 lane offset |
| `pid * BLOCK` | `int32[]` | 当前 tile 的标量起点 |
| `offsets` | `int32[128]` | 标量被广播后与 lane 相加 |
| `x_ptr` | `pointer<fp16>[]` | 指向输入首元素的标量指针 |
| `x_ptrs` | `pointer<fp16>[128]` | 标量指针广播后，对每个 lane 做 addptr |
| `x` | `fp16[128]` | `tl.load` 才真正从这 128 个地址读值 |

这里的 `+` 是 Triton DSL 重载，不是 CPython 对裸地址做整数运算。调用链是：

```text
tl.tensor.__add__
  -> triton.language.add
  -> semantic.add
  -> binary_op_type_checking_impl（类型检查与广播）
  -> builder.create_addptr（生成指针偏移 IR）
```

对应源码在 [`tensor.__add__`](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/python/triton/language/core.py#L881-L887)、[`binary_op_type_checking_impl`](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/python/triton/language/semantic.py#L171-L206) 和 [`semantic.add`](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/python/triton/language/semantic.py#L226-L255)。源码还明确禁止“pointer + pointer”，并拒绝“pointer + float”；合法的组合是 pointer 加整数元素偏移。

偏移量以**元素**为单位，而不是以字节为单位。`x_ptr + 3` 表示移动到第 3 个 FP16 元素，概念上的字节地址变化是 `3 * sizeof(fp16)`。因此 PyTorch 的 stride 也正好可以直接参与公式，因为 `Tensor.stride()` 返回的就是元素步长。

## 4. 逐类型拆解二维地址例子

下面是一段合法的 Triton kernel 主体语法。为避免把未声明变量藏在片段外，函数签名也一起给出：

```python
import triton
import triton.language as tl


@triton.jit
def load_2d_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    row_offsets = offs_m[:, None] * stride_xm
    col_offsets = offs_n[None, :] * stride_xn
    ptrs = x_ptr + row_offsets + col_offsets

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    values = tl.load(ptrs, mask=mask, other=0.0)
    tl.store(out_ptr + row_offsets + col_offsets, values, mask=mask)
```

假设 `x_ptr/out_ptr` 指向 FP16，`BLOCK_M=16`、`BLOCK_N=32`：

| 变量 | Triton 类型 | shape | 含义 |
|---|---|---:|---|
| `pid_m`, `pid_n` | `tl.tensor<int32>` | `[]` | 二维 grid 中的逻辑坐标 |
| `M`, `N` | 整数标量 `tl.tensor` | `[]` | launch 时传入的运行时边界，不改变 block 静态 shape |
| `stride_xm`, `stride_xn` | 整数标量 `tl.tensor` | `[]` | 相邻行/列相隔多少个元素 |
| `offs_m` | `tl.tensor<int32>` | `[16]` | 16 个全局行坐标 |
| `offs_n` | `tl.tensor<int32>` | `[32]` | 32 个全局列坐标 |
| `offs_m[:, None]` | `tl.tensor<int32>` | `[16,1]` | 插入长度为 1 的列轴 |
| `offs_n[None, :]` | `tl.tensor<int32>` | `[1,32]` | 插入长度为 1 的行轴 |
| `row_offsets` | 整数 block tensor | `[16,1]` | 每行相对首元素的元素偏移 |
| `col_offsets` | 整数 block tensor | `[1,32]` | 每列相对行首的元素偏移 |
| `ptrs` | `tl.tensor<pointer<fp16>>` | `[16,32]` | 广播后得到的 512 个元素地址 |
| `mask` | `tl.tensor<int1>` | `[16,32]` | 每个地址是否有效 |
| `values` | `tl.tensor<fp16>` | `[16,32]` | 从有效地址加载的数据；无效位置取 FP16 的 0 |

`[:, None]` 和 `[None, :]` 只改变 IR shape，不会读取内存。两个 shape 分别为 `[16,1]`、`[1,32]` 的 offset block 做二元运算时，[`broadcast_impl_value`](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/python/triton/language/semantic.py#L767-L817) 把它们广播成 `[16,32]`。标量 `x_ptr` 也会被 splat 成同 shape 的 pointer block。

[`tl.load`](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/python/triton/language/core.py#L2077-L2111) 的规则是：pointer block 是 N 维，load 结果就是同 shape 的 N 维 value block；`mask` 和 `other` 会广播到 pointer shape，`other` 再转换为指针所指的元素 dtype。

## 5. MatMul 的类型传播

对 `C[M,N] = A[M,K] @ B[K,N]`，某一轮 K tile 中的典型类型是：

| 变量 | 类型与 shape | 说明 |
|---|---|---|
| `a_ptrs` | `pointer<a_dtype>[BM,BK]` | A tile 的地址块 |
| `b_ptrs` | `pointer<b_dtype>[BK,BN]` | B tile 的地址块 |
| `a` | `a_dtype[BM,BK]` | load 后的 A tile 值 |
| `b` | `b_dtype[BK,BN]` | load 后的 B tile 值 |
| `acc` | `fp32[BM,BN]` | 静态 shape 的 FP32 累加块 |
| `tl.dot(a,b)` | 通常为 `fp32[BM,BN]` | 高层 block matmul 结果，后端再降低到 Cube 路径 |
| `c_ptrs` | `pointer<c_dtype>[BM,BN]` | 输出地址块 |

源码里 `a_ptrs += BK * stride_ak` 仍然是 pointer block 加整数标量：标量先广播到 `[BM,BK]`，然后每个地址整体前进一个 K tile。它不会加载下一块；下一次 `tl.load(a_ptrs, ...)` 才会发生搬运。

## 6. Ascend C：模板类型与存储位置

Ascend C 的类型更加显式，但“对象类型”和“数据位于哪里”仍要分开看。

这些类型的**定义**由目标 CANN Toolkit 提供，算子源码通常只通过 `#include "kernel_operator.h"` 使用它们；`sgl-kernel-npu` 保存的是调用代码，不会复制整套 `GlobalTensor/TQue` 类模板实现。阅读自己环境时，应在与部署版本一致的 CANN include 目录中对 `kernel_operator.h` 使用“转到定义”，不能拿另一个 CANN 版本的内部头文件行号当稳定 ABI。本课程因此把“定义归属”链接到 CANN，把“真实使用方式”固定到 `sgl-kernel-npu` commit。

| 源码类型 | 它是什么 | 数据在哪里 |
|---|---|---|
| `GM_ADDR` | kernel ABI 使用的通用 GM 地址类型 | 地址指向 Global Memory |
| `__gm__ T*` | 带 GM 地址空间限定的 `T` 指针 | Global Memory |
| `AscendC::GlobalTensor<T>` | 对 GM 中 `T` 元素区域的 typed view/句柄 | 对象保存地址信息，元素仍在 GM |
| `AscendC::LocalTensor<T>` | 对某片片上 buffer 中 `T` 元素区域的 typed view | 通常是 UB，也可能由 API/position 指向 L1/L0 等位置 |
| `AscendC::TPipe` | 片上 buffer、队列和事件资源的管理对象 | 不是业务数据 tensor |
| `AscendC::TQue<P,N>` | position 为 `P`、深度为 `N` 的队列对象 | 管理 local buffer 所有权与生产者/消费者同步 |

源码追踪顺序是：

```text
业务 kernel 的 #include "kernel_operator.h"
  -> CANN Toolkit 中 GlobalTensor/LocalTensor/TPipe/TQue 的声明与内联实现
  -> Ascend C 编译器识别 DataCopy/Add/EnQue 等 device API
  -> 目标架构上的搬运、Vector、事件与资源指令
```

不要把 C++ 模板类误认为硬件里存在一个同名“对象”。模板和 inline API 是编程抽象，编译后会降低为地址、资源配置与设备指令。

以 `sgl-kernel-npu` 的真实 `apply_token_bitmask.cpp` 为例：

```cpp
template <typename T>
class KernelApplyTokenBitmask {
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> inQueueLogits;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> outQueueLogits;
    AscendC::GlobalTensor<T> logitsGm;
};
```

逐项读法：

- `T` 是 C++ 模板类型形参，实例化时可以是 `half`、`float` 或 `bfloat16_t`；它在编译时决定生成哪个 kernel 变体。
- `GlobalTensor<T>` 的模板参数决定元素解释方式。`logitsGm[17]` 中的 `17` 是元素偏移，不是字节偏移。
- `TPosition::VECIN`/`VECOUT` 是逻辑队列位置，表达 Vector 流水的输入/输出角色；它不是一个可以做算术的 tensor dtype。
- 模板参数 `2` 是编译期队列深度。本例再通过 `pipe.InitBuffer(queue, 2, bytes_per_buffer)` 为队列实际分配两个 buffer。

真实初始化语句：

```cpp
logitsGm.SetGlobalBuffer((__gm__ T *)logitsGmAddr);
pipe.InitBuffer(inQueueLogits, 2, tileLength * sizeof(T));
```

第一行把无类型/通用 ABI 地址转换为 `__gm__ T*`，再绑定到 `GlobalTensor<T>`；不搬运任何数据。第二行按**字节数**配置 local queue buffer；这里与 Triton pointer offset 的“元素单位”不同，所以必须显式乘 `sizeof(T)`。

真正的 GM → Local 搬运是：

```cpp
AscendC::LocalTensor<T> logitsLocal = inQueueLogits.AllocTensor<T>();
uint32_t logitsGmOffset = batchId * logitsStride + vocabOffset;
AscendC::DataCopy(logitsLocal, logitsGm[logitsGmOffset], curTileLen);
inQueueLogits.EnQue(logitsLocal);
```

变量类型与动作：

| 变量/表达式 | C++ 类型 | 是不是搬运 |
|---|---|---|
| `logitsLocal` | `AscendC::LocalTensor<T>` | `AllocTensor` 取得 local buffer view，不读 GM |
| `logitsGmOffset` | `uint32_t` | 只做 host-like 标量地址计算 |
| `logitsGm[logitsGmOffset]` | 偏移后的 `GlobalTensor<T>` view | `operator[]` 选择起点，不读 GM |
| `DataCopy(...)` | Ascend C 搬运 API 调用 | 是，这里才从 GM 搬到 local buffer |
| `EnQue(logitsLocal)` | 队列状态/同步操作 | 不复制业务数据；发布 buffer 给消费者 |

完整实例可对照 [`apply_token_bitmask.cpp`](https://github.com/sgl-project/sgl-kernel-npu/blob/d5630dff41c8108216f835597e63f6d3a7445908/csrc/apply_token_bitmask/op_kernel/apply_token_bitmask.cpp)。

## 7. `torch.Tensor`、`at::Tensor` 与设备地址

同一个输入跨语言后常出现三种外观：

```text
Python wrapper:  torch.Tensor
                         |
PyBind/Dispatcher: at::Tensor
                         |
kernel launch ABI: device address / GM_ADDR / Triton pointer parameter
```

`torch.Tensor`/`at::Tensor` 是带 shape、stride、dtype、device 和 storage 生命周期的高层对象；设备 kernel 通常只收到地址与若干整数元数据。因而下面两个问题必须由 wrapper/Host 解决：

1. kernel 只按连续布局寻址时，wrapper 必须检查 `is_contiguous()`；
2. launch 是异步的时，框架必须保证底层 storage 在当前 NPU stream 使用完之前不被释放或复用。

这也是为什么 `record_stream`、schema、dispatch key 和 Host launch 不是“外围样板”：它们决定 kernel 获得的地址是否对应正确布局，以及地址在异步执行期间是否仍有效。

## 8. 本专题的代码标注规则

后续章节统一采用以下标签：

- **可运行最小例子**：包含必要 import、签名和已声明变量；仍需匹配章节注明的软件/硬件环境。
- **固定 commit 的源码摘录**：保持真实 API 和类型，只截取与主题有关的连续部分；用链接回到完整上下文。
- **结构图/执行序列**：使用 `text` 或 Mermaid，只表达关系，不冒充源码。

如果为了聚焦省略错误处理或环境初始化，会在代码前明确列出“省略了什么”，不再用未声明的 `...`、`mask=...` 或虚构 API 混进可执行语言代码块。

阅读任何新片段时，请至少回答：

1. 每个名字是 Python 对象、Triton IR value，还是 Ascend C C++ 对象？
2. 它是编译期常量、运行时标量、value block，还是 pointer block？
3. shape 和元素 dtype 是什么？
4. 地址偏移单位是元素还是字节？
5. 哪一行只构造 view/地址，哪一行才真正搬运或计算？
6. 对应的类型检查、广播或资源管理在源码哪一层实现？
