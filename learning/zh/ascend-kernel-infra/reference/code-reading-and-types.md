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

## 8. C++ 命名空间：`at::`、`c10::`、`torch::` 和 Ascend 相关前缀

阅读 Ascend C 算子时，通常不是只读 Device kernel。一个完整算子还包含 PyTorch 注册、Host wrapper、torch_npu stream 适配、CANN launch 和 Device 实现，所以同一个 `.cpp` 调用链中会出现多个命名空间。

### 8.1 先理解 `::`：它在回答“去哪个作用域找这个名字”

C++ 的**命名空间（namespace）**是组织名字、避免符号冲突的作用域。例如：

```cpp
at::Tensor logits;
std::int64_t rows = logits.size(0);
AscendC::GlobalTensor<half> logitsGm;
```

逐个读：

- `at::Tensor`：在 `at` 命名空间中查找类型 `Tensor`；
- `std::int64_t`：在 C++ 标准库命名空间 `std` 中查找类型 `int64_t`；
- `AscendC::GlobalTensor`：在 `AscendC` 命名空间中查找类模板 `GlobalTensor`；
- `AscendC::GlobalTensor<half>`：再用类型 `half` 实例化这个类模板。

`::` 是**作用域解析运算符**，不是“访问对象内部数据”。`logits.size(0)` 中的 `.` 才是在对象 `logits` 上调用成员函数；`ptr->method()` 中的 `->` 是通过指针调用成员。命名空间本身不创建对象、不分配内存，也不说明代码运行在 CPU 还是 NPU；必须继续看这个名字的定义和调用位置。

下面两种写法都建立项目自己的嵌套命名空间：

```cpp
namespace sglang {
namespace npu_kernel {
at::Tensor apply_token_bitmask(at::Tensor logits);
}  // namespace npu_kernel
}  // namespace sglang
```

```cpp
namespace sglang::npu_kernel {
at::Tensor apply_token_bitmask(at::Tensor logits);
}
```

函数的完整限定名是 `sglang::npu_kernel::apply_token_bitmask`。项目用它把自己的函数与其他库的同名函数隔离开；这不代表 `sglang` 是一个独立进程或运行时。

### 8.2 常见命名空间分别属于哪一层

| 写法 | 定义/主要归属 | 所在层 | 常见内容 | 初学者应形成的判断 |
|---|---|---|---|---|
| `std::` | C++ 标准库 | Host C++ 与可用的 Device C++ 子集 | `std::min`、`std::tuple`、`std::apply` | 通用语言库，不属于 PyTorch、CANN 或 sgl-kernel-npu |
| `at::` | PyTorch 的 ATen tensor library | PyTorch/Host | `at::Tensor`、`at::ScalarType`、`at::empty`、`at::zeros` | 高层 tensor 句柄与算子 API；`at::Tensor` 的 storage 可以在 NPU，但这个 C++ 对象和这些调用本身位于 Host 框架侧 |
| `c10::` | PyTorch 的 C10 core library | PyTorch 核心/Host | `c10::optional`、`c10::Device`、`c10::DeviceType`、`c10::Scalar`、`c10::Stream` | 跨 backend 的基础类型、设备/流抽象和工具，不是“第十个 C 语言版本” |
| `torch::` | PyTorch C++ frontend 与注册 API | PyTorch/Host | `torch::Library`、`torch::Tensor`、`torch::nn` | 面向扩展作者和 C++ 用户的入口；注册相关类型多在这里 |
| `c10_npu::` | `torch_npu` 的 NPU core 适配 | torch_npu/Host | `c10_npu::getCurrentNPUStream()`、设备函数和 NPU guard | 把 PyTorch 的设备/stream 抽象具体化为 NPU backend；它是一个顶层名字 `c10_npu`，不是 `c10::npu` |
| `at_npu::native::` | `torch_npu` 的原生算子/执行框架实现 | torch_npu/Host | `at_npu::native::OpCommand` | 组织 NPU backend 的 Host 执行逻辑；不是 `at::` 内的 Device kernel |
| `platform_ascendc::` | CANN Ascend C Host platform 接口 | CANN/Host tiling | `PlatformAscendCManager`、`CoreMemType::UB` | 查询核数、UB 等硬件事实，供 Host 计算 tiling；不在 AI Core 上执行 |
| `AscendC::` | CANN 的 Ascend C 编程 API | 主要是 Device kernel | `GlobalTensor`、`LocalTensor`、`TPipe`、`TQue`、`DataCopy`、`Add` | 操作 GM/片上 buffer、搬运、计算和同步的 Device 编程抽象 |
| `sglang::npu_kernel::` | sgl-kernel-npu 项目自己定义 | 取决于定义文件；常见为 Host wrapper | `apply_token_bitmask`、`TorchNpuHelper` | 表示代码所有权和作用域，不能只凭名字判断 Host/Device |

最重要的边界是：

```text
at:: / c10:: / torch::        PyTorch 通用 Host 世界
             |
c10_npu:: / at_npu::native:: torch_npu 的 NPU Host 适配
             |
platform_ascendc:: / ACLRT    CANN Host 查询与 launch
             |
AscendC::                     Ascend C Device kernel API
```

它们不是互相替代的四套 Tensor API，而是一次调用在不同层需要的不同抽象。

### 8.3 `at::Tensor` 到底是什么，为什么 Host 可以拿着 NPU Tensor

ATen（A Tensor Library）是 PyTorch 的基础 tensor 与算子库，`at::Tensor` 是它的核心 C++ tensor 类型。这个对象保存/引用：

- shape、stride、dtype、device、layout 等元数据；
- 指向底层 `TensorImpl`/storage 的引用计数句柄；
- 与 autograd、版本计数等框架语义相关的状态。

如果 `logits.device()` 是 NPU，真实元素位于 NPU Global Memory；Host 上的 `at::Tensor logits` 只是管理元数据和 storage 生命周期的轻量句柄。复制一个 `at::Tensor` 句柄通常增加共享引用，不等于把所有 NPU 元素复制一遍。只有 `copy_`、`.to(...)` 或 kernel 等具有数据移动/计算语义的操作才可能读写真实元素。

例如：

固定源码中的实际赋值语句是：

```cpp
workingLogits = at::zeros({numIndices, paddedVocabSize}, selectedLogits.options());
```

类型逐层解释：

| 表达式 | 类型/作用 |
|---|---|
| `at::Tensor` | C++ Host 变量类型；持有 tensor 句柄 |
| `at::zeros(...)` | ATen factory op；创建并清零一个 tensor |
| `{numRows, paddedVocabSize}` | 新 tensor 的 shape |
| `selectedLogits.options()` | 继承 `selectedLogits` 的 dtype、device、layout 等创建选项 |
| `workingLogits` | Host 句柄；若 options 指向 NPU，其元素 storage 位于 NPU |

这行可能触发 NPU 分配/清零算子，但它不是 Ascend C kernel 内的 `AscendC::LocalTensor`，也不会把数据放进某个 AI Core 的 UB。

### 8.4 `torch::Tensor` 与 `at::Tensor` 是两个不同对象吗

在常见 LibTorch/PyTorch C++ extension 头文件中，`torch` 命名空间会公开 ATen 的声明，因此 `torch::Tensor` 和 `at::Tensor` 通常解析到同一个底层 Tensor 类型，而不是两个需要复制转换的容器。PyTorch 的 [`torch/types.h`](https://github.com/pytorch/pytorch/blob/main/torch/csrc/api/include/torch/types.h#L11-L55) 通过 `namespace torch { using namespace at; }` 暴露这些名字。

但不要据此认为 `torch::` 与 `at::` 中所有 API 都完全相同：

- `at::empty`、`at::zeros` 等强调底层 ATen tensor/算子接口；
- `torch::nn`、`torch::optim` 属于 C++ frontend 的更高层能力；
- `torch::Library` 是 custom operator 注册句柄，不是 tensor；
- `torch::Library` 与 `at::Tensor` 可以出现在同一注册函数中，因为它们承担不同职责。

源码使用哪个拼法往往由项目风格和包含的头文件决定。判断语义时，应看最终定义，而不是把 `torch::Tensor` 和 `at::Tensor` 想成 Python Tensor 与 C++ Tensor 的一一对应：Python `torch.Tensor` 跨过 binding 后才由 C++ 代码以 `at::Tensor`/`torch::Tensor` 句柄接收。

### 8.5 `c10::optional<at::Tensor>` 要从外向内读

真实 Host 签名如下：

```cpp
HOST_API at::Tensor apply_token_bitmask(
    at::Tensor logits,
    at::Tensor bitmask,
    c10::optional<at::Tensor> indices);
```

`c10::optional<at::Tensor>` 应从外向内拆成：

1. `at::Tensor`：存在时装入的值类型，是一个 tensor 句柄；
2. `c10::optional<T>`：一个“可能有 `T`，也可能没有”的包装；
3. 所以整个参数表示“可能没有传入 `indices`；若传入，它是 `at::Tensor`”。

在当前 PyTorch 公共头中，`c10::optional` 是对 `std::optional` 名字的兼容公开；旧版本和项目代码仍大量使用 `c10::optional`，因此阅读固定版本源码时不要机械改名。常见操作是：

固定源码中的检查与规范化语句是：

```cpp
bool hasIndices =
    indices.has_value() && indices->defined() && indices->numel() > 0;
at::Tensor rowIndices;
if (hasIndices) {
    rowIndices = indices.value().to(at::kLong).contiguous();
}
```

- `has_value()` 只检查 optional 中有没有值；
- `value()`/`*indices` 取出内部 `at::Tensor` 句柄；
- optional 没有值不等于“有一个 shape 为 `[0]` 的 tensor”；
- 有值也不保证 tensor 可用，还可能需要检查 `defined()`、`numel()`、dtype、device 和 shape。

`c10::optional<at::Tensor>` 是两套 PyTorch 类型的合法组合：外层 C10 提供通用“可选值”容器，内层 ATen 提供 tensor。命名空间不同不妨碍 C++ 模板组合。

### 8.6 `c10_npu::` 和 `at_npu::native::` 为什么看起来像 C10/ATen，却不是同一命名空间

注意下划线与 `::`：

```cpp
c10::Device                         // c10 中的 Device
c10_npu::getCurrentNPUStream()      // 顶层 namespace c10_npu 中的函数
at::Tensor                          // at 中的 Tensor
at_npu::native::OpCommand           // at_npu 中嵌套 native 的类型
```

`c10_npu`、`at_npu` 是 torch_npu 为 NPU backend 定义的命名空间。它们借用相似命名表达“分别贴近 C10 设备基础层”和“贴近 ATen native/算子执行层”，但 C++ 编译器把 `c10` 与 `c10_npu` 视为完全不同的顶层名字。

在 sgl-kernel-npu 的 [`torch_helper.h#L122-L133`](https://github.com/sgl-project/sgl-kernel-npu/blob/d5630dff41c8108216f835597e63f6d3a7445908/csrc/utils/torch_helper.h#L122-L133) 中：

```cpp
auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
at_npu::native::OpCommand::RunOpApi(#kernel_name, acl_call);
```

这两行位于 `EXEC_KERNEL_CMD` 的宏定义中；`#kernel_name` 是 C/C++ 预处理器的**字符串化（stringification）**语法，把宏实参名变成字符串，并不是普通函数体里可随意使用的一元运算符。第一行从 torch_npu 取得当前 PyTorch NPU stream，再拿到底层 CANN stream handle；第二行把实际 launch callable 接入 torch_npu 的执行、错误处理和 stream 语义。二者都运行在 Host，不是两个额外进程。

### 8.7 `AscendC::` 与 `platform_ascendc::` 名字相似，但分处 Host/Device

以 `apply_token_bitmask` 为例，Host 文件查询平台：

```cpp
auto ascendcPlatform =
    platform_ascendc::PlatformAscendCManager::GetInstance();
uint64_t ubSize = 0;
ascendcPlatform->GetCoreMemSize(
    platform_ascendc::CoreMemType::UB, ubSize);
```

`platform_ascendc::` 提供 Host platform/tiling 能力。CPU 用它查询 UB 容量并计算 `tileLength`；`ubSize` 是 Host 标量，不是 UB 本身，也没有为 kernel 分配 UB。

Device 文件则使用：

```cpp
AscendC::LocalTensor<T> logitsLocal =
    inQueueLogits.AllocTensor<T>();
AscendC::DataCopy(
    logitsLocal, logitsGm[logitsGmOffset], curTileLen);
```

`AscendC::` 是 Device API。`LocalTensor` 是 local buffer view，`DataCopy` 才描述真实搬运。真实对应源码分别见 [`op_host/apply_token_bitmask.cpp#L94-L113`](https://github.com/sgl-project/sgl-kernel-npu/blob/d5630dff41c8108216f835597e63f6d3a7445908/csrc/apply_token_bitmask/op_host/apply_token_bitmask.cpp#L94-L113) 和 [`op_kernel/apply_token_bitmask.cpp#L82-L91`](https://github.com/sgl-project/sgl-kernel-npu/blob/d5630dff41c8108216f835597e63f6d3a7445908/csrc/apply_token_bitmask/op_kernel/apply_token_bitmask.cpp#L82-L91)。

因此，“名字里都有 ascendc”不能作为执行位置判断。最可靠的证据是源文件属于 `op_host` 还是 `op_kernel`、编译 target、函数修饰、参数 ABI 和调用的 API。

### 8.8 `aclrtStream`、`aclrtMemcpy` 为什么没有 `aclrt::`

CANN 的许多运行时接口是 C 风格 API。C 语言没有 C++ namespace，库通过名字前缀降低冲突：

- `acl`：AscendCL/ACL 公共前缀；
- `aclrt...`：runtime 相关类型或函数，例如 `aclrtStream`；
- `aclrtlaunch_<kernel>`：构建生成的 kernel launch stub；
- `ACLRT_LAUNCH_KERNEL(...)`：launch 宏。

所以 `aclrtStream` 是一个完整的 C 风格类型名，不应拆成命名空间 `aclrt` 加类型 `Stream`。类似地，`TORCH_CHECK`、`EXEC_KERNEL_CMD`、`ACLRT_LAUNCH_KERNEL` 是宏；宏在预处理阶段展开，也不通过 namespace 查找。

### 8.9 `TORCH_LIBRARY_FRAGMENT(npu, m)` 中的 `npu` 不是 C++ namespace

下面有三种不同的“命名”机制：

```cpp
namespace sglang::npu_kernel { /* C++ namespace */ }

TORCH_LIBRARY_FRAGMENT(npu, m) {
    m.def("apply_token_bitmask(Tensor logits, Tensor bitmask, "
          "Tensor? indices=None) -> Tensor");
}

// Python
torch.ops.npu.apply_token_bitmask(logits, bitmask, indices)
```

逐层解释：

- `sglang::npu_kernel` 是编译期 C++ 作用域；
- 宏参数 `npu` 是 PyTorch dispatcher 的 operator library namespace；
- `torch.ops.npu` 是 Python 对 dispatcher namespace 的动态访问入口；
- 它们不要求存在一个 C++ `namespace npu {}`；
- 多个扩展库都能向 dispatcher 的 `npu` namespace 注册算子，所以它也不是仓库所有权证明。

`pytorch_extensions.cpp` 还使用 `namespace { ... }` 包住注册代码。没有名字的 namespace 叫**匿名命名空间**，其成员只在当前 translation unit（大致可理解为当前 `.cpp` 编译单元）内可见，用于避免把注册辅助符号导出到整个 shared library。它与 dispatcher 的 `npu` namespace 也没有关系。

### 8.10 用一条真实调用链串起所有名字

下面按 `apply_token_bitmask` 的真实源码顺序阅读：

| 源码表达式 | 层 | 本行做什么 |
|---|---|---|
| `c10::optional<at::Tensor> indices` | PyTorch Host 类型合同 | 接收可能缺省的 tensor 参数 |
| `at::zeros(..., logits.options())` | ATen/Host | 按输入 options 创建工作 tensor |
| `platform_ascendc::...GetCoreMemSize(...)` | CANN Host platform | 查询 UB 容量，供 Host 计算 tile |
| `c10_npu::getCurrentNPUStream()` | torch_npu Host core | 取得当前 PyTorch NPU stream |
| `sglang::npu_kernel::TorchNpuHelper::ConvertTypes(...)` | 项目 Host wrapper | 把 `at::Tensor` 句柄转为 launch 所需设备地址 |
| `ACLRT_LAUNCH_KERNEL(...)` | CANN Runtime Host API | 把 kernel 工作提交到 stream |
| `at_npu::native::OpCommand::RunOpApi(...)` | torch_npu Host framework | 将 launch 接入 backend 执行与错误处理 |
| `AscendC::GlobalTensor<T>` | Ascend C Device | 将收到的 GM 地址绑定成 typed view |
| `AscendC::DataCopy(...)` | Ascend C Device | 在 GM 与 local buffer 之间搬运 |

源码阅读时可以使用一个很实用的规则：先圈出每个 `A::B`，查明 `A` 由哪个 header/仓库定义，再判断该行位于 Host 还是 Device。命名空间是线索，不是最终证据。

### 8.11 复杂算子里还会遇到哪些命名空间

当你继续读 Attention、MatMul、DeepEP 或 mega kernel，会看到更多前缀。先按“定义来自哪里、主要运行在哪一侧”分类：

| 写法 | 主要归属与职责 | 常见执行侧 |
|---|---|---|
| `ge::` | CANN Graph Engine（图引擎）相关基础类型，例如 `ge::DataType`、`ge::Format`、`ge::graphStatus` | Host 注册、shape/dtype/format 描述和错误状态 |
| `gert::` | CANN Graph Engine Runtime 相关上下文和描述对象，例如 `gert::Tensor`、`gert::StorageShape`、`gert::TilingContext` | Host runtime/tiling |
| `optiling::` | 算子 tiling 实现常用的组织空间，容纳 tiling class、校验函数和 tiling data helper | 主要是 Host；生成/共享的 tiling struct 也可能被 Device 头文件引用 |
| `matmul_tiling::` | CANN MatMul tiling API/helper，依据 shape、format 和硬件资源生成矩阵计算 tiling 参数 | Host tiling |
| `host_utils::`、`ge_helper::` | sgl-kernel-npu 自己的 Host 工具空间，例如 dtype 转换、hash、构造 GE/GERT 描述对象 | 项目 Host 代码；不是通用 CANN 公共 namespace |
| `pybind11::` | 第三方 pybind11 C++/Python binding 库，定义 Python module、class 和参数绑定 | Host，在 Python 扩展 `.so` 加载/调用边界 |
| `deep_ep::` | sgl-kernel-npu 的 DeepEP 子模块项目空间，例如 `Buffer`、`Config`、`EventHandle` | 表示模块所有权；具体函数仍可能是 Host wrapper 或 device launch |
| `pto::` | 当前 mega kernel 使用的低层 Device tile/instruction helper，包含 `Tile`、`GlobalTensor`、layout/shape 等类型 | Device；与标准 `AscendC::GlobalTensor` 不是同一个类模板 |
| `AscendC::MicroAPI::` | Ascend C 中更贴近 Vector 寄存器/微指令的 API，例如 `RegTensor`、`MaskReg`、寄存器级 `DataCopy`/算术 | Device Vector 路径 |

这里最容易混淆的是同名类型。例如 `gert::Tensor`、`at::Tensor`、`AscendC::GlobalTensor<T>` 和 `pto::GlobalTensor<...>` 都含有 “Tensor”，但分别是 runtime 描述对象、PyTorch Host 句柄、Ascend C GM view 和 PTO Device view，不能互相替换。仍要回到本手册开头的四列：宿主语言类型、已知时刻、元素/shape、地址空间与语义。

这些 CANN namespace 会随版本和算子工程生成方式变化。若遇到课程未列出的 `foo::Bar`，最可靠的做法不是根据缩写猜含义，而是：

1. 查看当前文件的 `#include`；
2. 用“转到定义”或 `rg "namespace foo"` 查定义仓库；
3. 看该文件编入 Host target 还是 Device target；
4. 看 `Bar` 是 tensor 句柄、描述对象、tiling helper、Device view 还是纯枚举。

### 8.12 `using namespace`、`using` 与 namespace alias 会把前缀“藏起来”

源码不一定总写完整限定名：

```cpp
using std::string;
using namespace matmul_tiling;
namespace py = pybind11;
```

三行含义不同：

- `using std::string;` 只把一个名字 `string` 引入当前作用域，之后 `string` 可解析为 `std::string`；
- `using namespace matmul_tiling;` 让该 namespace 中的名字参与当前作用域查找，所以某个无前缀的 tiling helper 仍可能来自 CANN；
- `namespace py = pybind11;` 创建 namespace alias，此后 `py::arg` 就是 `pybind11::arg`。

还有一种词法作用域情况：

```cpp
namespace AscendC {
MicroAPI::MaskReg mask = MicroAPI::CreateMask<int32_t>();
}
```

因为代码本来就在 `namespace AscendC` 内，`MicroAPI::` 会先在当前作用域下解析，完整名字是 `AscendC::MicroAPI::`。这解释了为什么同一文件有时写 `MicroAPI::RegTensor`，有时写 `AscendC::MicroAPI::RegTensor`。

`using namespace` 不会复制实现或产生运行时转换，但会隐藏名字来源并增加冲突风险。阅读时向上搜索 `using` 和当前包围的 `namespace`；编写新代码时，在公共头文件里通常优先使用完整限定名或精确 `using A::B`。

### 8.13 定义与真实源码从哪里查

- [PyTorch ATen C++ API](https://docs.pytorch.org/cppdocs/api/aten/index.html)：`at::Tensor`、factory op、dtype、device 等；
- [PyTorch C10 C++ API](https://docs.pytorch.org/cppdocs/api/c10/index.html)：`c10::Device`、`Stream`、`optional` 和 core utilities；
- [PyTorch Torch Library API](https://docs.pytorch.org/cppdocs/api/library/index.html)：`torch::Library` 与 custom operator 注册；
- [`apply_token_bitmask` Host wrapper](https://github.com/sgl-project/sgl-kernel-npu/blob/d5630dff41c8108216f835597e63f6d3a7445908/csrc/apply_token_bitmask/op_host/apply_token_bitmask.cpp)：同一函数中对照 `at::`、`c10::`、`platform_ascendc::` 和 `c10_npu::`；
- [`torch_helper.h`](https://github.com/sgl-project/sgl-kernel-npu/blob/d5630dff41c8108216f835597e63f6d3a7445908/csrc/utils/torch_helper.h)：对照 `std::`、`c10::`、`c10_npu::`、`sglang::npu_kernel::`、ACLRT 宏和 `at_npu::native::`；
- [`apply_token_bitmask` Device kernel](https://github.com/sgl-project/sgl-kernel-npu/blob/d5630dff41c8108216f835597e63f6d3a7445908/csrc/apply_token_bitmask/op_kernel/apply_token_bitmask.cpp)：对照 `AscendC::` 类型、搬运和计算 API。

源码定位时应优先使用与你部署环境一致的 PyTorch、torch_npu、CANN 和 sgl-kernel-npu 版本。namespace 的职责通常稳定，但具体 header 路径、别名和内部 API 可能变化。

### 8.14 本节自测与答案

#### 1. `c10::optional<at::Tensor>` 中为什么能同时出现两个命名空间？

**答案：**C++ 模板允许用任意满足要求的类型作为模板实参。`c10::optional<T>` 负责表达“可能存在一个 `T`”，`at::Tensor` 负责表达 tensor 句柄，把后者代入前者便得到“可选 tensor”。命名空间只用于唯一定位类型，不要求外层和内层来自同一个库。

#### 2. `at::Tensor x = y;` 会复制 NPU 上的全部元素吗？

**答案：**通常不会。这是 tensor 句柄复制，`x` 与 `y` 共享底层实现/storage 的引用。是否产生新的元素存储，要看具体算子，例如 `clone()`、某些 `contiguous()`、`.to()` 或显式 `copy_`；不能只凭变量赋值判断发生了数据搬运。

#### 3. 为什么 `at::Tensor` 出现在 Host 文件，却能代表 NPU 数据？

**答案：**Tensor 句柄与元素 storage 所在位置是两个维度。Host 对象保存 device、shape、stride、storage 引用等元数据，storage 可以位于 NPU GM。Host wrapper 用句柄检查合同、管理生命周期并取得设备地址，Device kernel 才通过该地址访问元素。

#### 4. `platform_ascendc::CoreMemType::UB` 会在 Host 上创建 UB 吗？

**答案：**不会。它是 Host platform 查询 API 中的枚举值，用来询问目标硬件 UB 容量。查询结果帮助计算 tiling；实际片上 buffer 由 Device kernel 的 `TPipe/TQue/TBuf` 等机制与编译器共同安排。

#### 5. `torch.ops.npu.foo`、`c10_npu::foo` 与 `namespace npu {}` 是一回事吗？

**答案：**不是。第一个是 Python 对 PyTorch dispatcher 逻辑 namespace 的访问；第二个是 C++ 顶层 namespace `c10_npu` 中的限定名；第三个才是在 C++ 源码中声明名为 `npu` 的 namespace。它们属于不同命名系统，只是都包含字符串 `npu`。

#### 6. 为什么判断 Host/Device 不能只看 `AscendC` 字样？

**答案：**`platform_ascendc::` 是 Host 平台查询接口，而 `AscendC::` 主要是 Device 编程 API；项目还可能在 Host 代码中包含 Ascend C 相关 tiling 类型。应联合查看目录、编译 target、入口修饰、参数类型和实际 API，不能按字符串猜测执行位置。

## 9. 本专题的代码标注规则

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
