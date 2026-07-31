# Triton-Ascend 01：Program、Grid、Tile 与第一个 Kernel

本章用 Triton-Ascend 官方 `01-vector-add.py` 建立完整编程模型。源码基线：[`triton-lang/triton-ascend@be90ac7`](https://github.com/triton-lang/triton-ascend/tree/be90ac7e52267822c0ea83d20b705c1e4eaf586f)。

阅读本章代码前，先记住：kernel 内的 `tl.tensor` 是编译器 IR value，不是 `torch.Tensor`。若“标量、block、pointer block”还不熟悉，请先读[代码阅读手册](../reference/code-reading-and-types.md)。本章每个核心变量都会同时标出 dtype、shape 与已知时刻。

## 1. Triton-Ascend 的定位

Triton-Ascend 保留 Triton 的 Python DSL 和 JIT 使用方式，并增加 Ascend language extension、compiler backend 与 runtime driver。开发者描述 tile 级计算，编译器负责把 TTIR 等中间表示降低为 Ascend NPU 可执行对象。

适合它的第一类任务是：

- 逻辑容易写成规则 tile；
- 希望快速融合几个 PyTorch 操作；
- 需要比纯 PyTorch 更少的中间 tensor 和 launch；
- 不想一开始就手工管理 Ascend C 的全部队列和存储细节。

## 2. 官方 Vector Add 的四层结构

官方样例位于 [`third_party/ascend/tutorials/01-vector-add.py`](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/third_party/ascend/tutorials/01-vector-add.py)。它可拆成：

```text
Python wrapper
  ├─ 分配 output
  ├─ 计算 n_elements
  ├─ 定义 grid
  └─ launch kernel

Triton kernel
  ├─ 取得 pid
  ├─ 构造 offsets
  ├─ 构造 mask
  ├─ load x/y
  ├─ compute x+y
  └─ store output
```

先看固定 commit 中 kernel 主体的等价源码摘录；只删除了注释，没有改 API 或补造变量：

```python
@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
```

假设 wrapper 传入 FP16 tensor、`BLOCK_SIZE=1024`，变量类型沿源码传播如下。`[]` 表示零维标量，`[1024]` 表示一个静态 block；它们都是 `tl.tensor`，不是 Python list：

| 变量 | Triton 类型 | 编译时还是运行时 | 含义 |
|---|---|---|---|
| `x_ptr/y_ptr/output_ptr` | `pointer<fp16>[]` | 地址值在 launch/device 运行时给出，pointee dtype 参与 JIT specialization | 三个 tensor 首元素的设备指针 |
| `n_elements` | 整数标量 `tl.tensor` | 运行时 | 输入真实长度；具体整数位宽由实参推导 |
| `BLOCK_SIZE` | `tl.constexpr` | 编译时 | 当前 program 的静态 lane 数 |
| `pid` | `int32[]` | 运行时 | 逻辑 program ID |
| `block_start` | `int32[]` | 运行时 | 当前 tile 的标量起始元素下标 |
| `tl.arange(...)` | `int32[1024]` | IR 在编译时定 shape，lane 值在设备语义中使用 | tile 内相对下标 |
| `offsets` | `int32[1024]` | 运行时 block value | 1024 个全局元素下标 |
| `mask` | `int1[1024]` | 运行时 block value | 每个 lane 是否落在输入范围内 |
| `x/y/output` | `fp16[1024]` | 运行时 block value | 加载值与逐元素相加结果 |

## 3. `@triton.jit`

`JIT` 是 **Just-In-Time Compilation（即时编译）**。这里的“即时”不是“安装 Triton 时编译”，也不是“每次执行每条算术语句时编译”，而是：

> Python 程序已经运行起来；当某个 Triton kernel 第一次以一种尚未编译过的参数特征被 launch 时，Triton 才为这个特征生成并编译 Device kernel。以后命中同一编译变体时直接复用。

因此要把三个时刻分开。

### 3.1 写下装饰器时，还没有执行 NPU kernel

Python 导入模块并执行：

```python
@triton.jit
def add_kernel(...):
    ...
```

等价于“先定义 Python 函数，再把它交给 `triton.jit(...)`”。[`jit()`](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/python/triton/runtime/jit.py#L880-L936) 返回的是 `JITFunction` 包装对象。此时通常只是保存函数源码、签名、参数注解和依赖，还没有因为这一行就把 kernel 发到 NPU。

这也解释了为什么被装饰后的对象不能像普通 Host 函数一样直接写：

```python
add_kernel(x, y, out, n)  # 错误：没有 grid，也不是普通 Python 调用
```

作为顶层 kernel 时应使用：

```python
add_kernel[grid](x, y, out, n, BLOCK_SIZE=256)
```

`JITFunction.__getitem__(grid)` 返回一个记住 `grid` 的 launch proxy，最终进入 `JITFunction.run(..., warmup=False)`。另一个 `@triton.jit` 函数可以在 Device 语义中调用它，但这与 Host 直接调用仍不是一回事。

### 3.2 第一次 launch 内部发生什么

结合 [`KernelInterface.__getitem__()` 与 `JITFunction.run()`](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/python/triton/runtime/jit.py#L411-L419) 以及 [`run()` 的编译/launch 分支](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/python/triton/runtime/jit.py#L709-L748)，一次调用可以拆成：

```text
add_kernel[grid](...)
  -> 取得 current NPU device 与 current stream
  -> 把 Python 实参与函数形参绑定
  -> 推导 signature、constexpr、specialization 和 compiler options
  -> 计算当前变体的内存 cache key
  -> 内存 cache 命中？
       是：复用 CompiledKernel
       否：进入 compiler.compile(...)
             -> 检查磁盘 cache
             -> 未命中才生成 TTIR、执行 lowering、调用后端编译
             -> 保存编译产物与 metadata
  -> 计算本次 launch 的 grid
  -> launcher 在 current stream 上提交已编译 kernel
```

这里有两层 cache：

1. `JITFunction` 的**进程内 cache**保存当前 Python 进程已经得到的 `CompiledKernel`；
2. compiler 的**磁盘 cache**保存 IR、NPU binary、metadata 和 launcher 等产物，让新进程也可能避免完整重编译。

[`compiler.compile()`](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/python/triton/compiler/compiler.py#L228-L393) 会把源码、target、compiler options 和会使缓存失效的环境变量等纳入编译缓存判定。完整 TTIR、MLIR、BiSheng、driver 与磁盘缓存链路见[第 04 讲](./04-ttir-mlir-driver-and-cache.md)。

### 3.3 什么变化会产生新的 JIT 变体

最常见的变体因素包括：

- pointer 指向元素的 dtype，例如 FP16 输入与 BF16 输入；
- `tl.constexpr` 的值，例如 `BLOCK_SIZE=256` 与 `BLOCK_SIZE=1024`；
- backend 选择的参数 specialization，例如某些整数值或对齐属性；
- target NPU/backend、编译选项以及会使 cache 失效的环境变量；
- kernel 源码、被捕获的编译期全局值或依赖发生变化。

反过来，**运行时 tensor shape 变化不必然重编译**。如果 `n_elements` 只是普通运行时整数，`1000` 与 `2000` 可以使用同一 binary，通过 mask 处理尾块；如果 shape 被拿来计算不同的 `BLOCK_SIZE: tl.constexpr`，才会间接选中另一个变体。

`grid` 也通常是 launch 时才求值的调度规格，不会仅仅因为 program 数从 4 变成 8 就自动重编译同一 kernel。若 grid lambda 同时根据 shape 选择了不同 meta-parameter，则变化的是 meta-parameter 对应的编译变体，而不是“grid 数字本身必然进 cache key”。

### 3.4 JIT、warmup 与稳定执行时间

第一次正常调用的耗时可能包含：

```text
Python 参数绑定
+ JIT 编译/磁盘 cache 查询
+ launcher 或 runtime 初始化
+ kernel launch
+ Device 执行
```

后续命中热 cache 的调用通常只剩参数绑定、launch 与 Device 执行。因此不能把第一次 wall time 当成稳定 kernel latency。

还要区分两种常被简称为 warmup 的操作：

- `kernel.warmup(...)` 是 Triton 提供的接口；源码进入 `run(..., warmup=True)`，可以准备/编译变体而跳过实际 launch；
- benchmark 前重复调用若干次 kernel，是工程上的运行 warmup，用来同时消化 JIT、runtime、allocator、cache 等一次性成本。

二者目的相关，但不是同一个 API，也不能默认产生完全相同的 Device 状态。

### 3.5 JIT 编译的不是“任意 Python”

Kernel 函数里只能使用 Triton 支持的语言构造。不要期待任意 Python 对象、动态容器和运行时反射都能进入 device code。

更精确地说，Python 定义函数只提供语法外壳；JIT 前端把 `pid`、`offsets`、`x` 等名字绑定到 [`triton.language.core.tensor`](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/python/triton/language/core.py#L841-L883) 对象。这个类保存 IR `handle`、完整 `type`、静态 `shape` 和标量 `dtype`。因此 `x + y` 调用的是 DSL 重载并生成 IR，不是 CPython 在 NPU 数据上直接做加法。

## 4. Pointer 参数

Wrapper 传入 PyTorch NPU tensor，launch 层将它作为 device pointer 交给 kernel：

```text
x: torch.Tensor on NPU
       |
       v
x_ptr: 指向 x 第一个元素的 device pointer
```

`x_ptr + offsets` 不是立刻加载数据，而是得到一组元素地址。若 `x_ptr` 是 `pointer<fp16>[]`、`offsets` 是 `int32[1024]`，结果就是 `pointer<fp16>[1024]`。真正读取发生在 `tl.load`，其结果是 `fp16[1024]`。

这里的 `+` 是 `tl.tensor.__add__` 的重载。Triton 的 [`binary_op_type_checking_impl`](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/python/triton/language/semantic.py#L171-L206) 先把标量 pointer 广播成与 offsets 相同的 block shape，[`semantic.add`](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/python/triton/language/semantic.py#L226-L255) 再生成 `create_addptr` IR。源码明确禁止 pointer 与 pointer 相加，也不允许 pointer 加浮点数；这里合法是因为 offsets 是整数元素偏移。

偏移单位是**元素**而非字节：`pointer<fp16> + 3` 指向第 3 个 FP16 元素。这个规则解释了为什么 tensor stride 可以直接出现在地址公式里。

## 5. `tl.constexpr`

`BLOCK: tl.constexpr` 表示 `BLOCK` 是编译期常量。编译器可以用它决定：

- `tl.arange` 的静态形状；
- 展开或优化循环；
- Local Memory 需求；
- 生成哪一个 kernel 变体。

代价是不同 `BLOCK` 可能产生不同编译缓存项。不要把每个运行时变化值都随意声明为 `constexpr`。

## 6. Program ID

```python
pid = tl.program_id(axis=0)
```

当前使用一维 grid，所以读取 axis 0。若 grid 是 `(G0, G1)`，则可以分别读取 `program_id(0)` 和 `program_id(1)`。

`pid` 的 Python 前端类是 `tl.tensor`，其值类型通常是零维 `int32[]`；它不是 Python `int`，也不是物理核编号。`axis=0` 则是编译函数时就能解析的 Python/constexpr 参数。

Program 以 tile 为工作单位。假设 `BLOCK=4`：

| pid | offsets |
|---:|---|
| 0 | `[0,1,2,3]` |
| 1 | `[4,5,6,7]` |
| 2 | `[8,9,10,11]` |

## 7. `tl.arange` 和 Block Tensor

```python
lane = tl.arange(0, BLOCK)
```

它产生一个 Triton block tensor。若 `BLOCK=256`，其完整教学类型为 `int32[256]`。[`semantic.arange`](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/python/triton/language/semantic.py#L625-L644) 创建 `block_type(int32, [256])` 与 `make_range` IR；它不是 Python `range`。后面的地址、mask、load 结果和加法结果也都是 block tensor：

```text
lane       shape=[BLOCK], integer offsets
x_ptr+lane shape=[BLOCK], pointers
x          shape=[BLOCK], values
x+y        shape=[BLOCK], values
```

这就是 tile-based 思维：代码写一次，作用于一整块元素。

## 8. Mask 与尾块

如果 `n=10, BLOCK=4`，最后一个 program 的 offsets 是 `[8,9,10,11]`，其中 10 和 11 越界。

```python
mask = offsets < n
x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
result = x + y
tl.store(out_ptr + offsets, result, mask=mask)
```

`load` 的 mask 决定哪些地址可读取；`other` 给被 mask 元素一个值。`store` 的 mask 决定哪些位置真正写回。

Mask 不只是防崩溃。错误 mask 可能静默地丢数据、污染结果或引入额外访问，因此必须进入单元测试的边界 shape。

## 9. Grid 的 Wrapper

官方样例用：

```python
grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024)
```

这一段运行在 Host 的 Python eager 阶段：`grid` 是 Python callable，`meta` 是包含编译 meta-parameter 的 Python mapping，返回值是单元素 Python tuple；`x/y/out` 是 NPU `torch.Tensor`，`n` 是 Python `int`。进入 kernel 后，它们才分别变成 pointer、运行时标量与 `tl.constexpr`。

`triton.cdiv` 是向上整除：

\[
grid = \lceil n / BLOCK \rceil
\]

`grid` 写成 lambda，是因为它可以读取 autotune/config 选择后的 meta-parameter。

## 10. 两种 NPU Grid 策略

### 10.1 一个逻辑 tile 对应一个 program

官方入门样例使用：

```text
grid = num_tiles
program pid -> tile pid
```

优点是直观，适合验证语义。缺点是大 tensor 可能产生远多于物理核数的 program。

### 10.2 固定物理核数，program 内循环

Triton-Ascend 的 Vector 开发指南更推荐生产路径考虑：

```python
@triton.jit
def persistent_add(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    num_tiles = tl.cdiv(n_elements, BLOCK_SIZE)

    for tile_id in range(pid, num_tiles, num_programs):
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        tl.store(out_ptr + offsets, x + y, mask=mask)
```

Host 端让 `grid=(num_vectorcore,)`。每个 program 处理 `pid, pid+num_programs, ...` 这些 tile。

这里 `tile_id`、`num_tiles`、`num_programs` 都是运行时整数标量 `tl.tensor`；`BLOCK_SIZE` 仍是编译期值。循环体中的 `offsets` 每轮重新成为 `int32[BLOCK_SIZE]`。这段代码没有用省略号隐藏 load/store，因此可以直接看清 pointer block 的生成位置。

这不是无条件更快：任务很小、负载不均或编译器优化能力变化时仍需 benchmark。但它揭示了 Ascend 与 GPU 迁移时最重要的差异之一——不要默认超大 grid 的调度成本可以忽略。

## 11. Wrapper 还应做什么

生产 wrapper 通常需要：

- 检查 shape 是否兼容；
- 检查 dtype、device 和 contiguous/stride 契约；
- 分配输出或接受 `.out` buffer；
- 选择 BLOCK/grid/config；
- 处理空 tensor；
- 暴露清晰的 Python API；
- 与 reference 实现比较正确性。

Device kernel 只有十几行，不代表完整算子只有十几行。

## 12. 正确性测试

至少覆盖：

```python
def checked_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.device.type == "npu" and y.device == x.device
    assert x.dtype == y.dtype and x.shape == y.shape
    assert x.is_contiguous() and y.is_contiguous()
    if x.numel() == 0:
        return torch.empty_like(x)
    return add(x, y)


for n in [0, 1, 31, 32, 33, 1023, 1024, 1025, 98432]:
    x = torch.randn(n, device="npu", dtype=torch.float16)
    y = torch.randn(n, device="npu", dtype=torch.float16)
    torch.testing.assert_close(checked_add(x, y), x + y)
```

还要覆盖支持的 dtype、非连续 tensor 是否拒绝或正确处理、极值/NaN/Inf 行为。

## 13. 本章检查点与参考答案

### 1. `program_id` 返回的是物理核 ID 吗？为什么只能近似关联？

**答案：**不是。`tl.program_id(axis)` 返回当前 program instance 在逻辑 launch grid 某一轴上的编号。

Runtime 会把逻辑 program 调度到可用的物理 AIV/AIC 上。物理核数量少于 program 数时，一个核会先后执行多个 program；调度关系也不应被用户 kernel 当作稳定 ABI。`pid=7` 的含义是“处理第 7 份逻辑数据”，不是“我永远运行在 7 号 Vector Core”。

之所以说只能近似关联，是因为当 grid 恰好按物理核数设置并采用 persistent 写法时，运行时通常可以让每个 program 占用一份核资源处理多轮任务。但这种性能策略仍不赋予 `pid` 查询物理核身份的语义。

### 2. `tl.arange` 为什么意味着 kernel 在处理一块数据？

**答案：**`tl.arange(0, BLOCK)` 产生的不是一个循环迭代器，而是一个包含 `BLOCK` 个整数的 Triton block tensor。

当它与 pointer 相加时会得到一组地址，`tl.load` 一次表达对这组地址的加载，`x+y` 也一次表达整组元素的逐元素加法。Kernel 的中间值因而带有静态 block shape，而不是单一标量。

例如 `offsets = pid*256 + tl.arange(0,256)` 表示当前 program 的 256 个逻辑 lane。编译器根据目标硬件把这块语义降低为向量指令、搬运和必要的内部循环，这正是 Triton blocked programming model 的核心。

### 3. `tl.constexpr` 带来什么优化机会和缓存代价？

**答案：**它让参数在编译期已知，从而允许 specialization；代价是不同取值可能生成不同 kernel 变体。

编译器知道 `BLOCK_SIZE=1024` 后，可以确定 `tl.arange` 形状、估算 UB/临时资源、展开循环、删除常量条件分支并选择特定 lowering。例如 `if HAS_BIAS:` 中 `HAS_BIAS` 是 constexpr 时，无 bias 版本可以完全移除该路径。

但若把频繁变化的序列长度、batch size 等都作为 constexpr，每个新组合都可能触发 JIT 和缓存项，造成首次请求抖动、编译时间和缓存膨胀。原则是：把真正决定代码结构或 tile 的少量 meta-parameter constexpr 化；普通数据长度尽量作为运行时参数配合 mask。

### 4. `mask` 为什么必须同时考虑 load 和 store？

**答案：**输入越界和输出越界是两个独立风险，保护其中一侧不能自动保护另一侧。

Load mask 防止读取 tensor 范围外的地址，并用 `other` 为无效 lane 提供安全值。如果只 mask load，却不 mask store，尾 program 仍会把计算结果写到输出边界外，破坏相邻内存。反之，只 mask store 也不能阻止之前的非法读取。

Reduction 还要求选择正确的 `other`：sum 常用 0，max 常用 `-inf`。所以 mask 不只是内存安全开关，也是数学语义的一部分。

### 5. 为什么官方入门 grid 正确，但生产 NPU kernel 仍可能改成固定物理核数？

**答案：**入门写法优先展示一一映射的正确性，生产写法还要控制任务下发成本。

`grid=ceil(N/BLOCK)` 让每个逻辑 tile 对应一个 program，容易理解、负载划分直接，并且结果完全正确。但当 N 很大、BLOCK 较小时，grid 可能远大于物理 Vector Core 数，runtime 要多轮启动和初始化 program。

固定 `grid=num_vectorcore` 后，每个 program 通过 `for tile_id in range(pid,num_tiles,num_programs)` 处理多块数据，可减少逻辑 program 数和下发开销。代价是 kernel 内循环更长，负载均衡和小任务行为可能不同。它是需要 benchmark 验证的 NPU 亲和优化，不是对入门实现正确性的否定。

### 6. `@triton.jit` 是在装饰器执行时、第一次 launch 时，还是每次 launch 时编译？

**答案：**装饰器执行时主要创建 `JITFunction`；第一次遇到未命中的 specialization 时才编译；以后命中同一进程内或磁盘缓存时复用，因而不是每次 launch 都重编译。

调用 `kernel[grid](...)` 后，runtime 先根据实参 dtype、`tl.constexpr`、specialization 属性、target 和 compiler options 等确定当前变体。内存 cache 未命中才进入 compiler；compiler 又会检查磁盘 cache。只有对应编译产物仍不存在或被强制失效时，才完整运行 lowering 和后端编译。

所以“这是同一个 Python 函数”不等于“永远只有一个 binary”，而“输入 shape 变了”也不等于“一定产生新 binary”。判断关键是：变化是否进入 signature、constexpr、specialization 或 compiler cache key。

## 官方源码与文档

- [Triton-Ascend Vector Add 源码](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/third_party/ascend/tutorials/01-vector-add.py)
- [Triton-Ascend Vector 算子开发指南](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/docs/zh/programming_guide/vector_operator.md)
- [Triton `program_id` API](https://triton-lang.org/main/python-api/generated/triton.language.program_id.html)
- [Triton `num_programs` API](https://triton-lang.org/main/python-api/generated/triton.language.num_programs.html)
