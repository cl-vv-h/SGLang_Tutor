# Triton-Ascend 01：Program、Grid、Tile 与第一个 Kernel

本章用 Triton-Ascend 官方 `01-vector-add.py` 建立完整编程模型。源码基线：[`triton-lang/triton-ascend@be90ac7`](https://github.com/triton-lang/triton-ascend/tree/be90ac7e52267822c0ea83d20b705c1e4eaf586f)。

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

先看一个等价的教学缩写版：

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)
```

## 3. `@triton.jit`

`@triton.jit` 表示这个函数不是普通 Python 函数。首次遇到某组参数和 meta-parameter 时，Triton 会编译 kernel；后续满足缓存键的调用可复用编译产物。

Kernel 函数里只能使用 Triton 支持的语言构造。不要期待任意 Python 对象、动态容器和运行时反射都能进入 device code。

## 4. Pointer 参数

Wrapper 传入 PyTorch NPU tensor，launch 层将它作为 device pointer 交给 kernel：

```text
x: torch.Tensor on NPU
       |
       v
x_ptr: 指向 x 第一个元素的 device pointer
```

`x_ptr + offsets` 不是立刻加载数据，而是得到一组元素地址。真正读取发生在 `tl.load`。

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

它产生一个 Triton block tensor。后面的地址、mask、load 结果和加法结果也都是 block tensor：

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
def persistent_add(x, y, out, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_tiles = tl.cdiv(n, BLOCK)

    for tile_id in range(pid, num_tiles, num_programs):
        offsets = tile_id * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        ...
```

Host 端让 `grid=(num_vectorcore,)`。每个 program 处理 `pid, pid+num_programs, ...` 这些 tile。

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
for n in [0, 1, 31, 32, 33, 1023, 1024, 1025, 98432]:
    # 构造 NPU tensor
    # reference = x + y
    # actual = triton_add(x, y)
    # torch.testing.assert_close(actual, reference)
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

## 官方源码与文档

- [Triton-Ascend Vector Add 源码](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/third_party/ascend/tutorials/01-vector-add.py)
- [Triton-Ascend Vector 算子开发指南](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/docs/zh/programming_guide/vector_operator.md)
- [Triton `program_id` API](https://triton-lang.org/main/python-api/generated/triton.language.program_id.html)
- [Triton `num_programs` API](https://triton-lang.org/main/python-api/generated/triton.language.num_programs.html)
