# Triton-Ascend 04：TTIR、MLIR、Driver 与 Cache

上一章已经讲了怎么调 Grid、看 UB 和做 benchmark；这一章回答另一个初学者常卡住的问题：`@triton.jit` 写出来的 Python kernel，究竟什么时候变成 `kernel.o`，又是谁把它送进 CANN runtime。

读完本章，你应该能做到三件事：

- 看到 `ttir`、`ttadapter`、`mlirbc`、`npubin` 这些文件名时，不再把它们当成一团“编译器黑盒”。
- 遇到报错时，先判断它更像是 TTIR 降低问题、BiSheng 编译问题，还是 driver/launcher/cache 问题。
- 知道哪些环境变量和源码入口可以帮助你抓到中间产物，而不是盲猜。

前置章节：

- [Triton-Ascend 01：Program、Grid、Tile 与第一个 Kernel](./01-program-grid-tile.md)
- [Triton-Ascend 02：地址、广播、归约与矩阵分块](./02-tensor-addressing-reduction-matmul.md)
- [Triton-Ascend 03：编译、调试与性能优化](./03-compile-debug-optimize.md)
- [代码阅读手册：变量类型、形状、地址与源码实现](../reference/code-reading-and-types.md)
- [参考：术语表](../reference/glossary.md)

## 1. 先把几个新名词就地讲清

这里先解释本章第一次出现、又最容易混淆的术语。后面还会回链到[术语表](../reference/glossary.md)，但第一次不会只把你扔过去。

- **IR（Intermediate Representation，中间表示）**：可以把它理解成“编译器内部流转的半成品图纸”。它不是最终可执行文件，但已经比 Python 源码更接近硬件。需要 IR，是因为编译器很难一步把高级语义直接变成 NPU 能跑的二进制，中间必须经过几层逐步变形。
- **TTIR（Triton IR）**：这是 Triton 世界里最核心的一层 IR。它还保留了 `program`、`tl.load`、`tl.dot` 这种 Triton 视角，所以最适合回答“我的 kernel 数学和地址表达式到底长什么样”。
- **Linalg IR**：这是 MLIR 生态里偏张量和线性代数的一层表达。把它想成“更通用、更方便继续交给后端工具链”的中间图纸。Triton-Ascend 会把很多 Triton 专属操作改写到这里，再继续下沉。
- **MLIR Bytecode**：它和文本 `.mlir` 不是两份不同语义的程序，而是“同一份 IR 的二进制封装版”。为什么需要它？因为编译器内部工具链之间传二进制序列化格式更紧凑，也方便某些后端工具继续读取。
- **Driver（驱动适配层）**：这里不是操作系统里的内核驱动，而是 Triton runtime 里那层“把已编译 kernel binary、stream、参数打包后交给 CANN runtime”的 Host 侧胶水。
- **Cache（缓存）**：不是 L1/L2/UB 这些硬件缓存，而是 Triton runtime 在 Host 文件系统里按哈希保存编译产物的机制。它缓存的是 `kernel.o`、launcher `.so` 之类文件，作用是下次别再重编译。
- **Launcher Stub（启动胶水）**：这是 driver 侧临时生成的小段 C++ 扩展代码。它不负责算数学本身，只负责把 Python/Triton 调用参数整理成 NPU runtime 能执行的 launch 形式。

## 2. 直观类比：同一件货，换了好几次“物流面单”

把一个 Triton kernel 想成你寄出的一件货：

- Python `@triton.jit` 源码，是你写给快递员的原始寄件说明。
- TTIR，是第一张标准化面单，先把“寄什么、去哪里、哪些元素有效”写清楚。
- Linalg IR / Bytecode / LLIR，是中转仓里不断改写的内部流转单，目的是让后面的专用分拣设备能看懂。
- `kernel.o` / `npubin`，是最终能被设备装车的“封箱件”。
- driver 和 launcher stub，是把封箱件、收件地址、运输车次一起交给仓库调度系统的人。
- cache，是仓库档案室。下次寄完全同一件货，不必重新打包。

这个类比的重点不是“文件名要背下来”，而是记住：**每一层都在回答不同问题**。所以报错时要先问“哪张面单写错了”，不要一上来就说“编译器坏了”。

## 3. 正式链路：从 Python 到 NPU binary

官方架构文档把 Triton-Ascend 拆成 language extension、compiler、driver 三块，并明确给出主链路：Triton IR -> Linalg IR -> AscendNPU IR -> `triton_xxx_kernel.o`，随后由 driver 对接 CANN runtime 运行。

```mermaid
flowchart LR
  PY["Python @triton.jit kernel"] --> TTIR["TTIR\n先看 Triton 语义是否正确"]
  TTIR --> TTA["`kernel.ttadapter.mlir`\nTriton pass + Linalg 降低结果"]
  TTA --> BC["`kernel.mlirbc` (可选)\nMLIR Bytecode"]
  BC --> BCMLIR["`kernel.mlir` (可选)\nBytecode 重新展开"]
  TTA --> OBJ["`kernel.o` / `kernel_reloc.o`"]
  BCMLIR --> OBJ
  OBJ --> DRIVER["driver.py + launcher stub"]
  DRIVER --> CANN["CANN runtime / stream / load_binary"]
  CANN --> NPU["Ascend NPU"]
```

这里最容易误解的是 `kernel.ttadapter.mlir`。它不是一门新的独立语言，而是 Triton-Ascend 在源码里给“TTIR 经过一串 Ascend pass 之后的文本中间文件”起的文件名。教学上把它单列出来，是为了让你知道：**从这一步开始，问题已经不只是 Python Triton 语法了，而是在看后端降低结果。**

## 4. 编译器源码到底注册了哪些阶段

官方源码 `third_party/ascend/backend/compiler.py` 的 `AscendBackend.add_stages()` 把阶段注册得很直白：

```text
ttir -> ttadapter -> (mlirbc -> bcmlir)? -> npubin
```

下面直接摘录固定 commit 的 `AscendBackend.add_stages()` 主体。这里的变量都是 Host 侧 Python 对象，不是 device `tl.tensor`：

```python
def add_stages(self, stages, options, language):
    if self.target.backend == "npu":
        stages["ttir"] = lambda src, metadata: make_ttir(src, metadata, options)
        if options.force_simt_only:
            stages["npubin"] = lambda src, metadata: ttir_to_npubin(src, metadata, options)
            return
        stages["ttadapter"] = lambda src, metadata: ttir_to_linalg(
            src, metadata, options, named_ops=True
        )
        if options.use_bytecode:
            stages["mlirbc"] = lambda src, metadata: linalg_to_bc_by_triton_mlir_opt(
                src, metadata, options
            )
            stages["bcmlir"] = lambda src, metadata: bc_to_linalg_by_bishengir_opt(
                src, metadata, options
            )
        if options.compile_on_910_95:
            stages["npubin"] = lambda src, metadata: linalg_to_bin_enable_npu_compile_910_95(
                src, metadata, options
            )
        else:
            stages["npubin"] = lambda src, metadata: linalg_to_bin_enable_npu_compile_A2_A3(
                src, metadata, options
            )
    else:
        raise NotImplementedError(f"Backend '{self.target.backend}' is not supported.")
```

类型逐项看：`stages` 是“阶段名 → Python callable”的可变 mapping；`options` 是 `NPUOptions` 实例；`src` 是上一阶段产物对象；`metadata` 是编译元数据；每个 lambda 都是 Host 编译流水线的函数对象。它们不会进入 NPU kernel，也没有 block shape。`stages["ttir"] = ...` 的含义是注册转换函数，不是现在立刻执行 `make_ttir`。

这段结构非常重要，因为它直接告诉你：

1. `ttir` 和 `ttadapter` 是始终存在的核心阶段。
2. `mlirbc` / `bcmlir` 是可选分支，由 `use_bytecode` 控制。
3. 最终 `npubin` 阶段还会区分 `910_95` 和 `A2/A3` 两条编译路径。

固定源码锚点：

- [`compiler.py#L1082-L1100`](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/third_party/ascend/backend/compiler.py#L1082-L1100)
- [架构设计文档：逻辑架构与目录结构](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/docs/zh/architecture_design_and_core_features.md#1逻辑架构)

## 5. `make_ttir()`：先把 Triton 语义整理干净

`make_ttir()` 做的不是“直接生二进制”，而是先对 Triton IR 做一轮通用整理，比如 inliner、CSE、LICM、loop unroll 等。直觉上可以把它理解成：

- 把能内联的先内联；
- 把重复子表达式合并掉；
- 把循环里不该重复算的东西往外提；
- 把 block tensor 级语义整理得更规整。

为什么初学者必须知道这一步？因为如果你 dump 出来的 TTIR 已经不符合预期，后面任何 Linalg、BiSheng、driver 调试都只是治标不治本。

当 `opt.debug` 打开时，`make_ttir()` 会通过 dump manager 把 `kernel.ttir.mlir` 写出来。这意味着：

- 看 TTIR，主要是确认“我的 Triton 语义是否已经错了”；
- 看后续 `.mlir`，主要是确认“后端 lowering 有没有把它改坏，或者根本不支持这个模式”。

固定源码锚点：

- [`compiler.py#L88-L108`](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/third_party/ascend/backend/compiler.py#L88-L108)

## 6. `ttir_to_linalg()`：为什么会有那么多 pass

`ttir_to_linalg()` 是本章最值得精读的函数，因为它把“后端为什么需要这么多中间变形”暴露得很清楚。

源码里能看到的关键 pass 顺序大意如下：

1. `add_auto_blockify`
2. `add_triton_to_structure`
3. `add_discrete_mask_access_conversion`
4. `add_triton_to_annotation`
5. `add_triton_to_unstructure`
6. `add_triton_to_hivm` / `add_triton_to_hfusion` / `add_triton_to_llvm`
7. `add_bubble_up_operation`
8. 再一次 `add_triton_to_structure`
9. `add_triton_to_linalg`

这些名字看着吓人，但可以按“它在解决哪类硬件现实”来理解：

- **structure / unstructure**：有些 Triton 地址和 mask 写法，在抽象层很自然，但对真实 NPU 后端并不天然好处理。于是编译器要么把它整理成更规则的结构化访问，要么干脆承认它是离散访问，改写成显式循环或 fallback 路径。
- **annotation**：把用户在 Triton 代码里表达的编译提示继续往后传，避免高层 hint 在后端阶段丢失。
- **hivm / hfusion / llvm**：这些是 Ascend 专属后端方言或更低层表示，目的是把“需要跨核同步”“需要硬件融合”“需要内联底层能力”的语义继续带下去。
- **triton_to_linalg**：把大量 Triton 专属 op 变成更通用的 Linalg/MLIR 表达，方便 BiSheng 工具链继续接手。

对初学者最有价值的判断方法是：

- 如果错误发生在 `ttir` 还没问题、但 `ttadapter` 已经很奇怪，优先怀疑 lowering 和 pass。
- 如果 `ttadapter` 看起来已经合理，但 `npubin` 阶段报 BiSheng 编译错误，优先去看 compile options、目标架构和后端约束。

固定源码锚点：

- [`compiler.py#L111-L186`](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/third_party/ascend/backend/compiler.py#L111-L186)
- [架构设计文档：TritonToStructured / TritonToUnstructured / TritonToLinalg](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/docs/zh/architecture_design_and_core_features.md#322-simd-compiler)

## 7. `use_bytecode` 到底多了一层什么

`NPUOptions` 里把 `use_bytecode` 的语义写得很明确：

- `True`：Linalg IR -> MLIR Bytecode -> LLIR -> Binary
- `False`：Linalg IR -> LLIR -> Binary

这不是“两个完全不同的编译器”，而是“中间是否先走一遍 bytecode 封装与再展开”。源码里对应的两个函数分别是：

- `linalg_to_bc_by_triton_mlir_opt()`
- `bc_to_linalg_by_bishengir_opt()`

为什么这一层值得初学者知道？

- 因为你看到 `kernel.mlirbc` 时，不该误以为“编译器突然多发明了一份程序”。
- 因为 `ir_override` 支持的文件后缀里，`mlirbc`、`bcmlir`、`npubin` 都被 Ascend patch 额外接管了；所以 override 调试时，后缀本身就代表你是在替换哪一层产物。

这一段是执行序列，不是可执行程序，因此明确使用 `text`：

```text
文本 Linalg/TTAdapter
  -> triton-mlir-opt --emit-bytecode
  -> kernel.mlirbc
  -> bishengir-opt 展开回可继续消费的 MLIR
  -> kernel.mlir
```

固定源码锚点：

- [`compiler.py#L897-L905`](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/third_party/ascend/backend/compiler.py#L897-L905)
- [`compiler.py#L189-L260`](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/third_party/ascend/backend/compiler.py#L189-L260)
- [`__init__.py#L53-L74`](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/third_party/ascend/backend/__init__.py#L53-L74)

## 8. `NPUOptions`：它不是“调参垃圾桶”，而是阶段开关面板

`NPUOptions` 看起来字段很多，但初学者先抓最重要的几类：

- **编译路径类**：`compile_mode`、`use_bytecode`
- **硬件/目标类**：`arch`、`compile_on_910_95`
- **流水与缓存类**：`num_stages`、`multibuffer`
- **调试与替换类**：`debug`、`ir_override`
- **大 grid 类**：`auto_blockify_size`、`enable_auto_blockify`

这几个字段背后分别在回答：

- 走纯 SIMD，还是掺入 SIMT 模板路径？
- 目标到底是 910_95 还是 A2/A3？
- 编译器是否默认启用多缓冲流水？
- 我是正常编译，还是想替换中间产物做定点调试？
- 遇到逻辑 block 远大于物理核数时，要不要让后端主动做 auto-blockify？

尤其要记住 `compile_mode` 的几个值不是“性能档位”，而是**后端允许采取的并行/访问处理策略**。例如 `simd_simt` 和 `simt_template` 只支持 910_95，不是所有目标都能开。

固定源码锚点：

- [`compiler.py#L796-L905`](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/third_party/ascend/backend/compiler.py#L796-L905)

## 9. 最终怎么变成 `kernel.o`

真正把后端中间表示送去生成 NPU binary 的，是 `linalg_to_bin_enable_npu_compile_910_95()` 和 `linalg_to_bin_enable_npu_compile_A2_A3()`。

这里建议初学者抓三个核心事实：

1. 最终输入给 BiSheng 工具链的，已经不是 Python 代码，而是 `kernel.ttadapter.mlir` 或 `kernel.mlir` 这类中间产物。
2. 最终输出文件名在源码里会落成 `kernel.o` 或 `kernel_reloc.o`，随后再被读回成二进制字节。
3. `multibuffer`、`num_stages`、`enable_auto_bind_sub_block`、`compile_mode`、`--enable-auto-blockify-loop` 这些选项，会在这里真正变成命令行开关。

所以当你做性能实验时，真正影响后端二进制的并不只是 kernel 源码；很多时候是：

- 你的 meta-parameter 有没有变；
- 目标架构有没有变；
- 编译模式有没有变；
- 这些变化是否导致 cache key 改变并触发重编译。

固定源码锚点：

- [`compiler.py#L379-L565`](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/third_party/ascend/backend/compiler.py#L379-L565)
- [`compiler.py#L596-L760`](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/third_party/ascend/backend/compiler.py#L596-L760)

## 10. Driver：不是再编一次 kernel，而是把“能跑的东西”接上线

`third_party/ascend/backend/driver.py` 的角色可以概括成两句：

- 它负责准备 Host 侧 launch 胶水。
- 它负责把已编好的 binary、stream 和参数送进 Ascend runtime。

源码里可以看到三层很实用的事实：

### 10.1 `NPUUtils` 先把 runtime 辅助模块也缓存起来

`NPUUtils.__init__()` 会读取 `npu_utils.cpp`，按源码内容求哈希，编成 `npu_utils.so` 后放进 cache。直觉上这是“driver 自己也有一小段要先编好再复用的胶水”。

### 10.2 `NPULauncher` 会生成 launcher stub

`NPULauncher.__init__()` 会：

1. 读取 kernel 的 constants 和 signature；
2. 生成 wrapper C++ 源码；
3. 调 `make_npu_launcher_stub()` 编成 launcher `.so`；
4. 再把这个 `.so` 动态加载进 Python 进程。

这一步为什么关键？因为它解释了：**有时候你看到的是 launcher 编译问题，不是 kernel binary 问题。**

### 10.3 `TRITON_COMPILE_ONLY=1` 只编不跑

`NPULauncher.__call__()` 对 `TRITON_COMPILE_ONLY` 做了显式判断。打开后，它会跳过真正运行，并打印“compiled kernel cache 在哪里”。这在没有把握运行环境是否稳定时很有价值，因为你至少能先拿到编译产物和 cache 目录。

固定源码锚点：

- [`driver.py#L40-L72`](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/third_party/ascend/backend/driver.py#L40-L72)
- [`driver.py#L97-L137`](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/third_party/ascend/backend/driver.py#L97-L137)
- [`driver.py#L140-L200`](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/third_party/ascend/backend/driver.py#L140-L200)

## 11. Cache：缓存了什么，为什么会“看起来没生效”

很多初学者第一次开 dump，会误判“环境变量失效”。更常见的真实原因是：**你命中了旧 cache，本轮根本没重新编译。**

Triton-Ascend 源码里至少有三类东西在进 cache：

- `npu_utils.so`
- launcher stub `.so`
- kernel 编译产物及相关中间文件

`make_npu_launcher_stub()` 还额外说明了一个细节：它会先尝试读取缓存的 `precompiled.h` / `precompiled.h.gch`，`TRITON_DISABLE_PRECOMPILE=1` 可以关掉这条预编译头路径。

因此调试 cache 时，建议脑中分清三层：

1. **kernel cache**：决定是否重跑 TTIR/Linalg/BiSheng 编译。
2. **launcher cache**：决定是否重编那段 Host 侧 `.cxx` 胶水。
3. **precompiled header cache**：决定 launcher 编译是否重用 `precompiled.h.gch`。

固定源码锚点：

- [`driver.py#L217-L297`](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/third_party/ascend/backend/driver.py#L217-L297)
- [环境变量与编译选项参考](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/docs/zh/environment_variable_and_compiler_options_reference.md)

## 12. 最小调试例子：先学会“抓中间产物”，再谈改 kernel

下面给出可直接用于 PowerShell 的执行模板。唯一需要替换的是最后一行真实测试文件路径；环境变量都是 `str -> str` 的进程环境配置，必须在启动测试进程前设置：

```powershell
$env:TRITON_ALWAYS_COMPILE = "1"
$env:TRITON_KERNEL_DUMP = "1"
$env:TRITON_DUMP_DIR = (Resolve-Path ".").Path + "\triton_dump"
$env:MLIR_ENABLE_DUMP = "1"
$env:TRITON_REPRODUCER_PATH = (Resolve-Path ".").Path + "\triton_reproducer.mlir"
python .\tests\test_your_real_kernel.py
```

预期学习目标不是“必须出现某个固定文件名”，而是按这条顺序检查：

1. 有没有触发重编译，而不是命中旧 cache？
2. 有没有拿到 `ttir` 或后续 `.mlir`/dump 产物？
3. 问题是停在 lowering，还是停在 binary 生成，还是停在 launcher/load/stream？

如果你只有一小时，最值钱的输出不是“乱试几十个 env var”，而是先把失败层级缩到上面三类之一。

## 13. 常见错误与误判

| 现象 | 更可能的问题层级 | 为什么 |
|---|---|---|
| 改了环境变量却看不到新 dump | cache / 编译未重新触发 | `TRITON_ALWAYS_COMPILE` 没开，或者命中了旧 cache |
| `compile_mode='simd_simt'` 在某些卡型直接报错 | options / target 约束 | 源码里明确限制它只支持 910_95 |
| `TRITON_ALL_BLOCKS_PARALLEL` 开了以后行为诡异 | 调度假设错误 | 这个优化要求逻辑 block 间顺序无关，否则可能死锁 |
| `ttir` 看着对，`ttadapter` 很怪 | lowering / pass 问题 | 说明高层语义还好，后端改写阶段出了问题 |
| `kernel.o` 已生成，但运行时还报错 | driver / stream / launch / runtime | 编译成功不等于 launch 参数、stream 或 runtime 一定正确 |
| 只盯着 `.o` 文件，不看 launcher `.so` | 诊断视角偏差 | 有些故障来自 Host 侧 wrapper，而不是 device binary 本身 |

## 14. 调试与性能方法：按层分工，不要混着看

建议把工具分成四类：

1. **抓 IR**
   - `MLIR_ENABLE_DUMP`
   - `TRITON_REPRODUCER_PATH`
   - `TRITON_KERNEL_DUMP`
   - `TRITON_DUMP_DIR`
2. **强制重新编译**
   - `TRITON_ALWAYS_COMPILE`
3. **只编不跑**
   - `TRITON_COMPILE_ONLY`
4. **大 grid 调度实验**
   - `TRITON_ALL_BLOCKS_PARALLEL`
   - `auto_blockify_size`

性能侧要特别警惕一个误区：**“改了 `num_stages` 或 `multibuffer`，性能没变，所以编译器选项没用。”** 更稳妥的解释路径是：

1. 这次是否真的重编译了？
2. cache key 是否变化了？
3. 后端是否把对应选项真正翻译成了 BiSheng 命令行？
4. 你的 shape / grid / kernel 类型是否足以触发这类优化收益？

## 15. 练习

1. 画出你自己的“Python -> TTIR -> ttadapter -> npubin -> driver -> runtime”数据流图，并在每个节点写一句“这层最常见的错误是什么”。
2. 选一个已有 Triton kernel，只做静态分析：哪些 meta-parameter 变化会让 cache miss，哪些不会？先写推断，再去源码核对。
3. 假设 `ttir` 正常、`npubin` 失败，列出你会优先检查的三个信息：目标架构、`compile_mode`、哪条 BiSheng 命令行。
4. 用自己的话解释：为什么 `ttadapter` 不是一门新语言，而更像是“教学上拿来识别阶段边界的文件名”。

## 16. 自测问题

- `ttir`、`ttadapter`、`mlirbc`、`npubin` 各自最适合回答哪一类问题？
- 为什么说 driver 问题和 compiler 问题不能混为一谈？
- `TRITON_COMPILE_ONLY` 和 `TRITON_ALWAYS_COMPILE` 分别在解决什么问题？
- `use_bytecode=True` 时，多出来的是哪两步？
- 为什么打开 dump 后还可能什么都看不到？

## 17. 下一步学什么

学完这一章，最自然的下一步不是继续背编译器名词，而是先去看“为什么 Triton-Ascend 在 NPU 上常常要改写 grid 调度”，再回到真实 kernel。建议顺序是：

- [Triton-Ascend 05：Persistent Kernel、大 Grid 与 Task Queue 边界](./05-persistent-kernel-and-large-grid.md)
- [源码 02：Triton Fused Split Q/K Norm](../sgl-kernel-npu/02-triton-fused-split-qk-norm.md)

当你能把 `05` 章里的调度改写，再和这里讲的 compiler/driver/cache 链路拼起来，最后回到 `sgl-kernel-npu` 的真实 Triton kernel，反向映射“TTIR 长什么样、哪一步会过 `ttadapter`、哪一步会进 driver/cache”，这条链才算真的连起来。

## 官方源码与文档

- [Triton-Ascend 架构设计与核心特性](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/docs/zh/architecture_design_and_core_features.md)
- [Triton-Ascend 环境变量与编译选项](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/docs/zh/environment_variable_and_compiler_options_reference.md)
- [`third_party/ascend/backend/compiler.py`](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/third_party/ascend/backend/compiler.py)
- [`third_party/ascend/backend/driver.py`](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/third_party/ascend/backend/driver.py)
- [`third_party/ascend/backend/__init__.py`](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/third_party/ascend/backend/__init__.py)

## 本章验证边界

- 本章依据的是固定 commit 的官方文档和源码静态阅读。
- 当前工作区没有 Ascend NPU / CANN 运行环境，因此我没有声称实际运行或 profiling 了 NPU kernel。
- 你现在能验证的是：阶段命名、源码入口、环境变量、调试路径和数据流解释是否与固定源码一致；还不能据此声称某个 kernel 在真实硬件上已经跑通。
