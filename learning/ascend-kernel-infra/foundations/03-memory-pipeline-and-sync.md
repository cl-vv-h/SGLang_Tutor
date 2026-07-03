# 基础 03：搬运、计算、同步与流水

高性能 kernel 不是“先把所有数据复制完，再全部计算，最后全部写回”。它更像工厂流水线：前一块在计算时，后一块可以搬入，前前一块可以搬出。

## 1. 三个最基本阶段

Vector kernel 常抽象为：

```text
CopyIn  : GM -> UB
Compute : UB -> Vector -> UB
CopyOut : UB -> GM
```

如果串行处理三个 tile：

```text
时间 --->
Tile 0: [CopyIn][Compute][CopyOut]
Tile 1:                         [CopyIn][Compute][CopyOut]
Tile 2:                                                  [CopyIn][Compute][CopyOut]
```

很多硬件单元在大段时间里是空闲的。

## 2. 流水 Pipeline

**流水**是让不同 tile 的不同阶段重叠：

```text
时间 --->
Tile 0: [CopyIn][Compute][CopyOut]
Tile 1:         [CopyIn][Compute][CopyOut]
Tile 2:                 [CopyIn][Compute][CopyOut]
```

稳态吞吐不再接近三阶段耗时之和，而更接近最慢阶段的耗时：

\[
T_{steady} \approx \max(T_{copyin}, T_{compute}, T_{copyout})
\]

首尾仍有填充和排空开销，所以 tile 太少时流水收益有限。

## 3. 队列不是普通容器

Ascend C 的 `TQue` 同时承担两件事：

1. 管理某个流水阶段可使用的 LocalTensor buffer；
2. 通过 `EnQue` / `DeQue` 表达生产者与消费者之间的数据就绪关系。

```text
CopyIn  --EnQue--> [VECIN queue] --DeQue--> Compute
Compute --EnQue--> [VECOUT queue] --DeQue--> CopyOut
```

它不是为了保存无限多对象，也不是 Python `queue.Queue` 的普通设备版。队列位置和深度都与片上内存及同步事件相关。

## 4. TPipe 做什么

`TPipe` 是 Ascend C Pipe/Queue 编程范式中的资源管理对象，常负责：

- 为 `TQue` / `TBuf` 初始化片上 buffer；
- 管理事件资源；
- 支撑阶段间的流水同步。

典型代码：

```cpp
AscendC::TPipe pipe;
AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue;
AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;

pipe.InitBuffer(inQueue, 2, tileBytes);
pipe.InitBuffer(outQueue, 2, tileBytes);
```

这里模板参数里的 `1` 是队列深度；`InitBuffer` 的第二个参数 `2` 是分配两个 buffer，从而开启 double buffer。二者不是同一个概念。

## 5. Double Buffer

只有一块输入 buffer 时，搬入下一 tile 前必须等当前 tile 不再使用该 buffer。Double buffer 准备 ping/pong 两组空间：

```text
时刻 0: MTE 搬入 ping
时刻 1: Vector 计算 ping；MTE 同时搬入 pong
时刻 2: Vector 计算 pong；MTE 同时复用 ping 搬入下一块
```

优点是增加重叠，代价是片上内存占用近似翻倍。UB 本就有限，因此 double buffer 不是免费按钮。

## 6. 搬运 Data Movement

搬运不仅是 `memcpy`。需要同时考虑：

- 源和目标属于什么存储层级；
- 连续还是跨 stride；
- 搬运多少字节；
- 对齐与尾块；
- 是否能随路做格式/类型转换；
- 多次小搬运能否合成一次大搬运。

带宽有效利用率通常偏爱连续、对齐、大粒度的访问。大量离散标量 load 即使总字节数不大，也可能效率很差。

## 7. 计算 Compute

“计算”也要区分：

- Vector 指令：逐元素、reduce、数学函数；
- Cube 指令：矩阵乘加；
- Scalar 运算：循环、地址、分支；
- 融合：多步计算共享一次搬入和中间 LocalTensor。

一个融合 kernel 是否值得，取决于减少的 GM 流量和 launch 开销，不能只看少了几个 Python 函数。

## 8. 同步 Synchronization

同步是在表达“下一步必须等什么完成”。常见层次：

| 层次 | 例子 | 目的 |
|---|---|---|
| 同一核的流水阶段 | Queue 的 EnQue/DeQue、event | 防止消费者读到尚未完成的数据 |
| 同一核不同指令通路 | MTE 与 Vector/Cube event | 保证搬运和计算顺序 |
| 多核之间 | block sync、cross-core event | 共享结果或阶段协作 |
| Kernel/Stream 之间 | stream 顺序、event、同步 API | 保证跨 kernel 依赖 |
| 多卡之间 | HCCL collective | 保证通信数据正确 |

同步不足会算错；同步过多会把异步硬件重新串行化。正确性是下限，最小必要依赖是性能目标。

## 9. Tiling 的三个尺度

```text
全局任务切分：多少核 / program，各自负责哪些行或 token
核内 tile：每轮搬入多少元素，是否能放入 UB/L1
指令粒度：Vector/Cube 一次处理的对齐与矩阵基本块
```

Host tiling 常依据 shape、dtype 和硬件资源计算：

- `blockDim`；
- 每核元素数；
- tile 长度与循环次数；
- 尾核/尾 tile 参数；
- workspace 大小；
- kernel 变体或 tiling key。

Triton 把其中较多工作交给编译器和 meta-parameter；Ascend C 往往让开发者显式管理更多层次。

## 10. 算术强度：判断更缺计算还是带宽

算术强度（Arithmetic Intensity）可粗略定义为：

\[
AI = \frac{计算操作数}{从较慢存储搬运的字节数}
\]

向量加法 `z=x+y` 对每个 FP16 元素：

- 读 `x`：2 bytes；
- 读 `y`：2 bytes；
- 写 `z`：2 bytes；
- 计算：1 次加法。

约为 `1 op / 6 bytes`，通常是带宽型。再怎么提高加法单元峰值，也很难绕开数据搬运。

矩阵乘法则会反复复用 A/B tile，同一份数据贡献大量乘加，算术强度高，更可能受 Cube 算力与 tile 利用率影响。

## 11. 一个简单的性能推理流程

```text
1. 算子读写多少字节、做多少计算？
2. 是连续搬运还是离散访问？
3. tile 能否放进 Local Memory？
4. grid/blockDim 是否让所有物理核有工作且不过度下发？
5. CopyIn/Compute/CopyOut 哪一段最长？
6. 能否 double buffer 或融合？
7. 同步是否比必要的更多？
```

先回答这些问题，再调一个看似神秘的 compiler option，通常更有效。

## 12. 常见错误

### 错误一：队列已同步，所以不需要考虑 buffer 生命周期

Queue 只在正确使用 `AllocTensor`、`EnQue`、`DeQue`、`FreeTensor` 的前提下管理生命周期。过早释放或漏释放都会出问题。

### 错误二：Double buffer 一定更快

若只有一个 tile、计算阶段极短，或 UB 因双份 buffer 导致 tile 大幅缩小，收益可能为负。

### 错误三：核越多越快

任务太小会导致每核工作不足，启动、分核和尾块开销占比上升。Triton-Ascend 还特别提醒不要照搬 GPU 的超大 grid。

### 错误四：只优化 Compute

带宽型算子经常是搬运占主导。减少中间 tensor、合并访问或改善连续性可能比替换一条数学指令更重要。

## 13. 本章检查点

- Queue 的两个职责是什么？
- Queue depth 和 double buffer 数量为什么不能混为一谈？
- Pipeline 为什么接近由最慢阶段决定稳态吞吐？
- 向量加法为什么通常是带宽型？
- 同步过少和同步过多分别会发生什么？

## 官方资料

- [Ascend C：TPipe/TQue 流水编程范式](https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_00033.html)
- [Ascend C：TQue 与队列深度](https://www.hiascend.com/document/detail/zh/canncommercial/900/API/ascendcopapi/atlasascendc_api_07_0137.html)
- [Ascend C：InitBuffer 与 Double Buffer](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0110.html)
- [Ascend C：基础 Vector 算子三阶段](https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0033.html)
