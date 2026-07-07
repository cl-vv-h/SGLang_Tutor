# Ascend C 01：Global/Local Tensor、TPipe 与 TQue

Ascend C 把 AI Core 的存储、搬运、计算和同步能力包装成 C/C++ 风格 API。它比 Triton-Ascend 暴露更多硬件细节，也要求开发者承担更多资源与流水设计责任。

## 1. 一个 Ascend C 算子的两面

```text
Host 侧
  ├─ 接收 PyTorch/CANN 调用
  ├─ 检查 shape/dtype/format
  ├─ 计算 tiling 与 blockDim
  ├─ 准备 workspace/stream
  └─ launch kernel

Device 侧
  ├─ 每个核取得 blockIdx
  ├─ 解析 tiling
  ├─ 建立 GlobalTensor
  ├─ 分配 LocalTensor buffer
  ├─ CopyIn / Compute / CopyOut
  └─ 写回 GM
```

初学者常只看 device kernel，但生产算子的一半复杂度可能在 Host tiling、注册和构建。

## 2. Kernel Function

典型入口：

```cpp
extern "C" __global__ __aicore__
void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR tiling) {
    KernelAdd op;
    op.Init(x, y, z, tiling);
    op.Process();
}
```

| 修饰/类型 | 含义 |
|---|---|
| `extern "C"` | 使用稳定的 C linkage 名称，便于 launch/链接 |
| `__global__` | 标记 kernel 入口，可从 Host launch |
| `__aicore__` | 标记在 AI Core device 侧执行 |
| `GM_ADDR` | 指向 Global Memory 的地址参数 |

一份 kernel 代码由多个核以 SPMD 方式执行。`GetBlockIdx()` 决定当前实例的数据范围。

## 3. GlobalTensor

`GlobalTensor<T>` 是对 GM 中 tensor 数据的 device 侧抽象：

```cpp
AscendC::GlobalTensor<half> xGm;
xGm.SetGlobalBuffer((__gm__ half*)x);
```

它不把数据复制到片上。它只是建立类型化的 GM 访问视图，类似“这个地址开始是一段 half 数据”。

可以通过 offset 访问某个分片：

```cpp
AscendC::DataCopy(xLocal, xGm[globalOffset], tileLength);
```

## 4. LocalTensor

`LocalTensor<T>` 表示片上 Local Memory 中的一块数据。Vector/Cube API 的操作数通常是 LocalTensor。

```cpp
AscendC::LocalTensor<half> xLocal = inQueue.AllocTensor<half>();
```

LocalTensor 生命周期要和 buffer/queue 匹配：

```text
AllocTensor -> 填充/计算 -> EnQue
DeQue -> 消费 -> FreeTensor
```

不能把 LocalTensor 当成可无限持有的普通 C++ 对象；它背后占用稀缺片上内存。

## 5. TPosition

`TPosition` 描述 tensor 的逻辑位置：

| 逻辑位置 | 常见物理角色 | 用途 |
|---|---|---|
| `VECIN` | UB | Vector 输入 |
| `VECCALC` | UB | Vector 中间计算 |
| `VECOUT` | UB | Vector 输出 |
| `A1/B1` | L1 | Cube 输入中转 |
| `A2/B2` | L0A/L0B | Cube 指令输入 |
| `CO1` | L0C 等 | Cube 累加结果 |
| `CO2` | 架构相关输出位置 | Cube 输出中转 |

映射随产品架构可能不同，因此代码使用逻辑位置，文档负责给出目标硬件映射。

## 6. TQue

声明一个 Vector 输入队列：

```cpp
AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue;
```

模板参数：

- `VECIN`：队列属于 Vector 输入逻辑位置；
- `1`：队列深度，即最多允许多少次连续入队而不出队。

典型生产者：

```cpp
auto xLocal = inQueue.AllocTensor<half>();
AscendC::DataCopy(xLocal, xGm[offset], length);
inQueue.EnQue(xLocal);
```

典型消费者：

```cpp
auto xLocal = inQueue.DeQue<half>();
// 使用 xLocal 计算
inQueue.FreeTensor(xLocal);
```

`EnQue` / `DeQue` 不只是容器操作，还帮助表达跨流水任务的数据就绪与同步。

## 7. TPipe

```cpp
AscendC::TPipe pipe;
pipe.InitBuffer(inQueue, 2, tileBytes);
```

`TPipe` 为 queue/buffer 分配片上资源并管理事件。上例分配两个 `tileBytes` buffer，常用于 double buffer。

一个 kernel 中所有 buffer 都要纳入 Local Memory 预算：

```text
输入队列数 × buffer 数 × 每块字节
+ 输出队列
+ 临时 TBuf
+ 对齐/padding
<= 可用片上存储
```

## 8. TBuf 与 TQue 的区别

`TBuf` 是不需要在流水阶段间通过 queue 传递的临时 buffer，适合局部 scratch。`TQue` 则用于存在生产者/消费者关系的 tensor，并附带入队/出队同步语义。

粗略判断：

- `CopyIn -> Compute` 要交接数据：用 `TQue`；
- Compute 内部独占的临时 workspace：考虑 `TBuf`；
- 是否原地、是否跨阶段、是否需要 event 会影响最终选择。

## 9. Vector 三阶段骨架

教学缩写版：

```cpp
class KernelAdd {
public:
    __aicore__ void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z) {
        xGm.SetGlobalBuffer((__gm__ half*)x);
        yGm.SetGlobalBuffer((__gm__ half*)y);
        zGm.SetGlobalBuffer((__gm__ half*)z);
        pipe.InitBuffer(inX, 2, TILE_BYTES);
        pipe.InitBuffer(inY, 2, TILE_BYTES);
        pipe.InitBuffer(outZ, 2, TILE_BYTES);
    }

    __aicore__ void Process() {
        for (int tile = 0; tile < tileCount; ++tile) {
            CopyIn(tile);
            Compute(tile);
            CopyOut(tile);
        }
    }
};
```

编译器与 TPipe/TQue 会利用阶段依赖组织流水。代码写成顺序调用，不代表底层搬运和计算一定完全串行。

## 10. CopyIn

```cpp
auto xLocal = inX.AllocTensor<half>();
auto yLocal = inY.AllocTensor<half>();
AscendC::DataCopy(xLocal, xGm[offset], length);
AscendC::DataCopy(yLocal, yGm[offset], length);
inX.EnQue(xLocal);
inY.EnQue(yLocal);
```

这里完成：分配 LocalTensor、GM→Local 搬运、通知 Compute 数据已就绪。

## 11. Compute

```cpp
auto xLocal = inX.DeQue<half>();
auto yLocal = inY.DeQue<half>();
auto zLocal = outZ.AllocTensor<half>();
AscendC::Add(zLocal, xLocal, yLocal, length);
outZ.EnQue(zLocal);
inX.FreeTensor(xLocal);
inY.FreeTensor(yLocal);
```

Vector API 对 LocalTensor 执行。输入消费完后释放，输出入队等待 CopyOut。

## 12. CopyOut

```cpp
auto zLocal = outZ.DeQue<half>();
AscendC::DataCopy(zGm[offset], zLocal, length);
outZ.FreeTensor(zLocal);
```

Local→GM 搬运完成后释放输出 buffer。

## 13. `Process()` 循环中的两个编号

```text
blockIdx：当前物理/逻辑核实例负责哪一大片
tileIdx ：当前核正在处理自己大片中的哪一小块
```

全局 offset 常类似：

```cpp
globalOffset = blockIdx * elementsPerCore + tileIdx * tileLength;
```

尾核和尾 tile 需要 Host tiling 或 device 逻辑给出实际长度。

## 14. Ascend C 的优势与代价

| 方面 | 特点 |
|---|---|
| 硬件控制 | 能显式管理存储位置、搬运、队列、事件、AIC/AIV 协作 |
| 性能上限 | 适合需要精细流水和专用数据通路的 kernel |
| 开发量 | Host、tiling、kernel、注册、构建、测试都可能需要编写 |
| 可维护性 | 模板、硬件分支和同步增加阅读难度 |
| 版本耦合 | 需要匹配 CANN、编译器和目标架构 |
| 适用场景 | 复杂融合、专用 layout、极致性能、Triton 当前难以表达的能力 |

## 15. 本章检查点与参考答案

### 1. `GlobalTensor` 是否意味着数据已经在 UB？

**答案：**不意味着。`GlobalTensor<T>` 只是 device 侧对 GM 地址建立的类型化视图。

`SetGlobalBuffer()` 告诉 kernel：“从这个全局地址开始，把内容解释成 T 类型，并按给定长度/offset 访问。”它不会触发 GM→UB 搬运，也不会占用 TPipe 管理的 Local Memory。

只有执行 `DataCopy(local, global[offset], length)` 等操作后，相应数据才进入 LocalTensor 所在的 UB/L1/L0 路径。把“建立地址视图”和“实际搬数据”分开理解，是阅读 Init 与 CopyIn 的第一步。

### 2. `LocalTensor` 的生命周期由哪些操作构成？

**答案：**典型 Queue 管理生命周期是 `AllocTensor → 生产数据 → EnQue → DeQue → 消费数据 → FreeTensor`。

CopyIn 先从输入 Queue 分配 LocalTensor，把 GM 数据搬入，再 EnQue 表示输入就绪；Compute DeQue 输入，分配输出 LocalTensor，完成计算后 EnQue 输出，并释放已经消费完的输入；CopyOut DeQue 输出，写回 GM 后释放输出。

关键是释放时机必须落在最后一个消费者之后。过早 Free 会让 buffer 被下一 tile 复用并覆盖仍在使用的数据；忘记 Free 则资源无法循环使用，可能造成 buffer 枯竭或流水停顿。

### 3. `TPosition` 为什么是逻辑概念？

**答案：**它描述 tensor 在计算流水中的角色，而不是直接承诺某个产品上的唯一物理存储实现。

例如 `VECIN` 表示 Vector 输入位置，`A1/B1` 表示 Cube 输入的第一级逻辑位置。编译器依据目标架构把它们映射到 UB、L1 等实际资源和合法数据通路。

这样同一种 API 能适配耦合/分离架构及不同产品，但开发者做性能预算时仍需查目标架构的映射表。逻辑抽象提高可移植性，物理规格决定容量和性能，两者不能混为一谈。

### 4. `TQue` 比普通 buffer 多了什么？

**答案：**TQue 同时增加了资源周转和生产者—消费者同步语义。

普通 buffer 只是一段地址，开发者要自行保证谁可以写、谁可以读、何时复用。TQue 通过 Alloc/Free 管理 LocalTensor buffer，通过 EnQue/DeQue 表达数据何时就绪以及哪个流水阶段可以消费，并借助 TPipe 管理相关 event。

这让按顺序写出的 `CopyIn(); Compute(); CopyOut();` 有机会被编译器组织成跨 tile 流水。代价是必须遵守 Queue 生命周期；它不是可以随意随机访问和永久保存数据的容器。

### 5. `TPipe.InitBuffer(..., 2, ...)` 中的 2 与队列模板深度有何区别？

**答案：**`2` 是分配给该 Queue 的 buffer number，常用于 ping/pong 双缓冲；模板深度是允许连续入队而未出队的数量。

例如 `TQue<VECIN,1>` 配合 `InitBuffer(queue,2,tileBytes)`：Queue depth 仍为 1，但底层有两块内存可在相邻 tile 间轮换，使 MTE 搬下一块时 Vector 计算上一块。

如果写成 `TQue<VECIN,2>`，它表达允许两份数据排队，不自动保证已经分配两块适合双缓冲的 buffer。两者服务于不同问题：depth 是队列协议，buffer number 是存储资源配置。

### 6. `blockIdx` 和 `tileIdx` 分别切哪一层任务？

**答案：**`blockIdx` 选择当前核实例负责的全局大分片，`tileIdx` 选择该实例内部正在处理的片上小块。

Host 通过 `blockDim` 启动多个实例；每个实例用 `GetBlockIdx()` 得到 blockIdx，例如分别负责不同 row。由于一整 row 可能仍放不进 UB，该实例再在 `Process()` 中沿 vocab/hidden 维循环 tileIdx。

常见全局地址为 `blockOffset(blockIdx) + tileIdx*tileLength`。前者决定多核并行和负载均衡，后者决定 Local Memory 占用、搬运粒度和流水，两级 tiling 的调优目标不同。

## 官方资料

- [Ascend C：编程 API](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/developmentguide/opdevg/Ascendcopdevg/atlas_ascendc_10_0011.html)
- [Ascend C：基础 Vector 算子](https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0033.html)
- [Ascend C：TQue](https://www.hiascend.com/document/detail/zh/canncommercial/900/API/ascendcopapi/atlasascendc_api_07_0137.html)
- [Ascend C：逻辑位置与物理存储](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0004.html)
