# Ascend C 02：一个 Add 算子的端到端工程

本章不追求复制某个 CANN 模板的全部样板代码，而是解释一个生产算子从 Python/PyTorch 调用到 device kernel 的每个边界。

## 1. 先写规格，不要先写 Kernel

```text
语义：z = x + y
输入：x, y
输出：z
shape：相同的一维或展平 tensor
dtype：先支持 FP16
layout：ND contiguous
边界：允许总元素数不整除核数和 tile 长度
```

规格会决定测试、tiling 和注册 schema。没有规格，kernel 很容易只对作者手里的一个 shape 正确。

## 2. 端到端文件分工

一个典型工程可抽象为：

```text
add/
├── op_host/
│   ├── add.cpp              # PyTorch/CANN Host API、校验、launch
│   └── tiling_data.h        # Host 与 Device 共享的 tiling 结构
├── op_kernel/
│   └── add_kernel.cpp       # Ascend C device kernel
└── tests/
    ├── test_add.py          # 正确性
    └── bench_add.py         # 性能
```

具体仓库的生成工具和目录名可能不同，但职责基本稳定。

## 3. Tiling Data 是 Host/Device 协议

教学版结构：

```cpp
struct AddTilingData {
    uint32_t totalLength;
    uint32_t blockLength;
    uint32_t tileLength;
    uint32_t tilesPerBlock;
    uint32_t lastBlockLength;
};
```

Host 负责填写，Device 负责读取。字段一旦改动，两侧必须同步，否则最危险的情况不是编译失败，而是按错误偏移解释内存。

## 4. Host 侧选择 BlockDim

一个简单策略：

```text
numBlocks = min(物理 Vector Core 数, ceil(totalLength / targetPerCore))
base = totalLength // numBlocks
remainder = totalLength % numBlocks
```

前 `remainder` 个核可多处理一个对齐块，或专门给出尾核参数。

BlockDim 太小会浪费核；太大则每核工作不足、启动与尾块开销上升。

## 5. Host 侧选择 TileLength

Tile 至少容纳：

```text
xLocal + yLocal + zLocal
```

若开启 double buffer，则近似是两份：

```text
2 × (x tile + y tile + z tile) + 其他临时空间 <= UB budget
```

还要让 tile 字节满足搬运对齐，并处理最后不满 tile 的元素。

## 6. Device 侧初始化

教学伪代码：

```cpp
void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const AddTilingData& t) {
    blockIdx = AscendC::GetBlockIdx();
    blockLength = IsLastBlock(blockIdx) ? t.lastBlockLength : t.blockLength;
    blockOffset = ComputeBlockOffset(blockIdx, t);

    xGm.SetGlobalBuffer((__gm__ half*)x + blockOffset, blockLength);
    yGm.SetGlobalBuffer((__gm__ half*)y + blockOffset, blockLength);
    zGm.SetGlobalBuffer((__gm__ half*)z + blockOffset, blockLength);

    pipe.InitBuffer(inX, 2, t.tileLength * sizeof(half));
    pipe.InitBuffer(inY, 2, t.tileLength * sizeof(half));
    pipe.InitBuffer(outZ, 2, t.tileLength * sizeof(half));
}
```

重点：每个核把 GlobalTensor 视图定位到自己的分片，后续 tile offset 可以从 0 开始计算。

## 7. Device 侧 Process

```cpp
void Process() {
    uint32_t tileCount = CeilDiv(blockLength, tileLength);
    for (uint32_t tile = 0; tile < tileCount; ++tile) {
        uint32_t actual = Min(tileLength, blockLength - tile * tileLength);
        CopyIn(tile, actual);
        Compute(actual);
        CopyOut(tile, actual);
    }
}
```

`actual` 是尾 tile 的真实长度。实际 API 可能要求对齐搬运并对尾块采用特殊参数，不能把所有非对齐场景都简化成普通 `DataCopy(..., actual)`。

## 8. Kernel 入口

```cpp
extern "C" __global__ __aicore__
void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR tilingGm) {
    AddTilingData tiling;
    LoadTiling(tiling, tilingGm);
    KernelAdd op;
    op.Init(x, y, z, tiling);
    op.Process();
}
```

入口函数尽量薄，具体逻辑放在 kernel class 中，便于模板化和测试不同实现。

## 9. Host Launch

Host 侧最终需要在当前 NPU stream 上 launch：

```text
add_custom<<<blockDim, l2ctrl, stream>>>(x, y, z, tiling)
```

在实际工程中常由 CANN 生成的 launch stub 或宏封装，例如 `EXEC_KERNEL_CMD(...)`。核心参数仍是：kernel、blockDim、stream、输入输出、tiling/workspace。

## 10. PyTorch 注册

使用 PyTorch C++ extension 时，通常有 schema 与 backend 实现两步：

```cpp
TORCH_LIBRARY_FRAGMENT(npu, m) {
    m.def("add_custom(Tensor x, Tensor y) -> Tensor");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) {
    m.impl("add_custom", TORCH_FN(add_custom_host));
}
```

随后 Python 可调用：

```python
torch.ops.npu.add_custom(x, y)
```

`PrivateUse1` 是 PyTorch 为外部设备后端保留的 dispatch key；Ascend PyTorch 后端使用它接入 NPU tensor。

## 11. Shared Library 如何被加载

编译生成 `.so` 后，需要在 Python 进程中加载，注册代码才会运行：

```python
torch.ops.load_library("libcustom_ops.so")
```

如果 `.so` 未加载，`torch.ops.npu.add_custom` 不存在；如果 schema 已注册但 backend 实现缺失，会在 dispatch 时失败。这两个错误属于不同层。

## 12. 构建要连接哪些世界

构建通常同时需要：

- PyTorch C++ headers/library；
- `torch_npu` headers/library；
- Ascend C compiler 与 CANN headers；
- `ascendcl` runtime；
- tiling/platform/register 等库；
- Host C++ 与 Device kernel target。

这也是 Ascend C 算子比纯 Python Triton kernel 工程更重的原因。

## 13. 正确性测试矩阵

| 维度 | 样例 |
|---|---|
| 总长度 | 0、1、31、32、33、tile±1、大 tensor |
| dtype | 每个声明支持的 dtype |
| 对齐 | 整 32B 与非整 32B |
| 多核 | 少于核数、刚好、多轮 tile、尾核 |
| 数值 | 0、负数、极值、NaN/Inf 语义 |
| layout | contiguous；不支持的 view 应明确拒绝 |

Reference 用 `torch.add`，并在每次修改 tiling 后重新跑完整矩阵。

## 14. 性能测试

Vector Add 是带宽型。报告：

\[
effective\ bandwidth = \frac{bytes(x)+bytes(y)+bytes(z)}{kernel\ time}
\]

同时比较：

- `torch.add` / `torch_npu` reference；
- 单 kernel latency；
- 不同长度下的 GB/s；
- 是否包含首次加载/JIT/初始化；
- 是否正确同步 stream。

## 15. 本章检查点

- Tiling data 为什么是一份 ABI/协议？
- Host 为什么比 Device 更适合依据 shape 选择 blockDim？
- `.so` 加载、schema 注册、backend 实现分别在哪一步？
- `torch.ops.npu.*` 为什么不能单凭 namespace 判断源码归属？
- 尾 tile 为什么不仅是一个 `min()` 就一定处理正确？

## 官方资料

- [Ascend C Add 自定义算子教程](https://www.hiascend.com/document/detail/zh/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0006.html)
- [Ascend C 多核 Tiling](https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_10005.html)
- [Ascend C 核函数](https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/developmentguide/opdevg/Ascendcopdevg/atlas_ascendc_10_0014.html)
- [PyTorch Custom Operators Manual](https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html)
