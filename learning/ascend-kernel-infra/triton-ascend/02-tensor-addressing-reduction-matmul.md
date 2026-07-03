# Triton-Ascend 02：地址、广播、归约与矩阵分块

Vector Add 只处理一维连续数据。本章把相同模型扩展到二维 tensor、RMSNorm 和矩阵乘，掌握读 `sgl-kernel-npu` Triton 源码所需的核心语法。

## 1. 二维地址来自 Shape 与 Stride

对 `X[M,N]`，元素 `(i,j)` 的线性地址是：

```python
ptr = x_ptr + i * stride_xm + j * stride_xn
```

Contiguous 行主序通常有：

```text
stride_xm = N
stride_xn = 1
```

Transpose view 可能 shape 相同但 stride 不同。因此 wrapper 要么把 stride 传入 kernel，要么明确要求 contiguous。

## 2. 用 `[:, None]` 和 `[None, :]` 构造地址矩阵

假设一个 program 处理 `BLOCK_M × BLOCK_N` tile：

```python
offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
```

形状变化：

```text
offs_m[:, None] -> [BLOCK_M, 1]
offs_n[None, :] -> [1, BLOCK_N]
broadcast result -> [BLOCK_M, BLOCK_N]
```

这不是创建一个 Python 嵌套列表，而是在 Triton IR 中表达一个 block tensor 地址网格。

## 3. 二维 Mask

```python
mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
x = tl.load(ptrs, mask=mask, other=0.0)
```

每个维度都有自己的边界，最终通过逻辑与组合。复杂 kernel 最好给不同 mask 起名字：

```python
row_mask = offs_m < M
col_mask = offs_n < N
mask = row_mask[:, None] & col_mask[None, :]
```

可读性本身就是正确性工具。

## 4. Reduction：从一块数据归约成较小结果

RMSNorm 对一行 `x[D]` 的核心是：

\[
rstd = \frac{1}{\sqrt{\frac{1}{D}\sum_i x_i^2 + \epsilon}}
\]

Triton 可表达为：

```python
x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
mean_square = tl.sum(x * x, axis=0) / D
rstd = tl.rsqrt(mean_square + eps)
y = x * rstd
```

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

`C[M,N] = A[M,K] @ B[K,N]`：

```python
offs_m = pid_m * BM + tl.arange(0, BM)
offs_n = pid_n * BN + tl.arange(0, BN)
offs_k = tl.arange(0, BK)

a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
acc = tl.zeros((BM, BN), tl.float32)
```

沿 K 维循环：

```python
for k0 in range(0, K, BK):
    a = tl.load(a_ptrs, mask=... , other=0.0)
    b = tl.load(b_ptrs, mask=... , other=0.0)
    acc += tl.dot(a, b)
    a_ptrs += BK * stride_ak
    b_ptrs += BK * stride_bk
```

最后把 `acc` 转成输出 dtype 并写到 C tile。

## 7. Grid 如何覆盖 C

逻辑二维 grid：

```python
grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
```

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

## 10. Tile 选择的约束

对矩阵乘，`BM/BN/BK` 同时影响：

- A/B tile 的搬运字节；
- C accumulator 大小；
- Cube 基本块利用率；
- L1/L0/UB 占用；
- 多核并行 tile 数；
- K 循环次数；
- 尾块比例。

没有脱离 shape 分布的“最佳 BLOCK”。LLM decode 常有很小的 M，prefill 则有较大 M；同一个 config 不一定适合二者。

## 11. 从简单到复杂的练习顺序

1. 1D vector add：掌握 pid、offset、mask；
2. 2D add：掌握 stride 和广播地址；
3. row sum：掌握 reduction；
4. RMSNorm：掌握 FP32 累加和 fusion；
5. matmul：掌握 BM/BN/BK 与 K-loop；
6. fused matmul epilogue：理解 CV fusion 价值；
7. attention：组合 matmul、softmax、mask 与 streaming reduction。

## 12. 本章检查点

- 为什么 shape 相同的两个 tensor 可能需要不同地址计算？
- `offs_m[:,None] + offs_n[None,:]` 在做什么？
- Reduction padding 为什么常用 `other=0`？最大值归约也能无脑用 0 吗？
- `tl.dot` 为什么不能简单等同于一条 Cube 指令？
- Decode 和 prefill 为什么可能选择不同 matmul tile？

## 官方源码与文档

- [Triton-Ascend MatMul Tutorial](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/third_party/ascend/tutorials/03-matrix-multiplication.py)
- [Triton-Ascend LayerNorm Tutorial](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/third_party/ascend/tutorials/05-layer-norm.py)
- [Triton-Ascend Cube 算子开发指南](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/docs/zh/programming_guide/cube_operator.md)
- [Triton-Ascend CV 融合开发指南](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/docs/zh/programming_guide/cv_fusion_operator.md)
