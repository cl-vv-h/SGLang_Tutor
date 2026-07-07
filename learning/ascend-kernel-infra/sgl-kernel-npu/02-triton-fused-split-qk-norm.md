# sgl-kernel-npu 02：Triton 源码精读——Fused Split Q/K Norm

源码：[`norm/fused_split_qk_norm.py`](https://github.com/sgl-project/sgl-kernel-npu/blob/b2378ee05769cf7df209ffc5e1b669728f435a7e/python/sgl_kernel_npu/sgl_kernel_npu/norm/fused_split_qk_norm.py)。这是一个很好的入门案例：只有约百行，却包含 grid、tile、地址、归约、constexpr 分支和 Python wrapper。

## 1. 它解决什么问题

MLA 路径中的融合投影输出可抽象为：

```text
fused row = [Q latent | K latent(nope) | K positional(pe)]
```

Kernel 一次完成：

1. 切出 Q latent 并做 RMSNorm；
2. 切出 K latent 并做 RMSNorm；
3. 切出 K positional 部分并直接复制；
4. 写入三个独立输出。

输入输出 shape：

```text
input       fused_qkv_a_proj_out: [B, total_hidden]
output q    q_lora:               [B, q_lora_rank]
output k    k_nope:               [B, 1, kv_lora_rank]
output k_pe k_pe:                 [B, 1, qk_rope_dim]
```

## 2. 为什么值得融合

非融合逻辑可能是：

```text
split -> Q RMSNorm -> K RMSNorm -> reshape
```

融合后减少 Python/op dispatch 和中间 view/kernel 边界，并让同一 program 按已知 row layout 完成三段处理。它仍要写出三个最终 tensor，不是“零内存流量”。

## 3. Grid：一行一个 Program

Wrapper 使用：

```python
fused_split_qk_norm_kernel[(B,)](...)
```

所以：

```text
grid = (B,)
pid 0 -> 第 0 行
pid 1 -> 第 1 行
...
```

Kernel 中 [`pid = tl.program_id(0)`](https://github.com/sgl-project/sgl-kernel-npu/blob/b2378ee05769cf7df209ffc5e1b669728f435a7e/python/sgl_kernel_npu/sgl_kernel_npu/norm/fused_split_qk_norm.py#L24)；行首地址：

```python
base = pid * total_hidden
```

这里隐含输入第二维连续。Wrapper 取得 shape，但当前源码没有显式检查 input contiguous，调用者契约和上游 tensor layout 因而很重要。

## 4. Q Tile

```python
q_offs = tl.arange(0, q_lora_rank)
q = tl.load(fused_ptr + base + q_offs, ...).to(tl.float32)
```

一个 program 把当前行 Q 段当成一维 tile。`q_lora_rank` 是 `tl.constexpr`，编译器在编译期知道 tile 长度。

地址范围：

```text
[base, base + q_lora_rank)
```

## 5. Q RMSNorm

[`tl.sum`](https://github.com/sgl-project/sgl-kernel-npu/blob/b2378ee05769cf7df209ffc5e1b669728f435a7e/python/sgl_kernel_npu/sgl_kernel_npu/norm/fused_split_qk_norm.py#L38) 对整个 Q tile 归约：

```python
q_var = tl.sum(q * q, axis=0) / q_lora_rank
q_rstd = tl.rsqrt(q_var + eps)
q = q * q_rstd * qw
```

输入先转 FP32，平方、求和和归一化在 FP32 中完成，再由 store 转换到目标输出 dtype。这是 Norm kernel 常见的精度策略。

## 6. 编译期 Bias 分支

```python
if Q_HAS_BIAS:
    qb = tl.load(...)
    q += qb
```

`Q_HAS_BIAS` 是 `tl.constexpr`。有 bias 和无 bias 会形成编译变体；无 bias 版本可在编译期删除整个分支，而不是每行运行一次动态判断。

Wrapper 通过 LayerNorm/RMSNorm 对象是否含非空 `bias` 传入该常量。

## 7. K NOPE Tile

K 段起点：

```python
k_base = base + q_lora_rank
k_offs = tl.arange(0, kv_lora_rank)
```

地址范围：

```text
[base + q_lora_rank,
 base + q_lora_rank + kv_lora_rank)
```

它重复 Q 的 RMSNorm 模式，但使用 K 自己的 weight/bias 与 rank。

## 8. K PE Tile

位置编码段不做 RMSNorm：

```python
pe_base = k_base + kv_lora_rank
pe_offs = tl.arange(0, qk_rope_dim)
k_pe = tl.load(fused_ptr + pe_base + pe_offs, ...)
tl.store(k_pe_ptr + pid * qk_rope_dim + pe_offs, k_pe, ...)
```

三段地址必须严格拼接。理论上应该满足：

\[
total\_hidden = q\_lora\_rank + kv\_lora\_rank + qk\_rope\_dim
\]

当前 wrapper 检查三个 rank 为正，但这份源码中没有显式断言该等式。生产调用依赖模型配置和上游投影保证契约；为独立复用 wrapper 时，补充 shape 断言会更安全。

## 9. Wrapper 的职责

[`fused_split_qk_norm()`](https://github.com/sgl-project/sgl-kernel-npu/blob/b2378ee05769cf7df209ffc5e1b669728f435a7e/python/sgl_kernel_npu/sgl_kernel_npu/norm/fused_split_qk_norm.py#L92) 做了：

- rank 正数检查；
- 读取 `B,total_hidden`；
- 按输入 device/dtype 分配三个输出；
- 取得两组 norm weight/bias；
- 用 `(B,)` grid launch；
- 给 K 输出补回 sequence 维。

这里 `unsqueeze(1)` 只是 view/shape 恢复，不启动另一个数学 kernel。

## 10. 这个 Kernel 的 Tile 在哪里

它没有叫 `BLOCK_SIZE` 的变量，但仍然是 tile kernel：

| 数据段 | Tile 长度 |
|---|---:|
| Q | `q_lora_rank` |
| K NOPE | `kv_lora_rank` |
| K PE | `qk_rope_dim` |

变量名不是判断 tiling 的依据。`tl.arange` 的范围才揭示每个 program 同时处理的 block tensor。

## 11. 性能特征

这是一个 Vector/reduction 型 kernel：

- 读取 fused row 的三段；
- 读取两组 norm weight，可能还有 bias；
- 两次平方和归约与 rsqrt；
- 写三个输出。

潜在限制：

- `B` 很小时 grid 可能无法填满所有 Vector Core；
- rank 增大时，单 program 的 Local Memory 和归约成本增加；
- 不同 rank/bias 组合产生不同编译变体；
- 直接 `(B,)` grid 没有 program 内多行循环，是否最优取决于真实 B 和 compiler/hardware。

## 12. 如何验证

Reference 可写为：

```python
q, k, pe = torch.split(fused, [q_rank, kv_rank, rope_dim], dim=-1)
q_ref = rms_norm(q, q_weight, q_bias, eps)
k_ref = rms_norm(k, k_weight, k_bias, eps)
```

测试矩阵：

- B：1、小 batch、超过物理核数；
- FP16/BF16；
- 有/无 bias；
- 不同模型 rank；
- 极小值、大值、全零；
- 错误 total_hidden 应在 wrapper 层明确失败。

## 13. 从本例学到的通用读法

```text
先找 wrapper 输出 shape
  -> 找 launch grid
  -> 用 pid 判断 program 负责什么
  -> 用 tl.arange 找 tile
  -> 逐段写地址范围
  -> 找 reduction dtype
  -> 区分 constexpr 与运行时分支
  -> 统计 GM 读写和最终输出
```

## 14. 本章检查点与参考答案

### 1. 为什么说一个 program 负责一行，而不是一个 Q 元素？

**答案：**因为 launch grid 是 `(B,)`，`pid` 选择 batch/token 行，而行内元素由 `tl.arange` 组成 block tensor 一次处理。

`pid=2` 时，`base=2*total_hidden` 定位到输入第 2 行。随后 `q_offs=tl.arange(0,q_lora_rank)` 生成这一行 Q 段的全部逻辑 offset，K 与 PE 段同理。没有第二维 program ID 为每个元素创建独立 program。

因此并行层次是“program 间并行处理不同行，program 内以 block tensor 处理一行的三个片段”。编译器会进一步把行内 block 降低到 Vector 指令，但这不改变 program 的逻辑职责。

### 2. 这个 kernel 没有 `BLOCK_SIZE`，为什么仍然有三个 tile？

**答案：**tile 是算法一次处理的数据块，不要求变量必须叫 `BLOCK_SIZE`。

该 kernel 的三个 `tl.arange` 分别定义 Q、K-nope 和 K-PE 的 block shape，长度是 `q_lora_rank`、`kv_lora_rank` 和 `qk_rope_dim`。每组 offsets 对应 fused row 中一个连续片段，也对应一个输出片段。

这里模型 rank 本身充当编译期 tile 参数。阅读 Triton 源码时应寻找 `tl.arange`、block pointer 和 tensor shape，而不是依赖变量命名判断是否存在 tiling。

### 3. Bias 分支为什么没有每行动态判断开销？

**答案：**`Q_HAS_BIAS` 和 `K_HAS_BIAS` 是 `tl.constexpr`，分支在 JIT specialization 时已经确定。

有 bias 的配置会生成包含 bias load/add 的 kernel 变体；无 bias 配置在 IR 优化时删除整段分支。运行每个 pid 时不需要从 device memory 读取布尔值，也不需要每行做动态 branch。

代价是有/无 bias 与不同 rank 组合可能生成多个缓存变体。它用更多编译与缓存项换取更简单的运行时代码。

### 4. FP32 reduction 解决什么问题？

**答案：**它降低 FP16/BF16 平方和累加的舍入误差、下溢/上溢风险，使 RMSNorm 的方差和缩放更稳定。

RMSNorm 要累加一整行的 `x²`。低精度类型尾数较短，逐项累加误差会随 rank 增大；平方还扩大了数值动态范围。源码先将输入转成 FP32，再执行平方、`tl.sum`、除法和 `rsqrt`。

最终 store 可以转换回输入/输出 dtype，因此不是所有中间 tensor 都需要以 FP32 写入 GM。代价是 FP32 临时 block 占用更多片上资源，tile 和性能仍需权衡。

### 5. 哪个 shape 等式是源码调用契约的一部分？

**答案：**输入最后一维应由三个连续片段恰好组成：

\[
total\_hidden=q\_lora\_rank+kv\_lora\_rank+qk\_rope\_dim
\]

Kernel 依据这个顺序计算 `base`、`k_base` 和 `pe_base`。若 total_hidden 更小，会越界读取；若更大，尾部数据被忽略；若片段顺序不同，输出会语义错位。

当前源码主要依赖上游模型配置保证该关系，因此它属于调用契约。若把函数做成更通用的公共 API，wrapper 应显式断言该等式，并检查输入二维且最后一维 contiguous，使错误在 Host 侧尽早暴露。

## 对应源码

- [完整 Triton kernel 与 wrapper](https://github.com/sgl-project/sgl-kernel-npu/blob/b2378ee05769cf7df209ffc5e1b669728f435a7e/python/sgl_kernel_npu/sgl_kernel_npu/norm/fused_split_qk_norm.py)
- [SGLang GLM NPU 端到端导读中的调用位置](../../sglang-ascend-npu/source-code-walkthrough/examples/00-glm-4.7-flash-end-to-end.md)
