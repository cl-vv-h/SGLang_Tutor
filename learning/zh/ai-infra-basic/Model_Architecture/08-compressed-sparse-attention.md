# 压缩稀疏注意力（CSA）与分层压缩注意力（HCA）

压缩稀疏注意力（Compressed Sparse Attention，CSA）是 DeepSeek-V4 引入的长上下文注意力设计。它的核心动作可以概括为：

> 先压缩序列，再从压缩后的条目中检索少量候选，交给昂贵的注意力主核。

分层压缩注意力（Hierarchical Compressed Attention，HCA）提供一条压缩率更高的稠密路径；局部滑动窗口分支则保留最近 token 的细节。在模型的 layer 排列中，这些机制共同覆盖三种时间尺度：

- **SWA**：精确、未压缩的近期上下文；
- **CSA**：中等压缩、按内容选择的上下文；
- **HCA**：重度压缩、始终可见的全局上下文。

本章重点解释这种层次结构对执行与服务系统的影响。这里的 **CSA** 特指 DeepSeek-V4 的 Compressed Sparse Attention，而不是该缩写的其他含义。

## 1. 为什么先压缩再选择？

DeepSeek Sparse Attention（DSA）选择 token 级潜在 KV 条目。其注意力主核是稀疏的，但轻量 indexer 仍需扫描 token 历史。

CSA 把候选空间从 `L` 个 token 条目缩小为约 `L / m` 个压缩条目，其中 `m` 是压缩步长。若主核检索 `k` 个压缩条目，则：

| 组件 | DSA | CSA |
|---|---:|---:|
| 每个 query 的 indexer 候选数 | `L` | 约 `L / m` |
| 每个 query 的主核条目数 | `k` | `k` 个压缩条目 |
| 长程主缓存粒度 | token | 压缩块 |

压缩并非没有代价：每个压缩条目概括多个源位置，因此模型必须学习哪些信息需要保留。局部分支与分层分支用于补偿不同类型的信息损失。

## 2. 符号与形状账本

设：

- `X in R[L, d_model]`：隐藏状态；
- `m`：CSA 压缩步长；
- `m'`：HCA 压缩步长，通常远大于 `m`；
- `Nc = ceil(L / m)`：CSA 条目数；
- `Nh = ceil(L / m')`：HCA 条目数；
- `C in R[Nc, d_c]`：CSA 压缩 KV 条目；
- `K_I in R[Nc, d_i]`：压缩 indexer key；
- `k`：每个 query 选择的 CSA 条目数；
- `w`：未压缩滑动窗口长度。

请区分 `K_I` 与 `k`：前者是 key 张量，后者是 top-k 预算。

## 3. 重叠式序列压缩

CSA 并非简单地对每 `m` 个 token 求平均。它构造两条内容投影流与两条可学习门控流：

```text
C_a = X W_a^C       Z_a = X W_a^Z
C_b = X W_b^C       Z_b = X W_b^Z
```

对于压缩槽位 `i`，压缩器消费两个相邻源块：

```text
前一个块：X[(i-1)m : i*m]
当前块：  X[i*m : (i+1)m]
```

`a`、`b` 两条路径经过排列，使该槽位最多接收 `2m` 个源位置。对门控 logit 做 softmax 后得到按特征学习的混合权重。概念上：

```text
C_i = 对 2m 个源位置 j 求和：softmax(Z_i)[j] * projected_content[j]
```

实际实现会批量执行并重排这些运算，但接口契约由三点定义：

1. 输出步长是 `m`，缓存长度约为 `L / m`；
2. 相邻输出覆盖的源位置有重叠；
3. 增量解码时，不完整块需要持久化压缩器状态。

重叠能缓解块边界问题：一个块边缘附近的证据可以通过相邻路径参与压缩。

## 4. 压缩后的 Indexer

CSA 在压缩条目上执行检索，而非在原始 token 位置上检索。其 indexer 沿用 DSA 的高层模式：

```text
c_q = x_t W_DQ
q_I = c_q W_UQ^I
score(t, s) = sum_j w_j^I * ReLU(q^I_{t,j} dot k^I_s)
I_t = TopK_s(score(t, s), k)
```

区别在于 `s` 的取值域：它遍历 `Nc` 个压缩条目。因此，indexer key 需要独立的压缩缓存，并与主压缩 KV 缓存保持同步。

选出的 ID 是**压缩条目 ID**，不能把它解释为 token ID 或普通分页 KV 槽位。

## 5. 稀疏注意力主核

选中的压缩条目进入类似 Multi-Query Attention 的主核。DeepSeek-V4 将压缩表示同时用作 key 和 value，并通过分组输出投影恢复各个 head 的表达能力。

简化的 query 路径为：

```text
Q_t = x_t W^Q
selected = C[I_t]
A_t = softmax(Q_t selected^T + mask)
O_t = GroupedOutput(A_t selected)
```

实际实现还包含归一化、位置处理、head 分组和融合 kernel。需要明确概念边界：

- **indexer** 对候选排序；
- top-k 返回压缩逻辑 ID；
- **主核**在这些条目上重新计算注意力 logit 与 softmax。

## 6. HCA：稠密的全局安全网

HCA 以更大的步长 `m'` 压缩序列，并对得到的短序列执行稠密注意力：

```text
X[0:L] -> 非重叠重度压缩 -> H[0:Nh]
query -> 对 H 做稠密注意力
```

与 CSA 不同，HCA 不需要检索 indexer，因为它的序列已经足够短，可以全量可见。HCA 适合承载宽泛的全局信号，CSA 则为选中的远距离区域保留更多细节。

## 7. 局部 SWA 分支

压缩必然丢失 token 级细节。滑动窗口注意力（SWA）让最近 `w` 个 token 保持未压缩：

```text
context(t) = [t-w+1, ..., t]
```

该分支负责精确的局部语法、近期工具输出与近场依赖，无须要求压缩器保留每个微小差异。

每个压缩注意力 layer 都将局部路径与该层配置的长程分支配对：CSA 或 HCA。模型会交错排列不同 layer 类型，并不意味着每个 block 都同时执行 CSA 与 HCA。它们是互补且经过训练的路径，不是推理时可随意互换的选项。

## 8. 位置信息与 Attention Sink

序列压缩让位置处理比普通 MHA 更微妙。DeepSeek-V4 使用部分旋转位置编码（partial RoPE）：只有表示中指定的一部分维度携带旋转位置；输出路径按架构要求执行对应的逆旋转。

这是实现契约，而非无关紧要的变换。对所有通道应用 RoPE 或遗漏输出侧处理，都会改变模型函数。

该架构还包含 attention sink，为 query 提供稳定的兜底目标。稀疏与压缩 kernel 必须保留 sink 的 mask 与归一化语义。

## 9. 示例配置，不是通用常量

公开的 DeepSeek-V4 Flash 配置给出了以下示例值：

| 参数 | 示例值 |
|---|---:|
| CSA 压缩步长 `m` | 4 |
| CSA top-k `k` | 512 |
| HCA 压缩步长 `m'` | 128 |
| 局部窗口 `w` | 128 |

DeepSeek-V4 Pro 将 CSA 选择预算提高到 1024。这些值属于 checkpoint 架构字段，不能把它们当成纯服务参数静默调节。

## 10. 复杂度与内存

对历史长度为 `L` 的一个 decode query，不同配置 layer 的工作量可近似为：

```text
CSA layer：O(L / m) indexer + O(k) 主核 + O(w) 局部路径
HCA layer：O(L / m') 稠密主核 + O(w) 局部路径
```

从整个模型看，缓存包含多种表示；每个 layer 只持有其配置类型所需的子集：

```text
CSA 主缓存          ~ O(L / m)
CSA indexer-key 缓存 ~ O(L / m)
HCA 缓存            ~ O(L / m')
局部未压缩 KV        ~ 每个序列 O(w) 个活跃条目
压缩器尾部状态        ~ O(m + m')
```

真正的收益取决于字节数，而不只是条目数。需要记录每个池的 dtype、特征宽度、scale/元数据开销、对齐与页面碎片。

## 11. Prefill 与 Decode 是两套程序

### Prefill

Prefill 可并行压缩多个块，并批量处理 indexer query。风险包括临时张量、top-k 工作区、最后一个不完整块内部的因果正确性，以及进入稀疏主核的不规则 gather。

### Decode

Decode 每次追加一个 token。多数 step 只更新尚未完成的压缩块；当一个块完成时，才发射新的压缩缓存条目。因此 runtime 必须在 step 间携带部分内容与门控状态。

图捕获需要为以下数据提供有界缓冲区：

- 压缩缓存地址；
- indexer top-k 输出；
- 部分压缩器状态；
- 局部窗口元数据；
- 每个 request 的 token 长度与压缩长度。

重新分配或依赖 shape 的 host 工作会抵消图模式收益。

## 12. 异构缓存管理

CSA、HCA 与 SWA 的推进速度不同。把它们当成一个普通 KV 池会带来正确性与碎片问题。

健壮的 request 状态需要记录：

```text
token_length
csa_complete_length 与 csa_tail_state
hca_complete_length 与 hca_tail_state
local_window 映射
csa indexer/main-cache 对齐关系
```

page 与 block 大小应考虑压缩步长。如果物理分配单元无法自然对齐，就要定义显式的跨步携带规则，而不是把不完整数据直接取整丢弃。

前缀缓存必须在相互一致的边界上快照所有表示。只复制已完成的压缩条目而丢失压缩器尾部状态，会改变后续输出。

## 13. SGLang 源码地图

在本教程对应的源码快照中，相关路径为：

- 模型编排：[`deepseek_v4.py`](../../../../python/sglang/srt/models/deepseek_v4.py)；
- 架构字段：[`deepseek_v4.py`](../../../../python/sglang/srt/configs/deepseek_v4.py)；
- 压缩：[`compressor.py`](../../../../python/sglang/srt/layers/attention/dsv4/compressor.py)；
- 压缩检索：[`indexer.py`](../../../../python/sglang/srt/layers/attention/dsv4/indexer.py)；
- 注意力调度：[`deepseek_v4_backend.py`](../../../../python/sglang/srt/layers/attention/deepseek_v4_backend.py)；
- 缓存池：[`deepseek_v4_memory_pool.py`](../../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py)；
- 部分块状态：[`deepseek_v4_compress_state.py`](../../../../python/sglang/srt/mem_cache/deepseek_v4_compress_state.py)。

建议按上述顺序阅读：配置决定形状，模型决定数据流，backend/cache 文件决定服务契约。

## 14. Ascend NPU 优化视角

当前源码快照中的 DeepSeek-V4 专用路径主要围绕 CUDA/HIP/Triton 构建。存在通用 PyTorch 类，并不等于端到端 Ascend 路径已经生产可用。部署前必须核对固定版本的 SGLang、`torch_npu`、CANN 与 kernel 支持。

将该架构适配到 Ascend 时，应分别 profile：

1. 投影与重叠式压缩；
2. 压缩 indexer 与 top-k；
3. 压缩缓存从逻辑地址到物理地址的转换；
4. 稀疏 gather 与注意力主核；
5. HCA 稠密注意力；
6. 局部 SWA 与最终分支组合。

合适的融合候选包括：投影+归一化、压缩器门控+softmax+reduce、index score+mask，以及 gather+attention。减少 kernel launch 之前，必须先保留尾部状态与因果语义。

## 15. DSA、CSA 与 HCA 对比

| 属性 | DSA | CSA | HCA |
|---|---|---|---|
| 长程存储单元 | token 级潜在条目 | 中度压缩块 | 重度压缩块 |
| 检索方式 | top-k | top-k | 无；对短序列稠密计算 |
| 候选数 | `L` | `L / m` | `L / m'` |
| 主 query 工作量 | `k` | `k` | `L / m'` |
| 主要角色 | 选中的精细信息 | 选中的中分辨率历史 | 粗粒度全局覆盖 |

## 16. 常见误解

1. **“CSA 只是 KV 量化。”** CSA 学习序列压缩；量化改变的是数值表示。
2. **“一个压缩条目严格对应一个不重叠块。”** CSA 的源位置覆盖存在重叠。
3. **“top-k 返回 token 索引。”** 它返回压缩条目索引。
4. **“HCA 是另一条稀疏分支。”** HCA 对重度压缩后的短序列执行稠密注意力。
5. **“前缀复用只保存已完成压缩缓存即可。”** 部分压缩器状态也是 request 状态的一部分。
6. **“示例压缩率可以在运行时自由修改。”** 它们与 checkpoint 权重和架构耦合。

## 17. 正确性与性能检查清单

1. 压缩器的边界与 padding 行为和 checkpoint 实现一致。
2. CSA 主条目与 indexer key 在同一逻辑 step 发射。
3. 压缩 top-k ID 通过正确的 request 专属缓存表完成映射。
4. 部分块在 decode、前缀缓存、迁移与推测解码回滚时都不会丢失。
5. SWA 边界满足因果性与 request 隔离。
6. partial RoPE、逆旋转与 sink 语义和参考实现一致。
7. 内存核算包含所有缓存池、scale、元数据与碎片。
8. profile 分开统计压缩、检索、稀疏主核、HCA、SWA 与图开销。

## 18. 参考资料

- [DeepSeek-V4: Advancing Open-Source Intelligence](https://arxiv.org/abs/2606.19348)
- [本仓库的 DSA 教程](./07-deepseek-sparse-attention.md)
- [高效注意力技术地图](./06-efficient-attention-landscape.md)
