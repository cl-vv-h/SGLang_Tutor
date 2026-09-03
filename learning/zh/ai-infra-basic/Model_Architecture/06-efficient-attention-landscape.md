# 高效 Attention 全景：压缩、稀疏、递归与 Kernel

## 1. 为什么需要先画一张 Attention 地图

GQA、MLA、DSA、CSA、KDA、FlashAttention 经常出现在同一张列表里，但它们优化的并不是同一个问题：

- **状态压缩**关心历史信息以什么形式缓存；
- **序列稀疏**关心本轮读取哪些历史位置；
- **递归压缩**关心能否把 token 历史折叠进固定大小的状态；
- **Kernel 优化**关心如何在数学结果不变的情况下减少 HBM 读写。

如果混淆这些维度，就容易得出“MLA 让 prefill 变成线性复杂度”或“FlashAttention 是稀疏 Attention”之类的错误结论。

## 2. 从稠密因果 Attention 出发

设序列长度为 `L`，query/key head dimension 为 `Dk`，value dimension 为 `Dv`：

```text
score[i,j] = q_i dot k_j / sqrt(Dk),  j <= i
output[i] = sum_j softmax(score[i,:])[j] * v_j
```

忽略投影和 head 常数后：

| 阶段 | 核心 Attention 计算量 | 单步 decode 读取的历史状态 |
|---|---:|---:|
| 全序列 prefill/training | `O(L^2 * D)` | 不适用 |
| 单 token decode | `O(L * D)` | `O(L * Dcache)` |

各种高效 Attention 都是在改变这个基线中的一个或多个项。

## 3. 四条相互独立的优化轴

### 3.1 压缩 Head 轴：MQA 与 GQA

MHA 为每个 query head 保存一组 K/V；MQA 只保存一组共享 K/V；GQA 保存若干 K/V group：

```text
MHA 每 token cache: Nq  * (Dk + Dv)
GQA 每 token cache: Nkv * (Dk + Dv), Nkv < Nq
MQA 每 token cache:       Dk + Dv
```

它们缩窄了 cache entry，但 query 仍访问所有可见位置，因此稠密 prefill 对 `L` 仍是二次复杂度。

### 3.2 压缩 Feature 轴：MLA

MLA 不保存逐 head 展开的 K/V，而是保存训练得到的 latent vector 和位置分量：

```text
MLA 每 token cache: Dc + Dr
```

矩阵吸收让 decode 可以直接消费 latent cache。MLA 主要降低 cache 容量与读取带宽，并没有减少需要访问的历史位置数。

### 3.3 减少 Position 集合：SWA、NSA、DSA、CSA

这些方法会限制或变换可见历史：

- **Sliding Window Attention（SWA）**使用固定局部窗口；
- **Native Sparse Attention（NSA）**组合压缩全局 token、被选中的细粒度 block 与局部窗口；
- **DeepSeek Sparse Attention（DSA）**训练轻量 indexer，选择 token 级 MLA entry；
- **Compressed Sparse Attention（CSA）**先把多个原始 token 压成一个 entry，再选 top-k 压缩 entry。

这条路线可以把每个 query 的主 Attention 从访问 `L` 个 entry，降为访问一个有界或缓慢增长的子集。

### 3.4 用递归状态取代显式历史：GDN 与 KDA

GDN 和 KDA 不会在一组历史 token cache 上执行 softmax retrieval，而是把历史折叠进 state matrix：

```text
每请求、每层 state: [num_heads, value_dim, key_dim]
```

状态大小不随上下文长度增长，因此序列处理为线性、运行时状态为固定大小；代价是精确的 token 寻址检索被压缩的关联记忆取代。

## 4. 模型结构与 Kernel 不是一回事

FlashAttention 和 FlashDecoding 通常保持稠密 Attention 的数学结果：

```text
相同的可见 token + 相同的 softmax 结果
不同的 tiling、online softmax 与 HBM 流量
```

PagedAttention 主要改变 cache 分配和地址转换，同样不是一种新的训练期 Attention 架构。

模型结构与 kernel 可以组合：

```text
GQA + FlashAttention
MLA + paged latent cache + FlashMLA
DSA + learned indexer + sparse attention kernel
KDA + chunkwise prefill kernel + fused recurrent decode kernel
```

## 5. 统一对比

设 `k` 为选中 entry 数，`w` 为局部窗口，`m` 为压缩率，`Sstate` 为递归状态大小。下表省略常数与 projection 开销。

| 方法 | 历史表示 | 每 query 访问的 position/entry | Prefill 核心趋势 | Decode 状态增长 |
|---|---|---:|---:|---:|
| MHA | 逐 head 完整 K/V | `L` | `O(L^2)` | `O(L)` |
| GQA/MQA | 分组/共享 K/V | `L` | `O(L^2)` | `O(L)`，斜率更小 |
| MLA | latent KV + position key | `L` | `O(L^2)` | `O(L)`，斜率更小 |
| SWA | 最近窗口 K/V | `w` | `O(L*w)` | 若淘汰旧 entry，则为 `O(w)` |
| NSA | 压缩 entry + 选中 block + window | 取决于设计 | 固定预算时近似线性 | 通常仍随压缩/选择 cache 增长 |
| DSA | MLA entry + indexer key | 主路径 `k`；indexer 扫历史 | 主路径 `O(L*k)` | cache 为 `O(L)`；主 Attention 读取有界 |
| CSA | 压缩 entry + indexer key + 局部 raw tail | 主路径 `k+w`；indexer 扫约 `L/m` | 主路径 `O(L*(k+w))` | 约 `O(L/m)` 加有界局部状态 |
| HCA | 重度压缩 entry + 局部 raw tail | 约 `L/m' + w` | `O(L^2/m')` | 约 `O(L/m')` 加有界局部状态 |
| GDN/KDA | 递归矩阵 + 短卷积状态 | 固定 state | `O(L*Sstate)` | `O(Sstate)` |

DSA/CSA 的 indexer 仍要扫描随上下文增长的历史。“top-k 固定”不代表完整层的单 token decode 严格为 `O(1)`。

## 6. 三类压缩不能互换

### 6.1 Head/Feature 压缩

GQA 和 MLA 仍为每个 token 保留一个 entry，只是缩小 entry 的宽度。它们擅长降低 KV 容量和 decode 带宽，同时保留稠密 token 可见性。

### 6.2 Sequence 压缩

CSA 和 HCA 把若干原始 token 合并为一个 compressed entry。cache 的时间轴更短，但每个 entry 是学习到的摘要，不再是某个原始 token 的精确记录。

### 6.3 Recurrent 压缩

GDN 和 KDA 持续覆写固定大小的状态，不存在以后可任意 gather 的旧 entry 列表。因此 prefix sharing、rollback、speculative verification 与 state transfer 都需要 state-aware 语义，不能直接套用普通 KV page。

## 7. 稀疏 Attention 的三类设计

### 7.1 固定模式

SWA、local-global pattern、dilated pattern 仅根据位置确定连接关系。它们可预测、便于 kernel 优化，但如果没有额外 global path，就无法检索任意远处的内容。

### 7.2 Query-Dependent Retrieval

DSA 为每个 query-history pair 计算轻量分数并选择 top-k entry。选择模式随内容变化，但 indexer 计算、top-k reduction、不规则 gather 和 cache 地址转换会成为系统主路径的一部分。

### 7.3 分层压缩与选择

NSA 和 CSA 先通过压缩缩小搜索空间，再保留细粒度或选择路径；局部窗口负责修复压缩边界可能丢失的近场细节。

NSA 与 DSA 是两种不同的研究机制。本仓库 SGLang 快照中的 backend 字符串 `nsa` 是 `dsa` 的废弃兼容别名；这个别名**不代表两篇论文的算法相同**。

## 8. Prefill 与 Decode 必须分开分析

### Prefill

大量 query 可同时参与计算，稠密 Attention 容易形成高利用率的大矩阵 tile。稀疏结构虽然减少 FLOPs，但 selection、ragged layout、causal mask、backward 支持与负载均衡决定理论收益能否变成端到端加速。

### Decode

每个请求通常只有一个 query，读取历史状态经常比算术更重要。cache entry 宽度、选中 entry 数、分页局部性、反量化和 kernel launch overhead 往往比只看 prefill FLOPs 更关键。

对 KDA/GDN 而言，decode 是递归状态更新，不再扫描历史 cache；主要问题变成 state matrix 带宽与小 batch kernel 效率。

## 9. Serving 系统必须回答的问题

接入任何新 Attention 前，都应先回答：

1. 每请求、每层持久保存什么状态？
2. 状态随 token、block 增长，还是完全不增长？
3. Prefill 与 decode 是同一数学语义的不同调度，还是不同算法路径？
4. Kernel 需要 dense、paged、ragged、top-k 还是 recurrent metadata？
5. Prefix cache 能否直接共享该状态，还是必须重算？
6. Speculative branch 如何复制、验证、提交或回滚状态？
7. CUDA/NPU Graph capture 下哪些长度、page table、selected index、state slot id 必须原地刷新？
8. 量化是训练架构的一部分，还是纯 serving 优化？
9. Tensor/context parallelism 切分的是 head、sequence entry 还是 recurrent state？
10. 性能数据是端到端收益，还是只测 core attention kernel？

## 10. Ascend NPU 优化视角

不要把 CUDA kernel 名称直接翻译成 NPU 实现方案，应按数据流拆分：

| 阶段 | NPU 侧重点 |
|---|---|
| Projection | Q/K/V、latent、gate 与 indexer projection 能否形成 Cube 友好的大矩阵乘？ |
| Compression | projection、gated pooling、norm、RoPE、quant、cache store 能否融合？ |
| Indexing | score 与 top-k 能否避免在 HBM 物化巨大的完整 score tensor？ |
| Sparse core | selected index 是否已经转换为 kernel 所需的物理 paged-cache 地址？ |
| Recurrent core | gate activation、state decay、delta update 与 output read 能否停留在片上存储？ |
| Graph replay | length、page table、selected index、state-slot id 中哪些需要原地刷新？ |

最终必须用 profiler 验证。算法 FLOPs 更低，不代表延迟必然更低：非合并 gather、launch-bound top-k 或 Vector/Cube 串行都可能吞掉收益。

## 11. 推荐阅读顺序

1. 先读 [MHA/MQA/GQA shape](./02-gqa-attention-shapes.md)。
2. 读 [MLA](./04-multi-head-latent-attention.md)，理解 feature 轴上的 cache 压缩。
3. 读 [DSA](./07-deepseek-sparse-attention.md)，理解在 MLA entry 上学习 token selection。
4. 读 [CSA/HCA](./08-compressed-sparse-attention.md)，理解 sequence compression 与 sparse/dense retrieval 的组合。
5. 读 [KDA](./09-kimi-delta-attention.md)和 [GDN 专题](../Gated_Delta_Network/)，理解递归线性 Attention。
6. 读 [Attention Kernel](../Attention_Kernel/)，理解 exact attention 的执行优化。

## 12. 参考资料

- [Multi-Query Attention](https://arxiv.org/abs/1911.02150)
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
- [DeepSeek-V2 Technical Report（MLA）](https://arxiv.org/abs/2405.04434)
- [Native Sparse Attention](https://arxiv.org/abs/2502.11089)
- [DeepSeek-V3.2（DSA）](https://arxiv.org/abs/2512.02556)
- [DeepSeek-V4（CSA/HCA）](https://arxiv.org/abs/2606.19348)
- [Kimi Linear（KDA）](https://arxiv.org/abs/2510.26692)
- [FlashAttention](https://arxiv.org/abs/2205.14135)
