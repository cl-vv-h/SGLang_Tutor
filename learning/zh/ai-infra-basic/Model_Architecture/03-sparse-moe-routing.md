# Sparse MoE：Router、Top-K、Expert 与数据重排

## 1. MoE 替换了什么

Dense Transformer 的 FFN 对每个 token 使用同一组 MLP 权重：

```math
Y = \mathrm{FFN}\left(X\right), X,Y: \left[T,H\right]
```

Sparse Mixture-of-Experts 准备 `E` 组 expert MLP，但每个 token 只选择 `K` 个：

```math
Y_{t} = \sum_{e \in \mathrm{TopK}\left(t\right)} p_{t,e} \cdot  \mathrm{Expert\_e}\left(X_{t}\right)
```

当 `K << E` 时，模型可以拥有较大的总参数量，而单 token 只执行少数 experts。

![Sparse MoE 路由数据流](./assets/moe-routing-flow.svg)

## 2. MoE 主路径形状

```text
input X: [T,H]
router logits: [T,E]
topk ids: [T,K]
topk weights: [T,K]
```

```math
\text{dispatched rows}: \left[R,H\right], R = T\cdot K
```

```text
expert outputs: [R,H]
combined output Y: [T,H]
```

主路径输入输出相同，但内部把一个 token 复制成 `K` 条 route。这里的复制是逻辑路由；高性能实现可能通过索引、排序和 fused kernel 避免普通框架中的显式复制。

## 3. Router Linear

Router 是从 hidden space 到 expert space 的线性层：

```math
\mathrm{shape}\left(\mathrm{Wr}\right)=\left[H,E\right]
```

```math
G=X\,\mathrm{Wr}
```

```math
\mathrm{shape}\left(G\right)=\left[T,E\right]
```

`G[t,e]` 表示 token `t` 分配给 expert `e` 的原始分数。其核心计算为：

```math
\mathrm{router\_logits}=\mathrm{hidden\_states}\,\mathrm{router\_weight}
```

Router 参数相对于所有 expert 参数通常较小，复制在 ranks 上可以避免先为路由分数做切分通信。

## 4. Top-K 选择与归一化

对每个 token 独立选择分数最大的 `K` 个 expert：

```math
\mathrm{topk\_ids}\left[t,:\right] = \mathrm{indices\_of\_top\_k}\left(G\left[t,:\right]\right)
```

选中分数转换为权重。常见形式：

```math
{p}_{t,e}={\mathrm{softmax}\left({G}_{t,:}\right)}_{e}
```

或只在选中的 K 项上归一化：

```math
w\left[t,j\right] = \mathrm{exp}\left(g\left[t,j\right]\right) / \sum_{u \in \mathrm{TopK}\left(t\right)} \mathrm{exp}\left(g\left[t,u\right]\right)
```

结果：

```text
topk_ids: [T,K], integer
topk_weights: [T,K], floating point
```

Top-K 输出至少包含 expert ids 和对应权重；实现还可附带 token index、expert offset 和路由排序信息。

## 5. Dispatch：从 token 顺序变成 expert 顺序

输入按 token 排列：

```text
X = [x0, x1, x2, ..., x_(T-1)]
```

expert kernel 更适合按 expert 分组：

```text
expert 0: [token routes assigned to expert 0]
expert 1: [token routes assigned to expert 1]
...
```

Dispatch 为每个 `(token, selected_expert)` 建立 route：

```math
\text{route count R}= T\cdot K
```

典型 metadata：

| 变量 | 形状 | 含义 |
|---|---:|---|
| `route_token_id` | $`\left[R\right]`$ | 每条 route 来自哪个 token |
| `route_expert_id` | $`\left[R\right]`$ | 每条 route 去哪个 expert |
| `route_weight` | $`\left[R\right]`$ | combine 时使用的权重 |
| `permuted_input` | $`\left[R,H\right]`$ | 按 expert 分组的 token rows |
| `expert_offsets` | $`\left[E+1\right]`$ | 每个 expert 在 route buffer 中的区间 |

对 expert `e`：

```math
r_{e}={\mathrm{expert\_offsets}}_{e+1}-{\mathrm{expert\_offsets}}_{e}
```

```math
\mathrm{shape}\left(X_{e}\right)=\left[r_{e},H\right]
```

```math
\text{sum\_e r\_e}= R
```

## 6. 一个具体路由例子

设 $`T=4,\qquad E=4,\qquad K=2`$：

| token | Top-2 expert ids | weights |
|---|---|---|
| `x0` | $`\left[1,3\right]`$ | `[0.7,0.3]` |
| `x1` | $`\left[0,1\right]`$ | `[0.6,0.4]` |
| `x2` | $`\left[3,1\right]`$ | `[0.8,0.2]` |
| `x3` | $`\left[0,2\right]`$ | `[0.55,0.45]` |

$`R=T\cdot K=8`$。按 expert 排序后：

```text
expert 0 <- x1, x3       r_0=2
expert 1 <- x0, x1, x2   r_1=3
expert 2 <- x3           r_2=1
expert 3 <- x0, x2       r_3=2
```

expert 完成计算后，combine 恢复 token 顺序：

```math
\mathrm{y0}=0.7\cdot \mathrm{E1}\left(\mathrm{x0}\right)+0.3\cdot \mathrm{E3}\left(\mathrm{x0}\right)
```

```math
\mathrm{y1}=0.6\cdot \mathrm{E0}\left(\mathrm{x1}\right)+0.4\cdot \mathrm{E1}\left(\mathrm{x1}\right)
```

```math
\mathrm{y2}=0.8\cdot \mathrm{E3}\left(\mathrm{x2}\right)+0.2\cdot \mathrm{E1}\left(\mathrm{x2}\right)
```

```math
\mathrm{y3}=0.55\cdot \mathrm{E0}\left(\mathrm{x3}\right)+0.45\cdot \mathrm{E2}\left(\mathrm{x3}\right)
```

最终 $`Y=\left[\mathrm{y0},\mathrm{y1},\mathrm{y2},\mathrm{y3}\right]`$，形状恢复为 $`\left[4,H\right]`$。

## 7. 单个 Expert 的 SwiGLU

现代 MoE 常使用门控 SwiGLU expert。对 `X_e [r_e,H]`：

```math
\mathrm{gate}=X_{e}\,W_{\mathrm{gate}}\qquad\mathrm{shape}:\left[r_{e},\mathrm{Ie}\right]
```

```math
\mathrm{up}=X_{e}\,W_{\mathrm{up}}\qquad\mathrm{shape}:\left[r_{e},\mathrm{Ie}\right]
```

```math
\mathrm{mid}=\mathrm{SiLU}\left(\mathrm{gate}\right)\odot \mathrm{up}\qquad\mathrm{shape}:\left[r_{e},\mathrm{Ie}\right]
```

```math
\mathrm{out}=\mathrm{mid}\,W_{\mathrm{down}}\qquad\mathrm{shape}:\left[r_{e},H\right]
```

权重形状：

```math
\mathrm{shape}\left(W_{\mathrm{gate}}\right)=\left[H,\mathrm{Ie}\right]
```

```math
\mathrm{shape}\left(W_{\mathrm{up}}\right)=\left[H,\mathrm{Ie}\right]
```

```math
\mathrm{shape}\left(W_{\mathrm{down}}\right)=\left[\mathrm{Ie},H\right]
```

生产实现常把 `gate_proj` 和 `up_proj` 打包：

```math
\mathrm{gate\_up}=X_{e}\,\mathrm{W\_gate\_up}\qquad\mathrm{shape}:\left[r_{e},2\cdot \mathrm{Ie}\right]
```

```math
\text{gate, up}= \mathrm{split}\left(\mathrm{gate\_up}, 2\right) \left[r_{e},\mathrm{Ie}\right], \left[r_{e},\mathrm{Ie}\right]
```

因此一次 expert 的数据流是：

```text
[r_e,H] -> [r_e,2*Ie] -> [r_e,Ie] -> [r_e,H]
```

## 8. Combine

Expert 输出仍按 expert/route 顺序排列：

```math
\mathrm{shape}\left(\mathrm{expert\_output}\right)=\left[R,H\right]
```

Combine 使用 `route_token_id` 恢复 token，并使用 `route_weight` 加权累加：

```math
Y\left[t,:\right] = \sum_{r: \mathrm{token}\left(r\right)=t} \mathrm{route\_weight}\left[r\right] \cdot  \mathrm{expert\_output}\left[r,:\right]
```

```math
\mathrm{shape}\left(Y\right)=\left[T,H\right]
```

Combine 同时完成逆置换和 reduce。每个 token 恰有 `K` 条有效 route 时，每个输出行合并 `K` 个 expert 结果。

## 9. Expert Parallel

当 expert 参数不能放在单卡或需要提高吞吐时，把 experts 分布到 `Pep` 个 ranks。若均匀划分：

```math
E_{\mathrm{local}}=\frac{E}{\mathrm{Pep}}
```

Router 后，每个 rank 上的 token 可能选择远端 expert。因此数据流变成：

```text
local tokens [T_local,H]
  -> local router/top-k
  -> dispatch + all-to-all
  -> received expert rows [R_local,H]
  -> local experts
  -> combine + all-to-all
  -> local token outputs [T_local,H]
```

第一次 all-to-all 把 token rows 发送到拥有目标 expert 的 rank；第二次把 expert 输出送回拥有原 token 的 rank。

### 9.1 发送量由路由决定

rank `i` 到 rank `j` 的发送行数不是固定的：

```text
send_count[i,j] = number of routes on rank i targeting experts on rank j
```

某些 expert 被大量选中时，会出现负载倾斜：

- 热门 expert 的 GEMM 行数更大；
- 对应 rank 接收更多 token；
- all-to-all 的慢 rank 决定整体延迟；
- 小 batch/decode 时，每个 expert 的行数过少，GEMM 利用率下降。

## 10. TP 与 EP 的区别

| 并行方式 | 切分对象 | token 是否跨 rank 移动 | 主要通信 |
|---|---|---|---|
| Tensor Parallel | 同一权重矩阵的维度 | 通常不因路由改变 owner | all-reduce / reduce-scatter |
| Expert Parallel | 不同 experts | 是，按 Top-K 目标重分布 | all-to-all dispatch/combine |

MoE 还可以在 expert 内部使用 TP，此时每个 expert 的矩阵又被多 rank 切分。系统需要同时处理 route 级 all-to-all 和矩阵分片级 reduction。

## 11. 参数量与激活计算量

忽略 bias，单个 SwiGLU expert 参数量约为：

```math
\mathrm{params\_per\_expert}=H\cdot \mathrm{Ie}+H\cdot \mathrm{Ie}+\mathrm{Ie}\cdot H=3\cdot H\cdot \mathrm{Ie}
```

全部 experts：

```math
\mathrm{expert\_params}=E\cdot 3\cdot H\cdot \mathrm{Ie}
```

单 token 激活 `K` 个 experts，其 expert matmul 计算量近似与：

```text
active_compute_per_token proportional to K * 3*H*Ie
```

相关，而不是 $`E\cdot 3\cdot H\cdot \mathrm{Ie}`$。但模型权重显存仍与 `E` 相关。

## 12. Prefill 与 Decode 中的 MoE

### Prefill

`T` 较大，路由后每个 expert 往往能获得较多 rows：

```text
router GEMM: [T,H] @ [H,E]
expert GEMMs: [r_e,H] @ expert weights
```

较大的 `r_e` 有利于 GPU GEMM 利用率，但 all-to-all 数据量也更大。

### Decode

普通 decode 中 `T` 约等于活跃请求数。小 batch 时：

```math
R=T\cdot K
```

可能仍然很小，并分散到多个 experts。此时 router、dispatch、通信和 kernel launch 的固定开销占比明显上升。

## 13. MoE 的参考实现顺序

一个与框架无关的 MoE forward 可以表示为：

```math
\mathrm{router\_logits}=X\,W_{\mathrm{router}}\qquad\text{[T,E]}
```

```math
\left[\mathrm{topk\_scores},\mathrm{topk\_ids}\right]=\mathrm{topk}\left(\mathrm{router\_logits}\right)\qquad\text{[T,K], [T,K]}
```

```math
\mathrm{topk\_weights} = \mathrm{normalize}\left(\mathrm{topk\_scores}\right) \qquad\text{[T,K]}
```

```math
\text{permuted\_X, route\_meta}= \mathrm{dispatch}\left(X, \mathrm{topk\_ids}\right)
```

```math
\qquad\text{[R,H], R=T*K}
```

```math
\mathrm{permuted\_Y} = \mathrm{grouped\_expert\_swiglu}\left(\mathrm{permuted\_X}, \mathrm{route\_meta}\right) \qquad\text{[R,H]}
```

```math
Y = \mathrm{combine}\left(\mathrm{permuted\_Y}, \mathrm{route\_meta}, \mathrm{topk\_weights}\right) \qquad\text{[T,H]}
```

`dispatch` 必须保存逆置换信息，否则无法把 expert 顺序恢复为 token 顺序。`combine` 必须同时执行逆置换和按路由权重求和。

## 14. 可并行调度的阶段划分

分布式 MoE 可以拆成以下数据依赖阶段：

```text
Gate
  -> Select Experts
  -> Count Routes per Destination
  -> Dispatch All-to-All
  -> Grouped Expert GEMM
  -> Combine All-to-All
  -> Weighted Reduction
```

数据依赖：

| 阶段 | 消费 | 产生 |
|---|---|---|
| Gate | `X [T,H]` | `router_logits [T,E]` |
| Select Experts | logits | ids/weights $`\left[T,K\right]`$ |
| Count Routes | expert ids | 每个目标 rank/expert 的行数 |
| Dispatch | token rows + route metadata | expert 分组 rows $`\left[R_{\mathrm{local}},H\right]`$ |
| Expert GEMM | expert 分组 rows | expert outputs $`\left[R_{\mathrm{local}},H\right]`$ |
| Combine | expert outputs + inverse routes | token route outputs $`\left[R,H\right]`$ |
| Weighted Reduction | route outputs + weights | `Y [T,H]` |

当通信库支持异步操作时，route count、数据发送和本地 expert GEMM 可以流水化；数学结果仍由依赖顺序约束。

## 15. 不能混淆的三个数量

```text
E = 模型拥有的 routed experts 总数
K = 每个 token 激活的 expert 数量
R = 本轮 expert route 总数，通常为 T*K
```

`E` 决定总 expert 参数规模，`K` 决定单 token 激活计算，`R` 决定本轮 dispatch buffer 和 expert 输入总行数。
