# Sparse MoE：Router、Top-K、Expert 与数据重排

## 1. MoE 替换了什么

Dense Transformer 的 FFN 对每个 token 使用同一组 MLP 权重：

```text
Y = FFN(X), X,Y: [T,H]
```

Sparse Mixture-of-Experts 准备 `E` 组 expert MLP，但每个 token 只选择 `K` 个：

```text
Y_t = sum_(e in TopK(t)) p_(t,e) * Expert_e(X_t)
```

当 `K << E` 时，模型可以拥有较大的总参数量，而单 token 只执行少数 experts。

![Sparse MoE 路由数据流](./assets/moe-routing-flow.svg)

## 2. MoE 主路径形状

```text
input X: [T,H]
router logits: [T,E]
topk ids: [T,K]
topk weights: [T,K]
dispatched rows: [R,H], R = T*K
expert outputs: [R,H]
combined output Y: [T,H]
```

主路径输入输出相同，但内部把一个 token 复制成 `K` 条 route。这里的复制是逻辑路由；高性能实现可能通过索引、排序和 fused kernel 避免普通框架中的显式复制。

## 3. Router Linear

Router 是从 hidden space 到 expert space 的线性层：

```text
Wr: [H,E]
G = X @ Wr
G: [T,E]
```

`G[t,e]` 表示 token `t` 分配给 expert `e` 的原始分数。Qwen3-MoE 的 SGLang 实现使用：

```text
self.gate = ReplicatedLinear(H, E, bias=False)
router_logits, _ = self.gate(hidden_states)
```

Router 参数相对于所有 expert 参数通常较小，复制在 ranks 上可以避免先为路由分数做切分通信。

## 4. Top-K 选择与归一化

对每个 token 独立选择分数最大的 `K` 个 expert：

```text
topk_ids[t,:] = indices_of_top_k(G[t,:])
```

选中分数转换为权重。常见形式：

```text
p[t,e] = softmax(G[t,:])[e]
```

或只在选中的 K 项上归一化：

```text
w[t,j] = exp(g[t,j]) / sum_(u in TopK(t)) exp(g[t,u])
```

结果：

```text
topk_ids: [T,K], integer
topk_weights: [T,K], floating point
```

SGLang 的 `TopK` 根据模型配置和 backend 生成统一的 `topk_output`，其中至少包含 expert ids 和权重。

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

```text
route count R = T*K
```

典型 metadata：

| 变量 | 形状 | 含义 |
|---|---:|---|
| `route_token_id` | `[R]` | 每条 route 来自哪个 token |
| `route_expert_id` | `[R]` | 每条 route 去哪个 expert |
| `route_weight` | `[R]` | combine 时使用的权重 |
| `permuted_input` | `[R,H]` | 按 expert 分组的 token rows |
| `expert_offsets` | `[E+1]` | 每个 expert 在 route buffer 中的区间 |

对 expert `e`：

```text
r_e = expert_offsets[e+1] - expert_offsets[e]
X_e: [r_e,H]
sum_e r_e = R
```

## 6. 一个具体路由例子

设 `T=4, E=4, K=2`：

| token | Top-2 expert ids | weights |
|---|---|---|
| `x0` | `[1,3]` | `[0.7,0.3]` |
| `x1` | `[0,1]` | `[0.6,0.4]` |
| `x2` | `[3,1]` | `[0.8,0.2]` |
| `x3` | `[0,2]` | `[0.55,0.45]` |

`R=T*K=8`。按 expert 排序后：

```text
expert 0 <- x1, x3       r_0=2
expert 1 <- x0, x1, x2   r_1=3
expert 2 <- x3           r_2=1
expert 3 <- x0, x2       r_3=2
```

expert 完成计算后，combine 恢复 token 顺序：

```text
y0 = 0.7*E1(x0) + 0.3*E3(x0)
y1 = 0.6*E0(x1) + 0.4*E1(x1)
y2 = 0.8*E3(x2) + 0.2*E1(x2)
y3 = 0.55*E0(x3) + 0.45*E2(x3)
```

最终 `Y=[y0,y1,y2,y3]`，形状恢复为 `[4,H]`。

## 7. 单个 Expert 的 SwiGLU

每个 Qwen3-MoE expert 是门控 MLP。对 `X_e [r_e,H]`：

```text
gate = X_e @ W_gate      [r_e,Ie]
up   = X_e @ W_up        [r_e,Ie]
mid  = SiLU(gate) * up   [r_e,Ie]
out  = mid @ W_down      [r_e,H]
```

权重形状：

```text
W_gate: [H,Ie]
W_up:   [H,Ie]
W_down: [Ie,H]
```

生产实现常把 `gate_proj` 和 `up_proj` 打包：

```text
gate_up = X_e @ W_gate_up          [r_e,2*Ie]
gate, up = split(gate_up, 2)       [r_e,Ie], [r_e,Ie]
```

因此一次 expert 的数据流是：

```text
[r_e,H] -> [r_e,2*Ie] -> [r_e,Ie] -> [r_e,H]
```

## 8. Combine

Expert 输出仍按 expert/route 顺序排列：

```text
expert_output: [R,H]
```

Combine 使用 `route_token_id` 恢复 token，并使用 `route_weight` 加权累加：

```text
Y[t,:] = sum_(r: token(r)=t) route_weight[r] * expert_output[r,:]
Y: [T,H]
```

Combine 同时完成逆置换和 reduce。每个 token 恰有 `K` 条有效 route 时，每个输出行合并 `K` 个 expert 结果。

## 9. Expert Parallel

当 expert 参数不能放在单卡或需要提高吞吐时，把 experts 分布到 `Pep` 个 ranks。若均匀划分：

```text
E_local = E / Pep
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

```text
params_per_expert = H*Ie + H*Ie + Ie*H = 3*H*Ie
```

全部 experts：

```text
expert_params = E * 3*H*Ie
```

单 token 激活 `K` 个 experts，其 expert matmul 计算量近似与：

```text
active_compute_per_token proportional to K * 3*H*Ie
```

相关，而不是 `E * 3*H*Ie`。但模型权重显存仍与 `E` 相关。

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

```text
R = T*K
```

可能仍然很小，并分散到多个 experts。此时 router、dispatch、通信和 kernel launch 的固定开销占比明显上升。

## 13. SGLang 普通执行路径

`Qwen3MoeSparseMoeBlock.forward_normal` 的核心调用链：

```text
hidden_states [T,H]
  -> gate(hidden_states)
     router_logits [T,E]
  -> topk(hidden_states, router_logits)
     topk_output: ids [T,K], weights [T,K]
  -> experts(hidden_states, topk_output)
     final_hidden_states [T,H]
  -> optional EP/TP reduction
```

`get_moe_impl_class(quant_config)` 根据量化和硬件后端选择 expert 实现。模型层依赖统一接口，不直接绑定某一种 Triton、CUDA、NPU 或量化 kernel。

## 14. DeepEP 分解执行路径

在 DeepEP 路径中，MoE 被拆成显式阶段：

```text
op_gate
  -> op_select_experts
  -> op_dispatch_a
  -> op_dispatch_b
  -> op_experts
  -> op_combine_a
  -> op_combine_b
  -> op_output
```

数据依赖：

| 阶段 | 消费 | 产生 |
|---|---|---|
| `op_gate` | `hidden_states_mlp_input` | `router_logits` |
| `op_select_experts` | hidden + logits | `topk_output` |
| `op_dispatch_a/b` | hidden + top-k | `dispatch_output` |
| `op_experts` | dispatched rows | `combine_input` |
| `op_combine_a/b` | expert output + route metadata | `hidden_states_after_combine` |
| `op_output` | combined rows | `hidden_states_mlp_output [T,H]` |

`dispatch_a/b` 和 `combine_a/b` 的拆分允许通信准备、数据传输和计算以 backend 支持的方式重叠。

## 15. 不能混淆的三个数量

```text
E = 模型拥有的 routed experts 总数
K = 每个 token 激活的 expert 数量
R = 本轮 expert route 总数，通常为 T*K
```

`E` 决定总 expert 参数规模，`K` 决定单 token 激活计算，`R` 决定本轮 dispatch buffer 和 expert 输入总行数。
