# Speculative Decoding / Speculative Inference 教学入口

投机推理在大模型 serving 中通常指 speculative decoding 或 speculative sampling。核心思想是：用一个更便宜的 draft 路径先提出多个未来 token，再让 target model 用一次或少数几次验证 forward 检查这些 token，从而把“每生成一个 token 调一次大模型”改成“每次大模型验证尽量提交多个 token”。

本专题关注三个问题：

1. 投机解码为什么可能加速。
2. 严格投机采样为什么可以不改变目标模型分布。
3. 在线 serving 系统如何管理 draft token、target verify、KV Cache、scheduler、logprob、stop condition 和多种算法实现。

## 本专题文件

| 文件 | 内容 |
|---|---|
| [01-speculative-decoding-principles.md](./01-speculative-decoding-principles.md) | 投机解码的核心直觉、参与模块、张量形状、收益模型和适用边界 |
| [02-rejection-sampling-math.md](./02-rejection-sampling-math.md) | 严格 speculative sampling 的数学原理、拒绝采样证明、链式验证和 greedy 特例 |
| [03-serving-implementation-dataflow.md](./03-serving-implementation-dataflow.md) | 在线 serving 中 draft、verify、accept、KV commit、draft extend、scheduler overlap 的数据流 |
| [04-algorithm-landscape.md](./04-algorithm-landscape.md) | 小模型 draft、tree verification、Medusa、EAGLE、MTP、NGRAM、REST、LayerSkip、Lookahead 等算法谱系 |

## 推荐阅读顺序

1. 先读 [01-speculative-decoding-principles.md](./01-speculative-decoding-principles.md)，建立“draft 提案、target 验证、一次提交多个 token”的整体模型。
2. 再读 [02-rejection-sampling-math.md](./02-rejection-sampling-math.md)，理解为什么严格投机采样可以保持 target distribution。
3. 接着读 [03-serving-implementation-dataflow.md](./03-serving-implementation-dataflow.md)，把数学上的 token 接受过程落到 KV Cache、ForwardBatch、scheduler 和输出流。
4. 最后读 [04-algorithm-landscape.md](./04-algorithm-landscape.md)，对比各种 draft 来源和 verify 几何结构，知道不同算法在什么场景下更合适。

## 统一符号

| 符号 | 含义 |
|---|---|
| `B` | 当前 decode batch 中的请求数 |
| `K` | 每轮最多 draft 的 token 数，线性链场景也称 draft length |
| `S` | 当前请求已有上下文长度 |
| `V` | 词表大小 |
| `p_i` | target model 在第 `i` 个候选位置给出的目标分布 |
| `q_i` | draft path 在第 `i` 个候选位置给出的提案分布 |
| `y_i` | draft 提出的第 `i` 个候选 token |
| `A` | 一轮中被接受的 draft token 数，不含 correction/bonus token |
| `C_T` | 一次普通 target decode step 的成本 |
| `C_D` | draft path 生成一个 token 的成本 |
| `C_V(K)` | target 一次验证 `K` 个候选 token 的成本 |

## 核心直觉

普通 decode 每轮只提交一个 token：

```text
target forward -> sample token_1
target forward -> sample token_2
target forward -> sample token_3
target forward -> sample token_4
```

投机解码把未来几步先猜出来：

```text
draft forward -> propose token_1, token_2, token_3, token_4
target verify -> check these positions in one target pass
accept prefix -> commit accepted tokens, then append correction/bonus token
```

如果平均每轮可以提交 `E[A]+1` 个 token，而 target verify 的成本没有线性增长到 `K` 倍，就能减少大模型调度轮数，降低 inter-token latency 并提高吞吐。

## 和 SGLang 代码阅读的连接点

SGLang 的 speculative decoding 不是单一函数，而是一条跨 scheduler、worker、attention backend、sampling kernel、KV memory pool 的执行链。阅读源码时可以重点看这些概念：

| 概念 | 典型代码位置 |
|---|---|
| 算法枚举与统一接口 | `python/sglang/srt/speculative/spec_info.py` 中的 `SpeculativeAlgorithm`、`SpecInput` |
| 接受/拒绝采样 | `python/sglang/srt/speculative/reject_sampling.py` |
| EAGLE 族 worker | `python/sglang/srt/speculative/eagle_worker_v2.py` |
| Standalone 小模型 draft | `python/sglang/srt/speculative/standalone_worker_v2.py` |
| NGRAM draft | `python/sglang/srt/speculative/ngram_worker.py`、`ngram_info.py` |
| DFlash / DSpark | `python/sglang/srt/speculative/dflash_worker_v2.py`、`dspark_components/` |
| Scheduler 状态衔接 | `python/sglang/srt/managers/scheduler.py`、`schedule_batch.py` |

阅读时要始终区分三类状态：

1. 正式 target 状态：已经提交给用户输出的 token，以及与之对齐的 target KV Cache。
2. 临时候选状态：draft token、draft tree、verify mask、临时 KV slot、target verify logits。
3. 下一轮 draft 状态：draft KV、EAGLE hidden state、NGRAM trie、adaptive step tier、future map 等。

## 阅读目标

读完本专题后，应能够回答：

1. 投机解码为什么可以减少 target decode step，但不一定降低单轮计算量。
2. `min(1, p(y)/q(y))` 的接受概率为什么能保持 target distribution。
3. target verify 为什么像一个短 prefill，而不是普通单 token decode。
4. 接受多个 token 后，KV Cache、请求长度、stop condition、logprob 如何更新。
5. EAGLE、Medusa、MTP、NGRAM、REST、LayerSkip 等方法到底是在改变 draft 来源，还是在改变 verify 结构。
