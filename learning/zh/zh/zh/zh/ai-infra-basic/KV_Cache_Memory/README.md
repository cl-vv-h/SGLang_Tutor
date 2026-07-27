# KV Cache Memory

这一章关注 LLM serving 中最珍贵的资源之一：KV Cache 显存。只要理解 KV Cache 的形状、分配、复用和回收，就能理解很多调度、attention backend、prefix cache 和 PD 分离设计。

## KV Cache 保存了什么

Transformer 每层 attention 都会产生 Key 和 Value。decode 第 `t` 步不需要重新计算前 `t-1` 个 token 的 K/V，而是从 KV Cache 中读取历史 K/V，再和当前 token 的 Query 做 attention。

```text
每一层:
    K cache: [历史 token, kv_heads, head_dim]
    V cache: [历史 token, kv_heads, head_dim]

每个请求:
    prompt token 的 K/V
    已生成 token 的 K/V
```

## 显存估算公式

粗略估算：

```text
KV bytes =
    num_layers
  * num_tokens
  * kv_heads
  * head_dim
  * 2           # K 和 V
  * dtype_bytes
```

如果是 GQA/MQA，`kv_heads` 会小于 `num_attention_heads`，KV Cache 压力会明显降低。

例子：

```text
layers = 32
tokens = 4096
kv_heads = 32
head_dim = 128
dtype = fp16 = 2 bytes

KV = 32 * 4096 * 32 * 128 * 2 * 2
   = 2 GB 左右
```

这只是单请求。如果 batch 中有很多长上下文请求，KV Cache 会迅速成为主瓶颈。

## 连续内存和分页内存

最直观的方式是给每个请求分配一段连续 KV Cache。但线上请求长度变化很大，连续分配容易造成碎片，也难以支持动态增长。

Paged KV Cache 把 token 存到固定大小的 block/page 中：

```text
request A: block 1 -> block 7 -> block 9
request B: block 2 -> block 3
request C: block 4 -> block 8 -> block 10 -> block 11
```

这样请求结束后可以回收 block，新请求再复用。调度器只需要维护 request 到 block table 的映射。

## Prefix Cache 和 Radix Tree

普通 KV Cache 是请求私有的。Prefix Cache 关注的是多个请求共享相同 prompt 前缀时，能否复用已经算过的 prefix K/V。

典型场景：

```text
系统 prompt: 你是一个代码助手...
用户 A prompt: 系统 prompt + 问题 A
用户 B prompt: 系统 prompt + 问题 B
```

如果系统 prompt 很长，重复 prefill 会浪费大量计算。Radix tree 适合表示共享 token 前缀，因此可以把 prefix 对应的 KV block 放进可复用缓存。

## Eviction 为什么困难

KV Cache eviction 不是简单的 LRU。它至少要考虑：

1. block 是否被正在运行的请求引用。
2. prefix cache block 是否被多个请求共享。
3. 释放某个 prefix 是否会影响子节点。
4. 长 prompt 的重算代价通常比短 prompt 更高。
5. 显存不足时，调度器是否应该暂停 admission，而不是强行驱逐。

## 和 SGLang 的连接点

- KV cache manager 负责分配、记录和释放 token slot 或 block。
- RadixAttention 利用 radix tree 管理 prefix cache。
- Scheduler 在组 batch 前需要判断 KV Cache 是否足够。
- Attention backend 需要通过 block table 或 cache location 找到 K/V。
- HiCache、PD disaggregation、KV transfer 都建立在 KV Cache 可以被管理和移动的前提上。

## 阅读任务

1. 解释“KV Cache 是 memory-bound”的含义。
2. 比较连续 KV Cache 和 paged KV Cache 的优缺点。
3. 说明 prefix cache 和普通 KV Cache 的区别。
4. 设计一个简单 eviction 策略，并说明它会在哪些场景下表现不好。
