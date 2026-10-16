# Parallel Strategy 教学入口

**简体中文** | [English](../../../en/ai-infra-basic/Parallel_Strategy/README.md)

本课程按“**切什么 → 为什么能这样切 → 数据怎样流动 → 源码怎样实现**”组织。先区分请求、权重、激活、query 和历史 KV，再讨论组合；不能把所有 degree 直接相乘，也不能把 SP、PCP、DCP 当成同一个功能。

## 课程结构

| 章节 | 核心问题 | 图与例子 |
|---|---|---|
| [00：策略地图与版本边界](./00-parallelism-map-and-version-boundaries.md) | 全部策略处于服务流程哪里？本地与上游接口有什么不同？ | 策略表、执行阶段图 |
| [基础讲义：DP / TP / PP / EP](./tutorial.md) | 请求、矩阵、层和专家怎样切？ | Column/Row TP、流水线、token dispatch |
| [01：DP Attention](./01-dp-attention-principles.md) | Attention 请求子组如何与全局 FFN/MoE 配合？ | partial/replica、gather/scatter、padding/IDLE |
| [02：SP 与 CP 的区别](./02-sp-versus-context-parallelism.md) | 为什么逐行 Norm 可以分，而 attention 需要历史？ | 四 token 手算、TP 完整链路、RS → Norm → AG、CP 数值对照 |
| [03：PCP](./03-prefill-context-parallelism.md) | 怎样分长 prompt 的 query 工作并保留因果关系？ | prefix 与 query 编号手算、逐行 softmax、输出恢复、zigzag/interleave |
| [04：DCP](./04-decode-context-parallelism.md) | 怎样分历史 KV，又精确重构 softmax？ | 两种推导、LSE 数值例子、虚拟/物理 slot |
| [05：DWDP](./05-dwdp-weight-parallelism.md) | 何时让权重来找 token？ | EP 对比、字节交点、VMM 与双缓冲时序 |
| [06：组合拓扑与 PD](./06-parallel-topology-and-pd-composition.md) | 一个 rank 有几种身份？P/D 不同并行度如何交接？ | 八卡网格、DCP 子组、KV head 重映射 |
| [07：Encoder DP 与弹性 EP](./07-encoder-dp-and-elastic-ep.md) | 模块并行、专家副本和动态扩容怎样区分？ | 编码工作分配、logical/physical expert、状态图 |
| [08：Diffusion 扩展](./08-diffusion-parallelism.md) | CFG、Ulysses、Ring、KV-gather 怎样配合？ | 分支图、shape 变换、环形数据流 |

每个新增专题均有中英文对应文件、数学推导、数据流图、源码入口和练习。基础讲义保留原有详细示例；其中旧 CP 部分已明确标注历史语境，SP/PCP/DCP 的定义与具体实现以 02–04 为准。

## 推荐路线

1. 初学者：00 → 基础 DP/TP/PP/EP → 01，先能解释“哪个量是副本，哪个量是 partial”。
2. 长上下文：02 → 03 → 04，比较逐行激活、prefill query、decode 历史 KV。
3. MoE 与组合部署：05 → 06 → 07，再接 [Communication](../Communication/README.md) 和 [KV Transfer](../KV_Transfer/README.md)。
4. 图像/视频生成：08，注意它属于另一套 runtime，不能直接复用 SRT 启动参数。

## 可运行的数学实验

下面四份只用 Python 标准库，在仓库根目录运行：

```bash
python3 learning/zh/ai-infra-basic/Parallel_Strategy/sp_step_by_step_demo.py
python3 learning/zh/ai-infra-basic/Parallel_Strategy/pcp_query_rows_demo.py
python3 learning/zh/ai-infra-basic/Parallel_Strategy/parallelism_math_demo.py
python3 learning/zh/ai-infra-basic/Parallel_Strategy/dp_attention_layout_demo.py
```

[sp_step_by_step_demo.py](./sp_step_by_step_demo.py) 配合 02，逐步打印 partial、完整行、Norm、矩阵分片与 residual，并用完整 FFN 对照普通 TP 和 TP+SP，最后演示 CP 为什么需要远端 KV。

[pcp_query_rows_demo.py](./pcp_query_rows_demo.py) 配合 03 第二节，用两个 prefix token 和三个新 token 展示可见历史、权重计算、query 分配与输出恢复，并复现遗漏 KV 和使用本地行号的错误。

[parallelism_math_demo.py](./parallelism_math_demo.py) 对照未分片结果，验证 SP、ragged PCP、稳定 DCP 合并（含空片和大 score）、EP/DWDP、rank 拓扑、PD head 重排、Encoder DP 和 Ulysses/CFG。它不模拟真实网络延迟，也不证明目标设备/backend 支持相同组合。

[dp_attention_layout_demo.py](./dp_attention_layout_demo.py) 专门解释 SUM_LEN/MAX_LEN gather、replica 去重与 reduce-scatter 返回。

| 保留的 PyTorch demo | 阅读时关注 |
|---|---|
| [dp_inference_demo.py](./dp_inference_demo.py) | 独立请求与完整模型副本 |
| [tp_inference_demo.py](./tp_inference_demo.py) | Column/Row 权重与 partial 求和 |
| [pp_inference_demo.py](./pp_inference_demo.py) | stage 间传激活与自回归依赖 |
| [sp_inference_demo.py](./sp_inference_demo.py) | 历史文件名；内容是 CP 分布式 softmax，不是 LayerNorm SP |
| [ep_moe_demo.py](./ep_moe_demo.py) | token dispatch、expert 执行与 combine |

部分旧 demo 需要 PyTorch、多进程或多设备，先读其运行说明。

## 源码阅读与版本

本地源码和固定上游的差别统一见 [00](./00-parallelism-map-and-version-boundaries.md)。每章把上游实现链接到 commit，而不是会移动的 main；不存在的本地功能不伪造入口。

读完 01，接 [DP Attention 源码走读](../../sglang-source-reading/05-layer-communication/03-dp-attention-code-walkthrough.md)；组合 rank 与 Ascend 路径接 [TP/DP/EP 源码篇](../../sglang-source-reading/05-layer-communication/02-tp-dp-ep-sharding-and-ascend-npu.md)。通信原语和 Gloo/NCCL/HCCL、DeepEP/ZBAL、Mooncake 的分工见 [通信课程](../Communication/README.md)。
