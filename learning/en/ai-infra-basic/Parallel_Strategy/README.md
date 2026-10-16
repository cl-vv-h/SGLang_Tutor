# Parallel Strategy

[简体中文](../../../zh/ai-infra-basic/Parallel_Strategy/README.md) | **English**

This course follows **what is split → why the mathematics permits it → how data moves → how source code implements it**. Separate requests, weights, activations, queries, and historical KV before composing strategies. Do not multiply every degree or treat SP, PCP, and DCP as one feature.

## Course structure

| Chapter | Main question | Diagrams and examples |
|---|---|---|
| [00: map and version boundaries](./00-parallelism-map-and-version-boundaries.md) | Where does each strategy act? How do local and upstream APIs differ? | Strategy table and execution-stage map |
| [Foundations: DP / TP / PP / EP](./tutorial.md) | How are requests, matrices, layers, and experts split? | Column/row TP, pipelines, token dispatch |
| [01: DP Attention](./01-dp-attention-principles.md) | How do attention request subgroups cooperate with global FFN/MoE? | Partial/replica, gather/scatter, padding/IDLE |
| [02: SP versus CP](./02-sp-versus-context-parallelism.md) | Why can Norm split by rows while attention needs history? | Four-token arithmetic, full TP flow, RS → Norm → AG, CP numerical comparison |
| [03: PCP](./03-prefill-context-parallelism.md) | How does query work split while preserving causality? | Prefix/query arithmetic, row-wise softmax, zigzag/interleave execution and source walkthrough |
| [04: DCP](./04-decode-context-parallelism.md) | How does sharded KV reconstruct exact softmax? | Two derivations, LSE example, virtual/physical slots |
| [05: DWDP](./05-dwdp-weight-parallelism.md) | When should weights move to tokens? | EP comparison, traffic crossover, VMM and double buffering |
| [06: topology and PD](./06-parallel-topology-and-pd-composition.md) | How many identities does a rank have? How do different P/D degrees hand off KV? | Eight-rank grid, DCP subgroups, KV-head remapping |
| [07: encoder DP and elastic EP](./07-encoder-dp-and-elastic-ep.md) | How do module parallelism, expert copies, and scaling differ? | Encoder assignment, logical/physical experts, state diagram |
| [08: diffusion extension](./08-diffusion-parallelism.md) | How do CFG, Ulysses, Ring, and KV-gather interact? | Branch graph, shape transformations, ring data flow |

Every new topic has corresponding English/Chinese files, mathematical derivations, diagrams, source entries, and exercises. Foundations retain existing examples; legacy CP terminology is labeled explicitly. Chapters 02–04 define the distinct SP/PCP/DCP mechanisms and implementations.

## Reading paths

1. Beginners: 00 → foundational DP/TP/PP/EP → 01. First explain which value is replicated and which is partial.
2. Long contexts: 02 → 03 → 04. Compare row activations, prefill queries, and decode historical KV.
3. MoE and deployment composition: 05 → 06 → 07, followed by [Communication](../Communication/README.md) and [KV Transfer](../KV_Transfer/README.md).
4. Image/video generation: 08. Its separate runtime does not share SRT launch settings directly.

## Runnable math experiments

These four use only Python's standard library. Run from the repository root:

```bash
python3 learning/en/ai-infra-basic/Parallel_Strategy/sp_step_by_step_demo.py
python3 learning/en/ai-infra-basic/Parallel_Strategy/pcp_query_rows_demo.py
python3 learning/en/ai-infra-basic/Parallel_Strategy/parallelism_math_demo.py
python3 learning/en/ai-infra-basic/Parallel_Strategy/dp_attention_layout_demo.py
```

[sp_step_by_step_demo.py](./sp_step_by_step_demo.py) accompanies chapter 02, printing partials, complete rows, Norm, weight shards, and residuals. It compares full FFN, ordinary TP, and TP+SP, then demonstrates why CP needs remote KV.

[pcp_query_rows_demo.py](./pcp_query_rows_demo.py) accompanies chapter 03 section 2. Two prefix tokens and three new tokens demonstrate history visibility, weights, query assignment, and output restoration, including missing-KV and local-index mistakes.

[parallelism_math_demo.py](./parallelism_math_demo.py) compares decompositions against unsharded results: SP, ragged PCP, stable DCP merging (empty shards and large scores), EP/DWDP, rank topology, PD head remapping, encoder DP, and Ulysses/CFG. It neither simulates network latency nor proves device/backend support for the same combinations.

[dp_attention_layout_demo.py](./dp_attention_layout_demo.py) focuses on SUM_LEN/MAX_LEN gathering, replica deduplication, and reduce-scatter returns.

| Retained PyTorch demo | What to follow |
|---|---|
| [dp_inference_demo.py](./dp_inference_demo.py) | Independent requests and complete-model copies |
| [tp_inference_demo.py](./tp_inference_demo.py) | Column/row weights and partial sums |
| [pp_inference_demo.py](./pp_inference_demo.py) | Activation handoff and autoregressive dependencies |
| [sp_inference_demo.py](./sp_inference_demo.py) | Legacy filename: CP distributed softmax, not LayerNorm SP |
| [ep_moe_demo.py](./ep_moe_demo.py) | Token dispatch, expert execution, and combine |

Some old demos require PyTorch, multiple processes, or multiple devices; read their launch instructions first.

## Source reading and versions

[00](./00-parallelism-map-and-version-boundaries.md) centralizes local versus pinned-upstream differences. Chapters link upstream implementations to a commit rather than moving main, and do not invent local entries for absent features.

After 01, follow the [DP Attention walkthrough](../../sglang-source-reading/05-layer-communication/03-dp-attention-code-walkthrough.md); for rank composition and Ascend, read [TP/DP/EP source analysis](../../sglang-source-reading/05-layer-communication/02-tp-dp-ep-sharding-and-ascend-npu.md). The [communication course](../Communication/README.md) connects collectives with Gloo/NCCL/HCCL, DeepEP/ZBAL, and Mooncake.

For the function-by-function implementation, read the [PCP source walkthrough](../../sglang-source-reading/05-layer-communication/04-prefill-context-parallel-code-walkthrough.md): runner, KV reordering, kernels, padding, DSA, and older NPU paths. Its [CPU layout trace](../../sglang-source-reading/05-layer-communication/pcp_layout_trace_demo.py) prints concrete metadata and permutations without device communication.
