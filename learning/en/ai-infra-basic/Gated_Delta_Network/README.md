# Gated Delta Network

[简体中文](../../../zh/ai-infra-basic/Gated_Delta_Network/README.md) | **English**

In this topic, **GDN** means **Gated Delta Network / Gated DeltaNet**, the linear-attention-style layer used by hybrid models such as Qwen3-Next. It is not Generalized Divisive Normalization from image compression. The core idea is to replace a KV Cache that grows with context length by a fixed-size recurrent state: every token reads and writes the state through `q/k/v`, `g` controls forgetting, and `beta` controls how strongly new information is written.

## Reading Order

| Order | File | Key Question |
|---|---|---|
| 1 | [01-gdn-math-and-state.md](./01-gdn-math-and-state.md) | GDN equations, variable meanings, and why the state update is called a delta rule |
| 2 | [02-gdn-layer-dataflow-and-shapes.md](./02-gdn-layer-dataflow-and-shapes.md) | End-to-end shapes from hidden states to `q/k/v/z/a/b`, convolution, gating, state update, and output |
| 3 | [03-gdn-training-and-serving.md](./03-gdn-training-and-serving.md) | Trainable parameters, training flow, prefill/decode/target-verify execution, and serving state management |

## One Overview Diagram

![GDN layer dataflow](./assets/gdn-layer-dataflow.svg)

## Key Takeaways

| Question | Answer |
|---|---|
| Is GDN attention? | It is a linear-attention / state-space-style alternative to attention. It represents history through a recurrent state. |
| Is the GDN state a parameter? | No. `S_t` or `ssm_states` is per-request runtime cache, similar to Mamba state, and is not updated by the optimizer. |
| What is trainable? | Input projections, short convolution, `A_log`, `dt_bias`, output-gated normalization, output projection, and the model's other MLP/MoE/embedding weights. |
| Are `a` and `b` parameters? | No. They are token-level activations produced by linear projections from hidden states; the projection weights that produce them are parameters. |
| Are `g` and `beta` parameters? | No. `g = -exp(A_log) * softplus(a + dt_bias)` and `beta = sigmoid(b)`. They are control signals computed from parameters and activations. |
| Why is GDN useful for long context? | Per-layer per-request state is roughly `num_value_heads * value_dim * key_dim`, and does not grow linearly with historical token count. |
| How does KDA differ? | GDN applies one retention scalar per token/head; Kimi Delta Attention applies a vector over key channels so state columns can decay independently. |

After these three chapters, continue with [Kimi Delta Attention](../Model_Architecture/09-kimi-delta-attention.md) to see how fine-grained gating changes the recurrence, kernels, and serving-state contract.

## Relationship to SGLang Source Reading

| Source Location | Role |
|---|---|
| `python/sglang/srt/models/qwen3_next.py` | `Qwen3GatedDeltaNet` defines projections, convolution, `A_log/dt_bias`, norm, and output projection |
| `python/sglang/srt/layers/radix_linear_attention.py` | Unified linear attention layer entry point |
| `python/sglang/srt/layers/attention/linear/gdn_backend.py` | GDN backend; separates prefill, decode, and target verify |
| `python/sglang/srt/layers/attention/fla/fused_gdn_gating.py` | Fused kernel that computes `g` and `beta` |
| `python/sglang/srt/layers/attention/fla/fused_recurrent.py` | Recurrent decode update and packed decode fast path |
| `python/sglang/srt/layers/attention/fla/chunk_fwd.py` | Parallel chunk gated delta rule core for prefill |
| `python/sglang/srt/configs/qwen3_next.py` | GDN head counts, head dimensions, convolution kernel size, and hybrid-layer configuration |

## References

- [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464)
- [Qwen3-Next model card](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)
- [Qwen blog: Qwen3-Next](https://qwenlm.github.io/blog/qwen3-next/)
- [Qwen blog: Qwen3-Next, NVIDIA Blackwell, and FlashQLA](https://qwenlm.github.io/blog/qwen3-next-flashqla/)
- [Flash Linear Attention project](https://github.com/fla-org/flash-linear-attention)
- [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)
