[中文](./README.md) | [English](./README_EN.md)

# Speculative Decoding

This topic explains how to accelerate LLM decoding by using a fast "draft" model to propose multiple tokens, which are then verified by the target model in a single forward pass.

## Core Principles

Instead of the standard autoregressive decode that produces one token per forward pass, speculative decoding:

1. **Draft**: A small/fast draft model generates K candidate tokens
2. **Verify**: The target model processes all K tokens in one forward pass
3. **Accept/Reject**: Rejection sampling determines how many tokens to accept
4. **Repeat**: Continue from the last accepted position

## Files

| File | Content |
|---|---|
| [01-speculative-decoding-principles.md](./01-speculative-decoding-principles.md) | Draft model, target verification, KV Cache management |
| [02-rejection-sampling-math.md](./02-rejection-sampling-math.md) | Mathematical foundations of rejection sampling for spec decode |
| [03-serving-implementation-dataflow.md](./03-serving-implementation-dataflow.md) | SGLang's `spec_info`, draft worker, and target verification dataflow |
| [04-algorithm-landscape.md](./04-algorithm-landscape.md) | EAGLE, MTP, NGRAM, Medusa, REST, and other speculative methods |

## Key SGLang Integration Points

- `Scheduler` manages draft worker and target worker
- `spec_info` carries draft tokens and verification metadata between forward passes
- KV Cache commits are deferred until verification completes
- `BatchResultProcessor` handles accept/reject logic and token output
