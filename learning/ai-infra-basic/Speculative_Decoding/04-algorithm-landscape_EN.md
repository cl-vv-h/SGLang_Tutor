[中文](./04-algorithm-landscape.md) | [English](./04-algorithm-landscape_EN.md)

# Speculative Decoding Algorithm Landscape

## 1. Method Comparison

| Method | Draft Source | Draft Quality | Overhead | SGLang Support |
|---|---|---|---|---|
| **EAGLE** | Trained draft model | High | Medium (separate forward) | ✅ EAGLE v1/v2 |
| **MTP** | Extra heads on target | Medium-High | Low (shared backbone) | ✅ |
| **NGRAM** | Cache of previous outputs | Low-Medium | Very low (lookup only) | ✅ |
| **Medusa** | Multiple extra heads | Medium | Low | ✅ |
| **REST** | Retrieval from datastore | Varies | Low | Partial |

## 2. EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency)

- Trains a separate draft model that predicts the target model's hidden states
- Uses the predicted hidden states to generate draft tokens
- Higher quality drafts than simple n-gram or extra-head approaches
- SGLang supports multi-layer EAGLE (multiple draft model layers)

## 3. MTP (Multi-Token Prediction)

- Adds auxiliary prediction heads to the target model
- Each head predicts one additional future token
- No separate draft model — lower memory overhead
- Draft tokens come from the main forward pass directly

## 4. NGRAM Speculative Decoding

- Maintains a cache of recently generated token sequences
- Proposes draft tokens by matching the current context against the cache
- Pros: zero additional model cost
- Cons: lower acceptance rate, especially for creative/open-ended generation

## 5. Spec v1 vs v2 in SGLang

SGLang evolved its speculative decoding implementation:

**v1**: Basic draft-target architecture
- Separate draft worker process
- Draft → spec_info → verify cycle
- Limited to EAGLE and NGRAM

**v2**: Unified speculative framework
- Draft can be any model (not just EAGLE)
- Better overlap with continuous batching
- Improved KV Cache management for spec
- Support for grammar-constrained speculative decoding

## 6. Choosing a Method

| Scenario | Recommended Method |
|---|---|
| Maximum speed, can train draft model | EAGLE |
| No extra model memory budget | MTP / Medusa |
| Quick setup, no training | NGRAM |
| Structured output / grammar constraints | EAGLE v2 + grammar |
| Low latency requirement | Smaller K (fewer draft tokens) |
| High throughput requirement | Larger K with high-quality draft |
