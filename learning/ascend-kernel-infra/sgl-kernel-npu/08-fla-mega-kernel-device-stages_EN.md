[中文](./08-fla-mega-kernel-device-stages.md) | [English](./08-fla-mega-kernel-device-stages_EN.md)

# sgl-kernel-npu 08: FLA Mega Kernel & Device Stages

## Mega Kernel Concept

Instead of launching multiple small kernels, fuse FLA operations into one "mega" kernel with internal stages.

## 7 Device Stages

```text
Stage 1: Load Q, K, V tiles from GM → UB
Stage 2: Compute chunked attention scores
Stage 3: Apply gated delta rule
Stage 4: Recurrent state update
Stage 5: Output computation
Stage 6: State write-back to GM
Stage 7: CopyOut final output to GM
```

## AIV/AIC Collaboration

```text
AIC (AI Core): Matrix multiply, Cube Unit ops
  → Handles chunked attention scores, state updates

AIV (AI Vector Core): Element-wise ops, reductions
  → Handles gating, delta rule, activation functions

Mega kernel orchestrates AIC ↔ AIV handoff within a single launch
```

## GM Workspace & Sync Cost

- **GM Workspace**: Intermediate state buffers between stages
- **Sync cost**: Inter-stage synchronization via TQue
- **Trade-off**: Mega kernel reduces launch overhead but requires more workspace
