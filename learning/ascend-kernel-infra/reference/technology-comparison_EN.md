[中文](./technology-comparison.md) | [English](./technology-comparison_EN.md)

# Technology Comparison: When to Use What

## Decision Tree

```text
Need to implement an NPU operator?
├─ Standard op exists? → Use torch.ops.npu / ACLNN
├─ Simple fused kernel (element-wise, reduction)?
│   → Triton-Ascend
│   Pros: Fast dev, readable code, autotune
│   Cons: Less control over memory/pipeline
├─ Complex kernel with explicit control?
│   → Ascend C
│   Pros: Fine-grained control, peak performance
│   Cons: More boilerplate, longer dev time
└─ Performance-critical, large problem?
    → Ascend C with double-buffer pipeline
    Pros: Best throughput, full hardware utilization
    Cons: Most complex
```

## Comparison Matrix

| Criterion | torch.ops.npu | Triton-Ascend | Ascend C |
|---|---|---|---|
| Dev speed | Fast (no code) | Fast (Python) | Slow (C++) |
| Performance | Good (ACLNN impl) | Good-Very Good | Excellent |
| Customizability | None | High | Full |
| Portability | Ascend only | Cross-platform (Triton) | Ascend only |
| Debugging | Hard (closed source) | Medium (IR inspection) | Hard (low-level) |
| Memory control | None | Limited | Full (UB/GM explicit) |
| Pipeline control | None | Implicit | Explicit (TQue) |

## Recommendation by Scenario

| Scenario | Recommended |
|---|---|
| Prototype / research | Triton-Ascend |
| Production, moderate perf needs | Triton-Ascend |
| Production, peak perf required | Ascend C |
| Standard ops, no customization | torch.ops.npu / ACLNN |
| SGLang-specific fused op | sgl-kernel-npu (Triton or Ascend C) |
