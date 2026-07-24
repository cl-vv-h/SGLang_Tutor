[中文](./README.md) | [English](./README_EN.md)

# Source Code Walkthrough: SGLang NPU Components

This topic provides a component-by-component walkthrough of how SGLang integrates with Ascend NPU at the source code level.

## Files

### Foundation (Read First)

| File | Content |
|---|---|
| [00-reading-method-and-branch-search.md](./foundation/00-reading-method-and-branch-search_EN.md) | How to read SGLang NPU source: search strategies, branch identification |
| [01-platform-detection-and-process-startup.md](./foundation/01-platform-detection-and-process-startup_EN.md) | `is_npu()`, platform detection, and process startup flow |
| [02-server-args-and-npu-defaults.md](./foundation/02-server-args-and-npu-defaults_EN.md) | Server arguments with NPU-specific defaults |
| [03-request-lifecycle-npu-branch-points.md](./foundation/03-request-lifecycle-npu-branch-points_EN.md) | Request lifecycle with NPU branch points annotated |
| [04-model-loading-dtype-and-layout.md](./foundation/04-model-loading-dtype-and-layout_EN.md) | Model loading, dtype handling, and tensor layout for NPU |
| [05-model-runner-forward-batch-and-input-buffers.md](./foundation/05-model-runner-forward-batch-and-input-buffers_EN.md) | ModelRunner forward, batches, and NPU input buffers |

### Component Deep Dives

| File | Content |
|---|---|
| [01-sglang-npu-component-map.md](./01-sglang-npu-component-map_EN.md) | Complete component map: SGLang ↔ sgl-kernel-npu |
| [15-distributed-hccl-and-communication.md](./15-distributed-hccl-and-communication_EN.md) | HCCL distributed communication on NPU |

### Examples

| File | Content |
|---|---|
| [00-glm-4.7-flash-end-to-end.md](./examples/00-glm-4.7-flash-end-to-end_EN.md) | End-to-end model execution: GLM-4.7-Flash on Ascend NPU |

## Reading Order

1. Start with Foundation 00 — understand the reading method
2. Foundation 01-02 — understand startup and configuration
3. Foundation 03 — trace the request lifecycle with NPU branches
4. 01-sglang-npu-component-map — see the full component landscape
5. Examples 00 — see a complete model example
6. Foundation 04-05 + 15 — dive into model loading, forward, and communication
