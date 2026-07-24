[中文](./02-cann-stack-and-boundaries.md) | [English](./02-cann-stack-and-boundaries_EN.md)

# 02. CANN Stack & Boundaries

## CANN Full Stack

```text
┌─────────────────────────────────────────┐
│              Applications                │
│    (SGLang, PyTorch, TensorFlow)         │
├─────────────────────────────────────────┤
│         Framework Adapters               │
│    (torch_npu, TF adapter)               │
├─────────────────────────────────────────┤
│        CANN Development Tools            │
│  ┌──────────┬──────────┬──────────┐     │
│  │ Ascend C │  Tiling  │  MSProf  │     │
│  │ Compiler │   API    │  Debug   │     │
│  └──────────┴──────────┴──────────┘     │
├─────────────────────────────────────────┤
│         CANN Runtime                     │
│  ┌──────────┬──────────┬──────────┐     │
│  │  ACLNN   │  ACLOP   │  Graph   │     │
│  │(Neural)  │(Operator)│ Compiler │     │
│  └──────────┴──────────┴──────────┘     │
│  ┌──────────────────────────────────┐   │
│  │     AscendCL (ACL) Runtime       │   │
│  │  Streams, Events, Memory, Exec   │   │
│  └──────────────────────────────────┘   │
│  ┌──────────────────────────────────┐   │
│  │  HCCL (Collective Communication) │   │
│  └──────────────────────────────────┘   │
├─────────────────────────────────────────┤
│         Driver / Firmware                │
│    (Device management, DMA, interrupts)  │
├─────────────────────────────────────────┤
│       Ascend NPU Hardware                │
│  (Da Vinci AI Core, HBM, HCCS, NIC)     │
└─────────────────────────────────────────┘
```

## Responsibility Boundaries

| Layer | Responsibility | Not Responsible For |
|---|---|---|
| **Driver/Firmware** | Device discovery, DMA, interrupt handling, memory mapping | Operator logic, tensor shapes |
| **ACL Runtime** | Stream management, memory allocation, execution submission | Operator implementation |
| **ACLNN/ACLOP** | Neural network operators (Conv, MatMul, Norm) | Custom operator optimization |
| **Graph Compiler** | Graph-level optimization, fusion, memory planning | Individual operator correctness |
| **HCCL** | All-reduce, all-gather, reduce-scatter, all-to-all | Application-level scheduling |
| **Tiling API** | Compute data partitioning across AI Cores | Operator logic itself |
| **Ascend C** | Kernel programming language + compiler | Runtime execution scheduling |
| **torch_npu** | PyTorch → CANN bridge, op registration, format handling | Custom kernel implementation |

## Key Distinctions

- **CANN ≠ CUDA**: CANN is a full toolkit (compiler, runtime, operator library); CUDA is primarily the programming model + driver
- **ACLNN ≠ cuDNN**: ACLNN covers neural network ops; ACLOP covers general-purpose ops — together roughly equivalent to cuBLAS + cuDNN
- **HCCL ≈ NCCL**: Both are collective communication libraries, but HCCL targets Ascend's HCCS/RoCE interconnect
- **Ascend C ≈ CUDA C++**: Both are low-level kernel programming languages giving direct hardware access
