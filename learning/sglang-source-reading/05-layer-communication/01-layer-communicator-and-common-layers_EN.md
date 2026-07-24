[中文](./01-layer-communicator-and-common-layers.md) | [English](./01-layer-communicator-and-common-layers_EN.md)

# LayerCommunicator & Common Layers

## 1. LayerCommunicator: The Hidden Glue

`LayerCommunicator` abstracts all distributed communication within a single decoder layer. Following `DecoderLayer.forward()`:

```text
DecoderLayer.forward(hidden_states):
  1. Attention sublayer:
     - QKV projection (local, possibly TP-split)
     - Attention computation (local, within visible range)
     - Output projection (local, possibly TP-split)
     - LayerCommunicator: all-reduce attention output if TP
  
  2. FFN/MoE sublayer:
     - Router (for MoE): local computation
     - LayerCommunicator: all-to-all dispatch tokens to expert-holding ranks
     - Expert computation: local GEMM
     - LayerCommunicator: all-to-all combine results back
     - LayerCommunicator: all-reduce if dense FFN with TP
```

## 2. Communication Patterns

| Pattern | When | Data Movement |
|---|---|---|
| All-reduce | TP attention/FFN output | Sum partial results across TP ranks |
| Reduce-scatter | TP weight gradient (training) | Scatter summed gradients |
| All-gather | DP attention, TP input | Gather tensors across ranks |
| All-to-all | MoE EP dispatch/combine | Redistribute tokens to expert ranks |
| Send/Recv | PP activation passing | Pass tensors between PP stages |

## 3. TP Communication Detail

With `TP_size = 4`, each attention layer:

```text
Rank 0: computes Q[0:Nq/4], K[0:Nkv/4], V[0:Nkv/4]
Rank 1: computes Q[Nq/4:Nq/2], K[Nkv/4:Nkv/2], V[Nkv/4:Nkv/2]
...
After attention: each rank has partial output [T, H/TP_size]
LayerCommunicator.all_reduce(): sum → full output [T, H]
```

## 4. EP Communication Detail

With `EP_size = 4`, `E = 32` experts:

```text
Rank 0: Experts 0-7
Rank 1: Experts 8-15
Rank 2: Experts 16-23
Rank 3: Experts 24-31

After router Top-K:
  all-to-all dispatch: send token rows to expert-owning ranks
  Expert GEMM: each rank computes its experts
  all-to-all combine: send results back to token-owning ranks
```

## 5. Common Layer Types

| Layer | Source | TP Strategy | EP Strategy |
|---|---|---|---|
| `RadixAttention` | `layers/attention/` | Column-parallel QKV, row-parallel O | N/A |
| `RMSNorm` | `layers/norm.py` | No communication needed | N/A |
| `Dense FFN` | Various | Column-parallel gate+up, row-parallel down | N/A |
| `SparseMoE` | `layers/moe/` | Optional TP within experts | All-to-all dispatch/combine |
| `Mamba/SSM` | `layers/mamba/` | State parallelism | State transfer |

## 6. Reading Path

1. Find `DecoderLayer.forward()` in a model file (e.g., `llama.py`)
2. Follow `LayerCommunicator` calls in attention and FFN sublayers
3. Understand TP all-reduce placement (after output projection)
4. For MoE models, trace `fused_experts.py` for dispatch/combine
5. Look at `parallel_state.py` for group initialization
