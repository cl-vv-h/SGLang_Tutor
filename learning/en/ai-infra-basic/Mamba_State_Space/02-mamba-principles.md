# Mamba Principles: Selective Scan & State Space

## 1. From Linear SSM to Selective SSM

The basic SSM:

```math
\mathrm{state\_t} = A \cdot  \mathrm{state}_{t-1} + B \cdot  x_{t}
```

```math
y_{t}=C\cdot \mathrm{state\_t}
```

In this linear form, parameters `A`, `B`, `C` are fixed regardless of input — the system is Linear Time-Invariant (LTI). LTI SSMs have efficient convolution-based computation but can't selectively focus on or ignore specific inputs.

**Mamba's key innovation**: Make parameters input-dependent:

```math
B_{t}=f_{B}\left(x_{t}\right)
```

```math
C_{t}=f_{C}\left(x_{t}\right)
```

```math
\mathrm{Delta\_t}=f_{\mathrm{Delta}}\left(x_{t}\right)\qquad\text{step size, controls how much state updates}
```

This makes the system "selective" — different tokens can cause different amounts of state retention vs. update.

## 2. Discretization

Continuous-time SSM parameters must be discretized for token-by-token processing. Using Zero-Order Hold (ZOH):

```math
A_{\mathrm{bar}}=\mathrm{exp}\left(\mathrm{Delta}\cdot A\right)
```

```math
B_{\mathrm{bar}} = \left(\mathrm{Delta} \cdot  A\right)^{-1} \cdot  \left(\mathrm{exp}\left(\mathrm{Delta} \cdot  A\right) - I\right) \cdot  \mathrm{Delta} \cdot  B
```

Simplified in practice:

```math
A_{\mathrm{bar}}=\mathrm{exp}\left(\mathrm{Delta}\cdot A\right)
```

```math
B_{\mathrm{bar}}=\mathrm{Delta}\cdot B
```

The discrete recurrence:

```math
\mathrm{state\_t} = A_{\mathrm{bar}} \cdot  \mathrm{state}_{t-1} + B_{\mathrm{bar}} \cdot  x_{t}
```

A large `Delta` means the state focuses more on the current input; a small `Delta` means it retains more history.

## 3. Selective Scan Algorithm

Mamba's core computation is the **selective scan**: a parallel prefix-sum-like operation that computes all `state_t` and `y_t` efficiently.

For each channel c:

For each position t:

```math
{\mathrm{state}}_{t,c}={A_{\mathrm{bar}}}_{t,c}\cdot {\mathrm{state}}_{t-1,c}+{B_{\mathrm{bar}}}_{t,c}\cdot {x}_{t,c}
```

```math
{y}_{t,c}={C}_{t,c}\cdot {\mathrm{state}}_{t,c}
```

While this looks sequential, specialized parallel scan kernels (like `selective_scan` in CUDA/Triton) compute it in O(log N) parallel steps.

## 4. Mamba Block Structure

A Mamba block typically contains:

```math
x=\mathrm{input}\qquad\mathrm{shape}:\left[B,S,H\right]
```

```math
\mathrm{residual}=x
```

```text
# Branch 1: Main path through SSM
```

```math
x_{\mathrm{norm}}=\mathrm{RMSNorm}\left(x\right)
```

```math
x_{\mathrm{proj}} = \mathrm{Linear}\left(x_{\mathrm{norm}}\right) \qquad\text{Project to expanded dimension}
```

```math
x_{\mathrm{conv}} = \mathrm{CausalConv1d}\left(x_{\mathrm{proj}}\right) \qquad\text{Local context via convolution}
```

```math
x_{\mathrm{silu}}=\mathrm{SiLU}\left(x_{\mathrm{conv}}\right)\qquad\text{Activation}
```

```text
# SSM core
```

```math
x_{\mathrm{ssm}} = \mathrm{SelectiveScan}\left(x_{\mathrm{silu}}\right) \qquad\text{State space computation}
```

```math
y_{\mathrm{ssm}}=\mathrm{SiLU}\left(x_{\mathrm{proj}}\right)\odot x_{\mathrm{ssm}}\qquad\text{Output gate}
```

```text
# Branch 2: Gated linear (like FFN)
```

```math
x_{\mathrm{out}} = \mathrm{Linear}\left(y_{\mathrm{ssm}}\right) \qquad\text{Project back to H}
```

```math
\mathrm{output}=\mathrm{residual}+x_{\mathrm{out}}
```

## 5. Mamba vs Transformer Attention

| Property | Transformer Attention | Mamba SSM |
|---|---|---|
| Context representation | Explicit KV pairs per token | Compressed recurrent state |
| Memory growth | O(n) with sequence length | O(1) with sequence length |
| Training parallelism | O(n²) attention matrix | O(n) via parallel scan |
| Long-range dependency | Direct token-to-token | Implicit via state propagation |
| Hardware efficiency | Memory-bandwidth bound for decode | Compute-bound, constant memory |

## 6. SGLang-Specific Considerations

- **MambaStatePool** stores one state per layer per active request
- State size is $`\left[B,D_{\mathrm{state}}\right]`$ per layer, where `D_state` is the SSM state dimension (typically 16-128)
- During prefill, state is computed for all prompt tokens; the final state is saved
- During decode, state is updated incrementally with each new token
- **Mamba radix cache**: For prefix sharing, Mamba states must be checkpointed at radix tree nodes rather than sharing KV pages
