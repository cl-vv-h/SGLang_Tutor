[中文](./02-mamba-principles.md) | [English](./02-mamba-principles_EN.md)

# Mamba Principles: Selective Scan & State Space

## 1. From Linear SSM to Selective SSM

The basic SSM:

```text
state_t = A * state_{t-1} + B * x_t
y_t     = C * state_t
```

In this linear form, parameters `A`, `B`, `C` are fixed regardless of input — the system is Linear Time-Invariant (LTI). LTI SSMs have efficient convolution-based computation but can't selectively focus on or ignore specific inputs.

**Mamba's key innovation**: Make parameters input-dependent:

```text
B_t = f_B(x_t)
C_t = f_C(x_t)
Delta_t = f_Delta(x_t)    # step size, controls how much state updates
```

This makes the system "selective" — different tokens can cause different amounts of state retention vs. update.

## 2. Discretization

Continuous-time SSM parameters must be discretized for token-by-token processing. Using Zero-Order Hold (ZOH):

```text
A_bar = exp(Delta * A)
B_bar = (Delta * A)^{-1} * (exp(Delta * A) - I) * Delta * B
```

Simplified in practice:

```text
A_bar = exp(Delta * A)
B_bar = Delta * B
```

The discrete recurrence:

```text
state_t = A_bar * state_{t-1} + B_bar * x_t
```

A large `Delta` means the state focuses more on the current input; a small `Delta` means it retains more history.

## 3. Selective Scan Algorithm

Mamba's core computation is the **selective scan**: a parallel prefix-sum-like operation that computes all `state_t` and `y_t` efficiently.

```text
For each channel c:
    For each position t:
        state[t,c] = A_bar[t,c] * state[t-1,c] + B_bar[t,c] * x[t,c]
        y[t,c] = C[t,c] * state[t,c]
```

While this looks sequential, specialized parallel scan kernels (like `selective_scan` in CUDA/Triton) compute it in O(log N) parallel steps.

## 4. Mamba Block Structure

A Mamba block typically contains:

```text
x = input                         [B,S,H]
residual = x

# Branch 1: Main path through SSM
x_norm = RMSNorm(x)
x_proj = Linear(x_norm)           # Project to expanded dimension
x_conv = CausalConv1d(x_proj)     # Local context via convolution
x_silu = SiLU(x_conv)             # Activation

# SSM core
x_ssm = SelectiveScan(x_silu)     # State space computation
y_ssm = SiLU(x_proj) * x_ssm      # Output gate

# Branch 2: Gated linear (like FFN)
x_out = Linear(y_ssm)             # Project back to H
output = residual + x_out
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
- State size is `[B, D_state]` per layer, where `D_state` is the SSM state dimension (typically 16-128)
- During prefill, state is computed for all prompt tokens; the final state is saved
- During decode, state is updated incrementally with each new token
- **Mamba radix cache**: For prefix sharing, Mamba states must be checkpointed at radix tree nodes rather than sharing KV pages
