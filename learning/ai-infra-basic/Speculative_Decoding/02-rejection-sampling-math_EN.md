[中文](./02-rejection-sampling-math.md) | [English](./02-rejection-sampling-math_EN.md)

# Rejection Sampling Math for Speculative Decoding

## 1. The Goal

Guarantee that speculative decoding produces tokens from exactly the same distribution as the target model would produce autoregressively.

## 2. Problem Setup

- Target model distribution: `p(x)` (the distribution we want)
- Draft model distribution: `q(x)` (the distribution we use for proposals)
- Draft proposes token `x` from `q(x)`

We need a rule to accept or reject `x` such that accepted tokens follow `p(x)`.

## 3. Rejection Sampling Rule

For a proposed token `x`:

```text
Accept if: p(x) >= q(x)              → always accept
Accept with probability p(x)/q(x) if: p(x) < q(x)    → probabilistic accept
```

This can be implemented as:

```python
def accept_token(proposed_token, p_target, q_draft):
    r = random()  # uniform [0, 1)
    if r < min(1.0, p_target(proposed_token) / q_draft(proposed_token)):
        return True   # Accept
    else:
        return False  # Reject
```

## 4. After Rejection: Resampling

When a token is rejected, we need to sample a replacement from a modified distribution:

```text
p_resample(x) = normalize(max(0, p(x) - q(x)))
```

This "patches" the gap between target and draft distributions. The combination of accept/reject + resample ensures the final output exactly follows `p(x)`.

## 5. Multi-Token Speculation

For K draft tokens `x_1, x_2, ..., x_K`:

```text
For i in 1..K:
    q_i(x) = q(x | prefix, x_1, ..., x_{i-1})  # Draft's distribution at step i
    p_i(x) = p(x | prefix, x_1, ..., x_{i-1})  # Target's distribution at step i

    If accept(x_i, p_i, q_i):
        output x_i, continue
    Else:
        Resample from max(0, p_i - q_i)
        Stop verification (remaining draft tokens are invalid)
```

## 6. Expected Acceptance Rate

The probability of accepting a draft token is:

```text
P(accept) = sum_x min(q(x), p(x))
```

This equals 1 when `q = p` (perfect draft). The closer `q` is to `p`, the more tokens are accepted per verification.

## 7. SGLang-Specific Details

- `spec_info` stores draft tokens, probabilities, and acceptance metadata
- The draft KV cache is only committed for accepted positions
- Rejected draft tokens' KV cache entries are discarded
- `BatchResultProcessor` orchestrates the accept/reject/resample logic
