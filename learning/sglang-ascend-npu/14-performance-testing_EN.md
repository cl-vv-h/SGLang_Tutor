[中文](./14-performance-testing.md) | [English](./14-performance-testing_EN.md)

# 14. Performance Testing

## 1. Key Metrics

| Metric | Definition | What It Reveals |
|---|---|---|
| QPS | Queries Per Second | System throughput |
| TTFT | Time To First Token | Prefill + scheduling latency |
| ITL | Inter-Token Latency | Decode step efficiency |
| TPS | Tokens Per Second (output) | Generation speed |
| Input TPS | Tokens Per Second (input) | Prefill throughput |
| P95/P99 | Tail latency | Worst-case performance |

## 2. Workload Design

| Type | Input Length | Output Length | Tests |
|---|---|---|---|
| Short-Short | 128 | 128 | Baseline latency |
| Long-Short | 4096 | 128 | Prefill throughput |
| Short-Long | 128 | 4096 | Decode stability |
| Long-Long | 4096 | 4096 | Worst-case stress |

## 3. Running Benchmarks

```bash
# Using SGLang benchmark tool
python -m sglang.bench_serving \
  --backend sglang \
  --host localhost --port 8000 \
  --num-prompts 1000 \
  --request-rate 10

# Key parameters:
# --request-rate: QPS target (inf = max throughput)
# --num-prompts: total requests
# --dataset-name: sharegpt / random
```

## 4. Analysis Framework

```text
High TTFT?
  → Check prefill speed, scheduling delay, queuing

High ITL?
  → Check decode graph replay, KV cache bandwidth

Low QPS?
  → Check batch utilization, TP efficiency

High P99?
  → Check tail issues: long prompts, retraction, GC pauses
```

## 5. Comparative Testing

Always compare against a controlled baseline:
- Same model, same hardware, same workload
- Change one variable at a time (TP size, graph config, chunk size)
- Record: QPS, TTFT (avg/p95/p99), ITL (avg/p95/p99), error rate
