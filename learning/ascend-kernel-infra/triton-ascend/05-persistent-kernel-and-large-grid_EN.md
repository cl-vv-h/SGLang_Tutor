[中文](./05-persistent-kernel-and-large-grid.md) | [English](./05-persistent-kernel-and-large-grid_EN.md)

# Triton-Ascend 05: Persistent Kernel & Large Grid

## Persistent Kernel Pattern

Instead of launching grid-size blocks (one-and-done), persistent kernels keep blocks alive, fetching work from a task queue:

```python
@triton.jit
def persistent_matmul(A, B, C, M, N, K, ...):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # Persistent: iterate over multiple tiles
    for tile_id in range(pid, num_pid_m * num_pid_n, grid_size):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        # Process tile (pid_m, pid_n)
```

## When to Use Persistent Kernels

| Scenario | Regular Grid | Persistent |
|---|---|---|
| Work fits in grid | ✅ Simpler | Overhead not worth it |
| Work >> grid capacity | Load imbalance possible | ✅ Better load balance |
| Variable work per tile | Tail effect | ✅ Dynamic scheduling |

## Large Grid on Ascend

```text
Ascend constraint: physical AI Cores are limited (e.g., 32 cores per NPU)

Large grid (> physical cores):
  → Triton-Ascend splits into waves
  → Each wave: grid_size cores active, rest idle
  → Multiple waves execute sequentially

Persistent kernel with large grid:
  → grid_size = physical_cores (max occupancy)
  → Each core processes multiple tiles (no idle waves)
```

## Auto-Blockify

Triton-Ascend can auto-blockify: automatically merge multiple small blocks into one larger block for better utilization. This is transparent to the kernel author.
