[中文](./04-platform-tiling-and-workspace-contracts.md) | [English](./04-platform-tiling-and-workspace-contracts_EN.md)

# Ascend C 04: Platform, Tiling, Workspace & Host/Device Contracts

## Platform Information

```cpp
// Platform provides hardware capability info
PlatformInfo platform;
int cube_core_count = platform.GetCoreCount(CoreType::AIC);
int vector_core_count = platform.GetCoreCount(CoreType::AIV);
int ub_size = platform.GetUBSize();      // Unified Buffer per core
int l1_size = platform.GetL1Size();      // L1 buffer per core
```

## Tiling API

```cpp
// Tiling: compute how to split work across cores
class TilingData {
public:
    uint32_t block_dim;     // Number of cores to use
    uint32_t tile_m;        // Tile size in M dimension
    uint32_t tile_n;        // Tile size in N dimension
    uint64_t workspace_size; // Scratch memory needed
};

TilingData ComputeTiling(int M, int N, int K) {
    // Balance: enough work per core, fits in UB
    TilingData tiling;
    tiling.block_dim = min(MAX_CORES, (M + TILE_M - 1) / TILE_M);
    tiling.tile_m = (M + tiling.block_dim - 1) / tiling.block_dim;
    tiling.tile_n = min(N, TILE_N_MAX);
    tiling.workspace_size = tiling.tile_m * tiling.tile_n * sizeof(float);
    return tiling;
}
```

## Workspace Contracts

```cpp
// Workspace: temporary scratch memory for kernel execution
// Contract: host allocates based on tiling.workspace_size
// Device uses within kernel, must not exceed declared size

// Host side:
void* workspace = malloc(tiling.workspace_size);
LaunchKernel(..., workspace, tiling.workspace_size);

// Device side:
__aicore__ void Process(uint8_t* workspace, uint64_t workspace_size) {
    // Use workspace for temporary buffers
    LocalTensor<float> temp = CreateTempTensor(workspace, 0, tile_size);
}
```

## Host/Device Execution Plan

```text
Host (CPU):
  1. Read platform info (core count, UB size)
  2. Compute tiling (blockDim, tile sizes)
  3. Allocate workspace
  4. Launch kernel with tiling parameters
  5. Wait for completion or overlap with next launch

Device (NPU):
  1. Each core receives its tile range
  2. CopyIn tile from GM → UB
  3. Compute on UB data
  4. CopyOut results UB → GM
  5. Signal completion
```
