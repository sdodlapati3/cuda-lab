# Day 5: Cooperative Groups

## Learning Objectives

- Master the Cooperative Groups API
- Understand flexible thread grouping
- Use tiled partitions for warp-level code

## Key Concepts

### What are Cooperative Groups?

A flexible API for synchronizing threads at any granularity:
- Warp-level
- Arbitrary subsets
- Block-level
- Grid-level

### Key Types

| Type | Description |
|------|-------------|
| `thread_block` | All threads in block |
| `thread_block_tile<N>` | Tile of N threads (warp or smaller) |
| `grid_group` | All threads in grid |
| `coalesced_group` | Currently active threads |

### Basic Usage

```cuda
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void kernel() {
    cg::thread_block block = cg::this_thread_block();
    block.sync();  // Like __syncthreads()
    
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    warp.sync();   // Warp sync
    
    // Warp shuffle
    float val = warp.shfl_down(my_val, 1);
}
```

### Tiled Partitions

Divide blocks into fixed-size tiles:
```cuda
cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
cg::thread_block_tile<16> half_warp = cg::tiled_partition<16>(block);
cg::thread_block_tile<8> quarter_warp = cg::tiled_partition<8>(block);
```

### Coalesced Groups

Work with currently active threads:
```cuda
cg::coalesced_group active = cg::coalesced_threads();
// active.size() = number of active threads
// Useful in divergent code
```

### Grid-Level Sync

Requires cooperative launch:
```cuda
cg::grid_group grid = cg::this_grid();
grid.sync();  // ALL threads in grid sync
```

## Exercises

1. **Thread block tiles**: Refactor reduction
2. **Coalesced groups**: Handle divergent code
3. **Grid sync**: Multi-block algorithms

## Build & Run

```bash
./build.sh
./build/cg_demo
./build/cg_reduction
```
