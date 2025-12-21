# Day 4: Block Size Selection

## Learning Objectives

- Choose optimal block sizes
- Understand block size constraints
- Use occupancy API for auto-tuning

## Key Concepts

### Block Size Constraints

```
- Must be multiple of 32 (warp size) for efficiency
- Max 1024 threads per block
- Affects register/smem per thread calculation
- Affects number of blocks per SM
```

### Common Block Sizes

| Size | Warps | Use Case |
|------|-------|----------|
| 128 | 4 | Low resource, high occupancy |
| 256 | 8 | Good default |
| 512 | 16 | When resources allow |
| 1024 | 32 | Max parallelism per block |

### Auto-tuning with API

```cpp
cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, kernel, smem, 0);
```

## Build & Run

```bash
./build.sh
./build/block_size_tuning
```
