# Day 4: CUB Advanced

## Learning Objectives

- Use block-level and warp-level CUB
- Custom reduction operators
- Segmented operations

## Key Concepts

### Block-Level Primitives

Use within your kernels for cooperative operations:

```cpp
#include <cub/cub.cuh>

__global__ void my_kernel(float* data, float* block_sums) {
    // Specialize for block of 256 threads
    typedef cub::BlockReduce<float, 256> BlockReduce;
    
    // Allocate shared memory
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    float val = data[threadIdx.x];
    
    // Collective reduction
    float block_sum = BlockReduce(temp_storage).Sum(val);
    
    if (threadIdx.x == 0) {
        block_sums[blockIdx.x] = block_sum;
    }
}
```

### Warp-Level Primitives

```cpp
typedef cub::WarpReduce<float> WarpReduce;
__shared__ typename WarpReduce::TempStorage temp[WARPS_PER_BLOCK];

int warp_id = threadIdx.x / 32;
float warp_sum = WarpReduce(temp[warp_id]).Sum(val);
```

### Custom Reduction Operators

```cpp
struct MaxOp {
    __device__ float operator()(float a, float b) {
        return (a > b) ? a : b;
    }
};

// Use with:
float result = BlockReduce(temp_storage).Reduce(val, MaxOp());
```

## Build & Run

```bash
./build.sh
./build/cub_advanced
```
