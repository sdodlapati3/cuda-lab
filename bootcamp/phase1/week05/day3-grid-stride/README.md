# Day 3: Grid-Stride Loops

## Learning Objectives

- Write kernels that handle any data size
- Understand the grid-stride pattern
- Know when and why to use grid-stride loops

## The Grid-Stride Pattern

### Problem
Launching enough threads for huge arrays is wasteful:
- 1 billion elements = 1 billion threads?
- Kernel launch overhead increases with grid size
- Occupancy may actually decrease

### Solution: Grid-Stride Loop

```cuda
__global__ void process(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        data[i] = process_element(data[i]);
    }
}
```

Each thread processes multiple elements, striding by the total grid size.

### Visualization

```
Data:    [0][1][2][3][4][5][6][7][8][9][10][11][12][13][14][15]...
Grid:    <------- 8 threads ------->
Thread 0: [0]            [8]              [16]...
Thread 1:    [1]            [9]              [17]...
Thread 2:       [2]            [10]             [18]...
...
Thread 7:             [7]            [15]             [23]...
```

## Benefits

1. **Reusable launch configuration**: Same kernel for any N
2. **Better occupancy**: Launch optimal number of blocks
3. **Coalesced access**: Adjacent threads access adjacent memory
4. **Persistent kernels**: Foundation for advanced patterns

## Exercises

1. **Compare approaches**: Measure 1-thread-per-element vs grid-stride
2. **Large array**: Process 100M element array efficiently
3. **Benchmark**: Find optimal grid size for your GPU

## Build & Run

```bash
./build.sh
./build/grid_stride_demo
./build/large_array
```
