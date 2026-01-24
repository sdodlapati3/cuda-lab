# Exercise 01: Timeline Basics

> **Learn to read Nsight Systems timelines and identify GPU idle time**

## 🎯 Objectives

1. Generate an Nsight Systems profile
2. Identify GPU idle time in the timeline
3. Measure kernel launch overhead
4. Find synchronization bottlenecks

---

## 📋 Prerequisites

- NVIDIA GPU with CUDA support
- Nsight Systems installed (`nsys --version`)
- Basic CUDA kernel knowledge

---

## 🔧 Setup

```bash
cd /path/to/profiling-lab/01-nsight-systems/ex01-timeline-basics
make
```

---

## 📝 Exercise Steps

### Part 1: Profile the Baseline

We provide a simple CUDA program with intentional inefficiencies.

```bash
# Compile
nvcc -o baseline baseline.cu -O3

# Profile
nsys profile -o baseline_report ./baseline

# View stats
nsys stats baseline_report.nsys-rep
```

**Questions to answer:**
1. What is the total GPU active time?
2. What is the total application time?
3. Calculate GPU utilization: `GPU active / Total time`

### Part 2: Identify Idle Gaps

Open the report in Nsight Systems GUI (or export to analyze):

```bash
nsys-ui baseline_report.nsys-rep
# OR
nsys stats baseline_report.nsys-rep > stats.txt
```

**Look for:**
- Gaps between kernel executions
- Time spent in `cudaDeviceSynchronize()`
- Time spent in `cudaMemcpy()` (synchronous)

**Record your findings:**
| Kernel | Duration (ms) | Gap After (ms) |
|--------|---------------|----------------|
| kernel_1 | ? | ? |
| kernel_2 | ? | ? |
| kernel_3 | ? | ? |

### Part 3: Find Synchronization Bottlenecks

In the CUDA API row, look for:
- `cudaDeviceSynchronize()` calls
- `cudaStreamSynchronize()` calls
- `cudaMemcpy()` with `cudaMemcpyDeviceToHost`

**Question:** How much time is spent in synchronization calls?

### Part 4: Analyze Improved Version

```bash
nvcc -o improved improved.cu -O3
nsys profile -o improved_report ./improved
nsys stats improved_report.nsys-rep
```

**Compare:**
| Metric | Baseline | Improved | Improvement |
|--------|----------|----------|-------------|
| Total Time | ? ms | ? ms | ?% |
| GPU Utilization | ?% | ?% | +?% |
| Sync Time | ? ms | ? ms | -?% |

---

## 💻 Code Files

### baseline.cu (provided)

```cuda
#include <cuda_runtime.h>
#include <stdio.h>

#define N (1 << 24)  // 16M elements
#define BLOCK_SIZE 256

__global__ void vector_add(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void vector_scale(float *c, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = c[idx] * scale;
    }
}

int main() {
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    
    // Allocate host memory
    h_a = (float*)malloc(N * sizeof(float));
    h_b = (float*)malloc(N * sizeof(float));
    h_c = (float*)malloc(N * sizeof(float));
    
    // Initialize
    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }
    
    // Allocate device memory
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));
    
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // INEFFICIENT: Multiple separate operations with syncs
    for (int iter = 0; iter < 10; iter++) {
        // Copy to device (synchronous)
        cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);
        
        // Launch kernel
        vector_add<<<numBlocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        
        // Unnecessary sync!
        cudaDeviceSynchronize();
        
        // Another kernel
        vector_scale<<<numBlocks, BLOCK_SIZE>>>(d_c, 2.0f, N);
        
        // Unnecessary sync!
        cudaDeviceSynchronize();
        
        // Copy back (synchronous)
        cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);
    }
    
    printf("Result[0] = %f (expected 6.0)\n", h_c[0]);
    
    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    
    return 0;
}
```

### Your Task: improved.cu

Create an improved version that:
1. Uses pinned memory for faster transfers
2. Removes unnecessary `cudaDeviceSynchronize()` calls
3. Keeps data on GPU across iterations (don't copy back every time)

---

## ✅ Success Criteria

- [ ] Generated baseline profile
- [ ] Identified at least 3 sources of GPU idle time
- [ ] Created improved version with >2x speedup
- [ ] GPU utilization improved by at least 30%

---

## 🔑 Key Takeaways

1. **Unnecessary syncs kill performance** - Only sync when you need results on CPU
2. **Synchronous memcpy blocks** - Use async copies with streams
3. **Keep data on GPU** - Minimize host-device transfers
4. **Profile first** - Never optimize without measurement

---

## 📊 Expected Results

| Version | Total Time | GPU Util | Notes |
|---------|------------|----------|-------|
| Baseline | ~150 ms | ~20% | Many gaps |
| Improved | ~40 ms | ~70% | Overlapped |

Your actual numbers will vary based on GPU, but the improvement ratio should be similar.
