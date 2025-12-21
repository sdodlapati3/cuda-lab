/**
 * Day 3: Simple Add - For PTX/SASS analysis
 * 
 * A minimal kernel to make PTX/SASS readable.
 * Complex kernels have too much code to analyze by hand.
 */

#include <cstdio>

// Minimal kernel for clean PTX/SASS output
__global__ void simple_add(const float* __restrict__ a, 
                           const float* __restrict__ b, 
                           float* __restrict__ c, 
                           int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// A more interesting kernel: fused multiply-add
__global__ void fma_kernel(const float* __restrict__ a, 
                           const float* __restrict__ b,
                           const float* __restrict__ c,
                           float* __restrict__ d,
                           float alpha,
                           int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // d = a * b + c * alpha
        // Should compile to FFMA instructions
        d[idx] = a[idx] * b[idx] + c[idx] * alpha;
    }
}

// Reduction kernel (more complex control flow)
__global__ void reduce_sum(const float* __restrict__ input, 
                           float* __restrict__ output,
                           int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load to shared memory
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

int main() {
    printf("This binary is for PTX/SASS extraction.\n");
    printf("Use:\n");
    printf("  cuobjdump -ptx simple_add\n");
    printf("  cuobjdump -sass simple_add\n");
    printf("  cuobjdump -res-usage simple_add\n");
    return 0;
}
