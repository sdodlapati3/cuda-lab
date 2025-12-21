/**
 * Day 5: Bottleneck Examples
 * 
 * Common performance problems visible in Nsight Systems.
 */

#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

__global__ void work_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        for (int i = 0; i < 100; i++) {
            val = sinf(val) + 0.1f;
        }
        data[idx] = val;
    }
}

// ============================================================================
// Bottleneck 1: Excessive Synchronization
// ============================================================================
void bottleneck_sync() {
    printf("\n=== Bottleneck: Excessive Sync ===\n");
    
    const int N = 1 << 18;
    float* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
    
    printf("BAD: Sync after every kernel...\n");
    for (int i = 0; i < 10; i++) {
        work_kernel<<<(N + 255) / 256, 256>>>(d_data, N);
        CUDA_CHECK(cudaDeviceSynchronize());  // Creates gaps!
    }
    
    printf("GOOD: Single sync at end...\n");
    for (int i = 0; i < 10; i++) {
        work_kernel<<<(N + 255) / 256, 256>>>(d_data, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());  // One sync
    
    CUDA_CHECK(cudaFree(d_data));
}

// ============================================================================
// Bottleneck 2: Pageable Memory
// ============================================================================
void bottleneck_pageable() {
    printf("\n=== Bottleneck: Pageable vs Pinned Memory ===\n");
    
    const int N = 1 << 24;
    size_t size = N * sizeof(float);
    
    float* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, size));
    
    // Pageable (slow)
    printf("Pageable memory transfer...\n");
    float* h_pageable = (float*)malloc(size);
    CUDA_CHECK(cudaMemcpy(d_data, h_pageable, size, cudaMemcpyHostToDevice));
    free(h_pageable);
    
    // Pinned (fast)
    printf("Pinned memory transfer...\n");
    float* h_pinned;
    CUDA_CHECK(cudaMallocHost(&h_pinned, size));
    CUDA_CHECK(cudaMemcpy(d_data, h_pinned, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaFreeHost(h_pinned));
    
    CUDA_CHECK(cudaFree(d_data));
}

// ============================================================================
// Bottleneck 3: Small Grid
// ============================================================================
__global__ void underutilized_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        for (int i = 0; i < 1000; i++) {
            data[idx] = sinf(data[idx]);
        }
    }
}

void bottleneck_small_grid() {
    printf("\n=== Bottleneck: Small Grid ===\n");
    
    const int N = 256;  // Only 1 block of work!
    float* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
    
    printf("Only 1 block - GPU is underutilized...\n");
    underutilized_kernel<<<1, 256>>>(d_data, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("Many blocks - GPU is busy...\n");
    const int N_BIG = 1 << 20;
    float* d_big;
    CUDA_CHECK(cudaMalloc(&d_big, N_BIG * sizeof(float)));
    underutilized_kernel<<<(N_BIG + 255) / 256, 256>>>(d_big, N_BIG);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_big));
}

int main() {
    printf("Bottleneck Examples for Nsight Systems\n");
    printf("======================================\n");
    printf("Profile with: nsys profile -o bottlenecks ./build/bottleneck_examples\n");
    
    bottleneck_sync();
    bottleneck_pageable();
    bottleneck_small_grid();
    
    printf("\nCompare timeline patterns before/after fixes!\n");
    
    return 0;
}
