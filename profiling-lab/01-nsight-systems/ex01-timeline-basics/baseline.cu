// baseline.cu - Intentionally inefficient CUDA program for profiling practice
// This code has multiple performance issues for you to find with Nsight Systems

#include <cuda_runtime.h>
#include <stdio.h>

#define N (1 << 24)  // 16M elements
#define BLOCK_SIZE 256

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

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
    
    // Allocate host memory (pageable - slow transfers!)
    h_a = (float*)malloc(N * sizeof(float));
    h_b = (float*)malloc(N * sizeof(float));
    h_c = (float*)malloc(N * sizeof(float));
    
    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Failed to allocate host memory\n");
        exit(EXIT_FAILURE);
    }
    
    // Initialize host data
    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float)));
    
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    printf("Running baseline (inefficient) version...\n");
    printf("Array size: %d elements (%.2f MB)\n", N, N * sizeof(float) / 1e6);
    
    // Create events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    
    // INEFFICIENT: Multiple separate operations with syncs
    for (int iter = 0; iter < 10; iter++) {
        // Issue 1: Synchronous copy with pageable memory
        CUDA_CHECK(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));
        
        // Launch first kernel
        vector_add<<<numBlocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        
        // Issue 2: Unnecessary synchronization between kernels!
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Launch second kernel
        vector_scale<<<numBlocks, BLOCK_SIZE>>>(d_c, 2.0f, N);
        
        // Issue 3: Another unnecessary sync!
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Issue 4: Copying back every iteration when we only need final result
        CUDA_CHECK(cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost));
    }
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    // Verify result
    printf("Result[0] = %f (expected 6.0)\n", h_c[0]);
    printf("Result[N-1] = %f (expected 6.0)\n", h_c[N-1]);
    printf("Total time: %.2f ms\n", milliseconds);
    
    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);
    
    return 0;
}
