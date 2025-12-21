/**
 * Day 1: Profile Demo - Comparing Kernel Variations
 * 
 * Same operation, different implementations for profiling comparison.
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

// ============================================================================
// Version 1: Naive (uncoalesced access pattern)
// ============================================================================
__global__ void saxpy_naive(int n, float a, float* x, float* y) {
    int stride = blockDim.x * gridDim.x;
    for (int i = threadIdx.x; i < n; i += stride) {
        y[i] = a * x[i] + y[i];
    }
}

// ============================================================================
// Version 2: Coalesced access
// ============================================================================
__global__ void saxpy_coalesced(int n, float a, float* x, float* y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = a * x[idx] + y[idx];
    }
}

// ============================================================================
// Version 3: Grid-stride loop (flexible)
// ============================================================================
__global__ void saxpy_grid_stride(int n, float a, float* x, float* y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        y[i] = a * x[i] + y[i];
    }
}

// ============================================================================
// Version 4: Vectorized (float4)
// ============================================================================
__global__ void saxpy_vectorized(int n, float a, float4* x, float4* y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n / 4) {
        float4 vx = x[idx];
        float4 vy = y[idx];
        vy.x = a * vx.x + vy.x;
        vy.y = a * vx.y + vy.y;
        vy.z = a * vx.z + vy.z;
        vy.w = a * vx.w + vy.w;
        y[idx] = vy;
    }
}

int main() {
    printf("SAXPY Comparison for Profiling\n");
    printf("==============================\n\n");
    
    const int N = 1 << 24;  // 16M
    size_t size = N * sizeof(float);
    
    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, size));
    CUDA_CHECK(cudaMalloc(&d_y, size));
    
    // Initialize
    float* h_x = (float*)malloc(size);
    float* h_y = (float*)malloc(size);
    for (int i = 0; i < N; i++) {
        h_x[i] = 1.0f;
        h_y[i] = 2.0f;
    }
    CUDA_CHECK(cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice));
    
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    float a = 2.0f;
    
    // Reset and run each version
    printf("1. Naive (strided access)\n");
    CUDA_CHECK(cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice));
    saxpy_naive<<<numBlocks, blockSize>>>(N, a, d_x, d_y);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("2. Coalesced (simple indexing)\n");
    CUDA_CHECK(cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice));
    saxpy_coalesced<<<numBlocks, blockSize>>>(N, a, d_x, d_y);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("3. Grid-stride loop\n");
    CUDA_CHECK(cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice));
    saxpy_grid_stride<<<256, 256>>>(N, a, d_x, d_y);  // Fewer blocks
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("4. Vectorized (float4)\n");
    CUDA_CHECK(cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice));
    saxpy_vectorized<<<(N/4 + blockSize - 1) / blockSize, blockSize>>>(
        N, a, (float4*)d_x, (float4*)d_y);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    free(h_x);
    free(h_y);
    
    printf("\nProfile and compare with:\n");
    printf("  ncu --set full ./build/profile_demo\n");
    
    return 0;
}
