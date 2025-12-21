/**
 * Day 2: Memory Access Patterns
 * 
 * Different patterns to analyze with ncu memory metrics.
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
// Pattern 1: Coalesced access (optimal)
// ============================================================================
__global__ void coalesced_access(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}

// ============================================================================
// Pattern 2: Strided access (suboptimal)
// ============================================================================
__global__ void strided_access(float* data, int n, int stride) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}

// ============================================================================
// Pattern 3: Random access (poor)
// ============================================================================
__global__ void random_access(float* data, int* indices, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int random_idx = indices[idx];
        data[random_idx] = data[random_idx] * 2.0f;
    }
}

// ============================================================================
// Pattern 4: Misaligned access
// ============================================================================
__global__ void misaligned_access(float* data, int n, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx + offset < n) {
        data[idx + offset] = data[idx + offset] * 2.0f;
    }
}

// ============================================================================
// Pattern 5: Vectorized access (float4)
// ============================================================================
__global__ void vectorized_access(float4* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float4 v = data[idx];
        v.x *= 2.0f;
        v.y *= 2.0f;
        v.z *= 2.0f;
        v.w *= 2.0f;
        data[idx] = v;
    }
}

int main() {
    printf("Memory Access Patterns for Profiling\n");
    printf("====================================\n\n");
    
    const int N = 1 << 24;  // 16M elements
    size_t size = N * sizeof(float);
    
    float* d_data;
    int* d_indices;
    CUDA_CHECK(cudaMalloc(&d_data, size));
    CUDA_CHECK(cudaMalloc(&d_indices, N * sizeof(int)));
    
    // Initialize
    CUDA_CHECK(cudaMemset(d_data, 1, size));
    
    // Create random indices
    int* h_indices = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) {
        h_indices[i] = (i * 7919) % N;  // Pseudo-random
    }
    CUDA_CHECK(cudaMemcpy(d_indices, h_indices, N * sizeof(int), cudaMemcpyHostToDevice));
    
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    
    // Pattern 1: Coalesced
    printf("1. Coalesced access (optimal)\n");
    coalesced_access<<<numBlocks, blockSize>>>(d_data, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Pattern 2: Strided (stride=32)
    printf("2. Strided access (stride=32)\n");
    strided_access<<<numBlocks, blockSize>>>(d_data, N, 32);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Pattern 3: Random
    printf("3. Random access (poor locality)\n");
    random_access<<<numBlocks, blockSize>>>(d_data, d_indices, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Pattern 4: Misaligned (offset=1)
    printf("4. Misaligned access (offset=1)\n");
    misaligned_access<<<numBlocks, blockSize>>>(d_data, N, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Pattern 5: Vectorized
    printf("5. Vectorized access (float4)\n");
    vectorized_access<<<(N/4 + blockSize - 1) / blockSize, blockSize>>>(
        (float4*)d_data, N / 4);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_indices));
    free(h_indices);
    
    printf("\nProfile with:\n");
    printf("  ncu --set memory -o mem_report ./build/memory_patterns\n");
    printf("\nKey metrics to compare:\n");
    printf("  - dram__throughput.avg.pct_of_peak_sustained_elapsed\n");
    printf("  - l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld\n");
    
    return 0;
}
