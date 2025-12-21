/**
 * Day 3: Warp Efficiency Analysis
 * 
 * Demonstrate divergence and efficiency.
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
// No divergence - all threads take same path
// ============================================================================
__global__ void no_divergence(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // All threads do the same thing
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}

// ============================================================================
// Warp-level divergence (bad)
// ============================================================================
__global__ void warp_divergence(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Divergence within warp!
        if (threadIdx.x % 2 == 0) {
            data[idx] = data[idx] * 2.0f;
        } else {
            data[idx] = data[idx] + 1.0f;
        }
    }
}

// ============================================================================
// Severe divergence (very bad)
// ============================================================================
__global__ void severe_divergence(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Each thread takes different path
        switch (threadIdx.x % 8) {
            case 0: data[idx] *= 1.1f; break;
            case 1: data[idx] *= 1.2f; break;
            case 2: data[idx] *= 1.3f; break;
            case 3: data[idx] *= 1.4f; break;
            case 4: data[idx] += 0.1f; break;
            case 5: data[idx] += 0.2f; break;
            case 6: data[idx] += 0.3f; break;
            case 7: data[idx] += 0.4f; break;
        }
    }
}

// ============================================================================
// Block-level divergence (OK - no warp divergence)
// ============================================================================
__global__ void block_divergence(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Different blocks take different paths
        // But all threads in a warp take same path
        if (blockIdx.x % 2 == 0) {
            data[idx] = data[idx] * 2.0f;
        } else {
            data[idx] = data[idx] + 1.0f;
        }
    }
}

// ============================================================================
// Predicated (optimized divergence)
// ============================================================================
__global__ void predicated_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Use branchless computation
        float mult = (threadIdx.x % 2 == 0) ? 2.0f : 1.0f;
        float add = (threadIdx.x % 2 == 0) ? 0.0f : 1.0f;
        data[idx] = data[idx] * mult + add;
    }
}

int main() {
    printf("Warp Efficiency Analysis\n");
    printf("========================\n\n");
    
    const int N = 1 << 22;
    float* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_data, 1, N * sizeof(float)));
    
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    
    printf("1. No divergence (100%% efficient)\n");
    no_divergence<<<numBlocks, blockSize>>>(d_data, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("2. Warp divergence (50%% efficient per branch)\n");
    warp_divergence<<<numBlocks, blockSize>>>(d_data, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("3. Severe divergence (12.5%% efficient per branch)\n");
    severe_divergence<<<numBlocks, blockSize>>>(d_data, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("4. Block divergence (100%% efficient - no warp divergence)\n");
    block_divergence<<<numBlocks, blockSize>>>(d_data, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("5. Predicated (branchless)\n");
    predicated_kernel<<<numBlocks, blockSize>>>(d_data, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaFree(d_data));
    
    printf("\nProfile with:\n");
    printf("  ncu --metrics smsp__thread_inst_executed_per_inst_executed.ratio ./build/warp_efficiency\n");
    
    return 0;
}
