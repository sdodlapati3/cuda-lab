/**
 * Day 5: Bottleneck Demonstration
 * 
 * Kernels with different bottlenecks for analysis.
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
// Memory Bound: High memory traffic, low compute
// ============================================================================
__global__ void memory_bound(const float* a, const float* b, 
                             const float* c, const float* d,
                             float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 4 reads, 1 write, 1 FLOP
        out[idx] = a[idx] + b[idx] + c[idx] + d[idx];
    }
}

// ============================================================================
// Compute Bound: High compute, low memory traffic
// ============================================================================
__global__ void compute_bound(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = in[idx];
        // Many FLOPs, little memory
        for (int i = 0; i < 100; i++) {
            v = sinf(v) * cosf(v) + v * 0.99f;
        }
        out[idx] = v;
    }
}

// ============================================================================
// Latency Bound: Low occupancy, stalls
// ============================================================================
__global__ void latency_bound(const float* in, float* out, int n) {
    // Force low occupancy with lots of registers/shared mem
    __shared__ float smem[8192];  // 32KB - limits blocks
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    if (idx < n) {
        smem[tid] = in[idx];
        __syncthreads();
        
        // Simple operation but limited by occupancy
        out[idx] = smem[tid] * 2.0f;
    }
}

// ============================================================================
// Balanced: Both memory and compute active
// ============================================================================
__global__ void balanced(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = in[idx];
        // Moderate compute
        for (int i = 0; i < 10; i++) {
            v = v * v + v * 0.1f + 0.01f;
        }
        out[idx] = v;
    }
}

int main() {
    printf("Bottleneck Analysis Demo\n");
    printf("========================\n\n");
    
    const int N = 1 << 22;  // 4M elements
    size_t size = N * sizeof(float);
    
    float *d_a, *d_b, *d_c, *d_d, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, size));
    CUDA_CHECK(cudaMalloc(&d_b, size));
    CUDA_CHECK(cudaMalloc(&d_c, size));
    CUDA_CHECK(cudaMalloc(&d_d, size));
    CUDA_CHECK(cudaMalloc(&d_out, size));
    
    CUDA_CHECK(cudaMemset(d_a, 1, size));
    CUDA_CHECK(cudaMemset(d_b, 1, size));
    CUDA_CHECK(cudaMemset(d_c, 1, size));
    CUDA_CHECK(cudaMemset(d_d, 1, size));
    
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    
    printf("1. Memory Bound (expect high DRAM%%, low SM%%)\n");
    memory_bound<<<numBlocks, blockSize>>>(d_a, d_b, d_c, d_d, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("2. Compute Bound (expect low DRAM%%, high SM%%)\n");
    compute_bound<<<numBlocks, blockSize>>>(d_a, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("3. Latency Bound (expect low both)\n");
    latency_bound<<<numBlocks, blockSize>>>(d_a, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("4. Balanced (moderate both)\n");
    balanced<<<numBlocks, blockSize>>>(d_a, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_d));
    CUDA_CHECK(cudaFree(d_out));
    
    printf("\nProfile and identify bottlenecks:\n");
    printf("  ncu --metrics \\\n");
    printf("    dram__throughput.avg.pct_of_peak_sustained_elapsed,\\\n");
    printf("    sm__throughput.avg.pct_of_peak_sustained_elapsed \\\n");
    printf("    ./build/bottleneck_demo\n");
    
    return 0;
}
