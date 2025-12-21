/**
 * Day 1: Vector Operations for Profiling
 * 
 * Different kernels with varying characteristics for ncu analysis.
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
// Kernel 1: Simple vector add (memory-bound)
// ============================================================================
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];  // 2 reads, 1 write, 1 FLOP
    }
}

// ============================================================================
// Kernel 2: Vector scale (even more memory-bound)
// ============================================================================
__global__ void vector_scale(const float* a, float* b, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        b[idx] = a[idx] * scale;  // 1 read, 1 write, 1 FLOP
    }
}

// ============================================================================
// Kernel 3: Vector with more compute (better arithmetic intensity)
// ============================================================================
__global__ void vector_compute(const float* a, float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = a[idx];
        // Many FLOPs per memory access
        for (int i = 0; i < 100; i++) {
            val = sinf(val) * cosf(val) + 0.1f;
        }
        b[idx] = val;  // 1 read, 1 write, ~300 FLOPs
    }
}

// ============================================================================
// Kernel 4: Fused multiply-add (FMA utilization)
// ============================================================================
__global__ void vector_fma(const float* a, const float* b, const float* c, 
                           float* d, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d[idx] = fmaf(a[idx], b[idx], c[idx]);  // a*b + c
    }
}

void run_kernel(const char* name, void (*launcher)(float*, float*, float*, int)) {
    const int N = 1 << 24;  // 16M elements
    size_t size = N * sizeof(float);
    
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, size));
    CUDA_CHECK(cudaMalloc(&d_b, size));
    CUDA_CHECK(cudaMalloc(&d_c, size));
    
    // Initialize
    CUDA_CHECK(cudaMemset(d_a, 1, size));
    CUDA_CHECK(cudaMemset(d_b, 1, size));
    
    // Warmup
    launcher(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Profile run
    printf("Running %s...\n", name);
    launcher(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
}

int main() {
    printf("Vector Operations for Nsight Compute\n");
    printf("=====================================\n\n");
    
    const int N = 1 << 24;
    size_t size = N * sizeof(float);
    
    float *d_a, *d_b, *d_c, *d_d;
    CUDA_CHECK(cudaMalloc(&d_a, size));
    CUDA_CHECK(cudaMalloc(&d_b, size));
    CUDA_CHECK(cudaMalloc(&d_c, size));
    CUDA_CHECK(cudaMalloc(&d_d, size));
    
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    
    printf("Problem size: %d elements (%.1f MB)\n", N, size / 1e6);
    printf("Grid: %d blocks x %d threads\n\n", numBlocks, blockSize);
    
    // Run each kernel
    printf("1. vector_add (memory-bound, low AI)\n");
    vector_add<<<numBlocks, blockSize>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("2. vector_scale (memory-bound, lowest AI)\n");
    vector_scale<<<numBlocks, blockSize>>>(d_a, d_c, 2.0f, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("3. vector_compute (compute-heavy, high AI)\n");
    vector_compute<<<numBlocks, blockSize>>>(d_a, d_c, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("4. vector_fma (FMA instruction)\n");
    vector_fma<<<numBlocks, blockSize>>>(d_a, d_b, d_c, d_d, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_d));
    
    printf("\nDone! Profile with:\n");
    printf("  ncu --set full -o report ./build/vector_ops\n");
    printf("  ncu-ui report.ncu-rep\n");
    
    return 0;
}
