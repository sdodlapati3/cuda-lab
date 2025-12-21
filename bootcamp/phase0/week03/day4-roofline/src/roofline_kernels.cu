/**
 * Day 4: Roofline Kernels for ncu Analysis
 * 
 * Different AI kernels for roofline profiling.
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

// AI ≈ 0.08 (very memory-bound)
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// AI ≈ 0.17 (memory-bound)
__global__ void saxpy(float a, float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = a * x[idx] + y[idx];
    }
}

// AI ≈ 2.5 (approaching ridge)
__global__ void poly_kernel(float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = x[idx];
        // 20 FLOPs
        v = v*v + v*2.0f + 1.0f;
        v = v*v + v*2.0f + 1.0f;
        v = v*v + v*2.0f + 1.0f;
        v = v*v + v*2.0f + 1.0f;
        y[idx] = v;
    }
}

// AI ≈ 25 (compute-bound)
__global__ void compute_bound(float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = x[idx];
        // ~200 FLOPs
        for (int i = 0; i < 25; i++) {
            v = v*v - v*0.5f + 0.25f;
            v = sqrtf(fabsf(v) + 0.001f);
        }
        y[idx] = v;
    }
}

int main() {
    printf("Roofline Kernels for ncu Profiling\n");
    printf("===================================\n\n");
    
    const int N = 1 << 24;
    size_t size = N * sizeof(float);
    
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, size));
    CUDA_CHECK(cudaMalloc(&d_b, size));
    CUDA_CHECK(cudaMalloc(&d_c, size));
    
    CUDA_CHECK(cudaMemset(d_a, 1, size));
    CUDA_CHECK(cudaMemset(d_b, 1, size));
    
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    
    printf("1. Vector Add (AI ≈ 0.08)\n");
    vector_add<<<numBlocks, blockSize>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("2. SAXPY (AI ≈ 0.17)\n");
    saxpy<<<numBlocks, blockSize>>>(2.0f, d_a, d_c, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("3. Polynomial (AI ≈ 2.5)\n");
    poly_kernel<<<numBlocks, blockSize>>>(d_a, d_c, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("4. Compute Bound (AI ≈ 25)\n");
    compute_bound<<<numBlocks, blockSize>>>(d_a, d_c, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    
    printf("\nProfile with ncu roofline:\n");
    printf("  ncu --set roofline -o roofline_report ./build/roofline_kernels\n");
    printf("  ncu-ui roofline_report.ncu-rep\n");
    
    return 0;
}
