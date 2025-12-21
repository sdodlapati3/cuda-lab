/**
 * Day 5: Optimization Examples
 * 
 * Before/after optimization for different bottlenecks.
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
// Memory Optimization: Before (scalar) vs After (vectorized)
// ============================================================================
__global__ void copy_scalar(const float* src, float* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

__global__ void copy_vector4(const float4* src, float4* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n / 4) {
        dst[idx] = src[idx];
    }
}

// ============================================================================
// Compute Optimization: Standard math vs fast math
// ============================================================================
__global__ void standard_math(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = in[idx];
        v = expf(v) * sinf(v) + logf(fabsf(v) + 0.001f);
        out[idx] = v;
    }
}

__global__ void fast_math(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = in[idx];
        v = __expf(v) * __sinf(v) + __logf(fabsf(v) + 0.001f);
        out[idx] = v;
    }
}

// ============================================================================
// Shared Memory Optimization: Global vs Cached
// ============================================================================
__global__ void stencil_global(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 0 && idx < n - 1) {
        // 3-point stencil from global memory
        out[idx] = 0.25f * in[idx-1] + 0.5f * in[idx] + 0.25f * in[idx+1];
    }
}

__global__ void stencil_shared(const float* in, float* out, int n) {
    __shared__ float smem[258];  // 256 + 2 halo
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Load to shared memory with halos
    if (idx < n) {
        smem[tid + 1] = in[idx];
    }
    
    // Load halos
    if (tid == 0 && idx > 0) {
        smem[0] = in[idx - 1];
    }
    if (tid == blockDim.x - 1 && idx < n - 1) {
        smem[tid + 2] = in[idx + 1];
    }
    
    __syncthreads();
    
    if (idx > 0 && idx < n - 1) {
        out[idx] = 0.25f * smem[tid] + 0.5f * smem[tid + 1] + 0.25f * smem[tid + 2];
    }
}

void benchmark(const char* name, void (*kernel)(const float*, float*, int),
               const float* d_in, float* d_out, int n) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    // Warmup
    kernel<<<numBlocks, blockSize>>>(d_in, d_out, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    int iters = 100;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) {
        kernel<<<numBlocks, blockSize>>>(d_in, d_out, n);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    
    printf("  %s: %.3f ms\n", name, ms / iters);
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main() {
    printf("Optimization Examples\n");
    printf("=====================\n\n");
    
    const int N = 1 << 24;
    size_t size = N * sizeof(float);
    
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, size));
    CUDA_CHECK(cudaMalloc(&d_out, size));
    CUDA_CHECK(cudaMemset(d_in, 1, size));
    
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    
    // Memory optimization
    printf("Memory Optimization (Copy):\n");
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Scalar
    copy_scalar<<<numBlocks, blockSize>>>(d_in, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < 100; i++) {
        copy_scalar<<<numBlocks, blockSize>>>(d_in, d_out, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_scalar;
    CUDA_CHECK(cudaEventElapsedTime(&ms_scalar, start, stop));
    printf("  Scalar: %.3f ms\n", ms_scalar / 100);
    
    // Vector4
    int numBlocks4 = (N/4 + blockSize - 1) / blockSize;
    copy_vector4<<<numBlocks4, blockSize>>>((float4*)d_in, (float4*)d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < 100; i++) {
        copy_vector4<<<numBlocks4, blockSize>>>((float4*)d_in, (float4*)d_out, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_vector;
    CUDA_CHECK(cudaEventElapsedTime(&ms_vector, start, stop));
    printf("  Vector4: %.3f ms (%.1fx faster)\n", 
           ms_vector / 100, ms_scalar / ms_vector);
    
    // Compute optimization
    printf("\nCompute Optimization (Math):\n");
    benchmark("Standard math", standard_math, d_in, d_out, N);
    benchmark("Fast math", fast_math, d_in, d_out, N);
    
    // Shared memory optimization  
    printf("\nShared Memory Optimization (Stencil):\n");
    benchmark("Global memory", stencil_global, d_in, d_out, N);
    benchmark("Shared memory", stencil_shared, d_in, d_out, N);
    
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return 0;
}
