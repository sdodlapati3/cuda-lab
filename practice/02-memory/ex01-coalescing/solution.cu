#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

struct Particle {
    float x, y, z;
    float vx, vy, vz;
};

// SOLUTION: Coalesced memory copy
// Consecutive threads access consecutive memory locations
__global__ void copy_coalesced(float* dst, const float* src, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

// SOLUTION: Strided memory copy (intentionally inefficient)
// Each thread accesses memory with a large stride
__global__ void copy_strided(float* dst, const float* src, int n, int stride) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int idx = tid * stride;  // Large stride between consecutive threads
    
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

// SOLUTION: Convert AoS to SoA
// This transforms scattered reads into coalesced writes
__global__ void aos_to_soa(
    float* x, float* y, float* z,
    float* vx, float* vy, float* vz,
    const Particle* particles, int n) {
    
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        // Read from AoS (scattered access within struct)
        Particle p = particles[idx];
        
        // Write to SoA (coalesced access to separate arrays)
        x[idx] = p.x;
        y[idx] = p.y;
        z[idx] = p.z;
        vx[idx] = p.vx;
        vy[idx] = p.vy;
        vz[idx] = p.vz;
    }
}

// Timing helper
float benchmark_kernel(void (*launch_func)(float*, const float*, int, cudaStream_t),
                       float* dst, const float* src, int n, int iterations) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    // Warmup
    launch_func(dst, src, n, 0);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        launch_func(dst, src, n, 0);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    return ms / iterations;
}

void launch_coalesced(float* dst, const float* src, int n, cudaStream_t stream) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    copy_coalesced<<<numBlocks, blockSize, 0, stream>>>(dst, src, n);
}

void launch_strided(float* dst, const float* src, int n, cudaStream_t stream) {
    int stride = 32;
    int blockSize = 256;
    int numBlocks = (n / stride + blockSize - 1) / blockSize;
    copy_strided<<<numBlocks, blockSize, 0, stream>>>(dst, src, n, stride);
}

int main() {
    const int N = 64 * 1024 * 1024;
    const int iterations = 100;
    
    printf("Memory Coalescing Exercise - SOLUTION\n");
    printf("======================================\n");
    printf("Array size: %d elements (%.1f MB)\n\n", N, N * sizeof(float) / 1e6);
    
    float *h_src, *h_dst;
    float *d_src, *d_dst;
    
    h_src = (float*)malloc(N * sizeof(float));
    h_dst = (float*)malloc(N * sizeof(float));
    CHECK_CUDA(cudaMalloc(&d_src, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dst, N * sizeof(float)));
    
    for (int i = 0; i < N; i++) {
        h_src[i] = (float)i;
    }
    CHECK_CUDA(cudaMemcpy(d_src, h_src, N * sizeof(float), cudaMemcpyHostToDevice));
    
    // Benchmark coalesced
    float time_coalesced = benchmark_kernel(launch_coalesced, d_dst, d_src, N, iterations);
    float bandwidth_coalesced = 2.0f * N * sizeof(float) / (time_coalesced * 1e6);
    
    printf("Coalesced Copy:\n");
    printf("  Time: %.3f ms\n", time_coalesced);
    printf("  Bandwidth: %.1f GB/s\n\n", bandwidth_coalesced);
    
    // Verify
    CHECK_CUDA(cudaMemcpy(h_dst, d_dst, N * sizeof(float), cudaMemcpyDeviceToHost));
    bool correct = true;
    for (int i = 0; i < 1000; i++) {
        if (h_dst[i] != h_src[i]) {
            correct = false;
            break;
        }
    }
    printf("  Verification: %s\n\n", correct ? "PASSED" : "FAILED");
    
    // Benchmark strided
    float time_strided = benchmark_kernel(launch_strided, d_dst, d_src, N, iterations);
    float bandwidth_strided = 2.0f * N * sizeof(float) / (time_strided * 1e6);
    
    printf("Strided Copy (stride=32):\n");
    printf("  Time: %.3f ms\n", time_strided);
    printf("  Bandwidth: %.1f GB/s\n\n", bandwidth_strided);
    
    printf("Performance Summary:\n");
    printf("  Speedup (coalesced vs strided): %.2fx\n", time_strided / time_coalesced);
    printf("  Bandwidth ratio: %.1f%%\n", (bandwidth_strided / bandwidth_coalesced) * 100);
    
    free(h_src);
    free(h_dst);
    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));
    
    printf("\n✓ Solution complete!\n");
    return 0;
}
