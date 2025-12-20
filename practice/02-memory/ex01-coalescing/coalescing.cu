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

// Structure for AoS (Array of Structures) example
struct Particle {
    float x, y, z;
    float vx, vy, vz;
};

// =============================================================================
// TODO 1: Implement coalesced memory copy
// Each thread should copy one element where consecutive threads access
// consecutive memory locations
// =============================================================================
__global__ void copy_coalesced(float* dst, const float* src, int n) {
    // TODO: Calculate global thread index
    // TODO: Check bounds and copy element
}

// =============================================================================
// TODO 2: Implement strided memory copy (intentionally inefficient)
// Each thread accesses memory with a large stride between consecutive threads
// =============================================================================
__global__ void copy_strided(float* dst, const float* src, int n, int stride) {
    // TODO: Calculate strided index
    // TODO: Check bounds and copy element
}

// =============================================================================
// TODO 3: Convert Array of Structures (AoS) to Structure of Arrays (SoA)
// Input: Array of Particle structures
// Output: Separate arrays for x, y, z, vx, vy, vz
// =============================================================================
__global__ void aos_to_soa(
    float* x, float* y, float* z,
    float* vx, float* vy, float* vz,
    const Particle* particles, int n) {
    // TODO: Each thread converts one particle
    // TODO: Read from AoS, write to separate arrays
}

// =============================================================================
// Timing helper
// =============================================================================
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
    int stride = 32;  // Each thread is 32 elements apart
    int blockSize = 256;
    int numBlocks = (n / stride + blockSize - 1) / blockSize;
    copy_strided<<<numBlocks, blockSize, 0, stream>>>(dst, src, n, stride);
}

int main() {
    const int N = 64 * 1024 * 1024;  // 64M elements = 256 MB
    const int iterations = 100;
    
    printf("Memory Coalescing Exercise\n");
    printf("==========================\n");
    printf("Array size: %d elements (%.1f MB)\n\n", N, N * sizeof(float) / 1e6);
    
    // Allocate memory
    float *h_src, *h_dst;
    float *d_src, *d_dst;
    
    h_src = (float*)malloc(N * sizeof(float));
    h_dst = (float*)malloc(N * sizeof(float));
    CHECK_CUDA(cudaMalloc(&d_src, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dst, N * sizeof(float)));
    
    // Initialize
    for (int i = 0; i < N; i++) {
        h_src[i] = (float)i;
    }
    CHECK_CUDA(cudaMemcpy(d_src, h_src, N * sizeof(float), cudaMemcpyHostToDevice));
    
    // Benchmark coalesced access
    float time_coalesced = benchmark_kernel(launch_coalesced, d_dst, d_src, N, iterations);
    float bandwidth_coalesced = 2.0f * N * sizeof(float) / (time_coalesced * 1e6);  // GB/s
    
    printf("Coalesced Copy:\n");
    printf("  Time: %.3f ms\n", time_coalesced);
    printf("  Bandwidth: %.1f GB/s\n\n", bandwidth_coalesced);
    
    // Benchmark strided access
    float time_strided = benchmark_kernel(launch_strided, d_dst, d_src, N, iterations);
    float bandwidth_strided = 2.0f * N * sizeof(float) / (time_strided * 1e6);
    
    printf("Strided Copy (stride=32):\n");
    printf("  Time: %.3f ms\n", time_strided);
    printf("  Bandwidth: %.1f GB/s\n\n", bandwidth_strided);
    
    printf("Speedup (coalesced vs strided): %.2fx\n", time_strided / time_coalesced);
    
    // Cleanup
    free(h_src);
    free(h_dst);
    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));
    
    printf("\n✓ Exercise complete!\n");
    return 0;
}
