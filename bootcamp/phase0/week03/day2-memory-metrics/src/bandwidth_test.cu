/**
 * Day 2: Bandwidth Measurement
 * 
 * Measure actual vs theoretical bandwidth.
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

// Simple copy kernel for bandwidth test
__global__ void copy_kernel(const float* src, float* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

// Read-only kernel
__global__ void read_kernel(const float* src, float* partial_sum, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Read only, no write (except one value)
        float val = src[idx];
        if (idx == 0) *partial_sum = val;
    }
}

// Write-only kernel
__global__ void write_kernel(float* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = 1.0f;
    }
}

float benchmark_kernel(void (*kernel)(const float*, float*, int),
                       const float* src, float* dst, int n,
                       int iterations) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Warmup
    kernel<<<numBlocks, blockSize>>>(src, dst, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        kernel<<<numBlocks, blockSize>>>(src, dst, n);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return ms / iterations;
}

int main() {
    printf("Bandwidth Measurement\n");
    printf("====================\n\n");
    
    // Get device info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s\n", prop.name);
    
    // Calculate theoretical bandwidth
    double mem_clock_khz = prop.memoryClockRate;
    double mem_bus_width_bits = prop.memoryBusWidth;
    double theoretical_bw = 2.0 * mem_clock_khz * 1e3 * mem_bus_width_bits / 8 / 1e9;
    printf("Theoretical Bandwidth: %.1f GB/s\n\n", theoretical_bw);
    
    const int N = 1 << 26;  // 64M elements
    size_t size = N * sizeof(float);
    int iterations = 100;
    
    float *d_src, *d_dst;
    CUDA_CHECK(cudaMalloc(&d_src, size));
    CUDA_CHECK(cudaMalloc(&d_dst, size));
    CUDA_CHECK(cudaMemset(d_src, 1, size));
    
    printf("Data size: %.1f MB\n", size / 1e6);
    printf("Iterations: %d\n\n", iterations);
    
    // Copy kernel bandwidth
    float copy_ms = benchmark_kernel(copy_kernel, d_src, d_dst, N, iterations);
    double copy_bw = 2.0 * size / copy_ms / 1e6;  // Read + Write
    printf("Copy Kernel:\n");
    printf("  Time: %.3f ms\n", copy_ms);
    printf("  Bandwidth: %.1f GB/s (%.1f%% of peak)\n\n", 
           copy_bw, 100.0 * copy_bw / theoretical_bw);
    
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
    
    printf("To verify with ncu:\n");
    printf("  ncu --metrics dram__bytes.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed ./build/bandwidth_test\n");
    
    return 0;
}
