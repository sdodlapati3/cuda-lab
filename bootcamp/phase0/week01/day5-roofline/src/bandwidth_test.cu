/**
 * Bandwidth Test - Measure peak memory bandwidth
 * 
 * Strategy: Large memcpy-like kernel that moves data without computation.
 * This gives us the "memory roof" of the roofline.
 */

#include <cuda_runtime.h>
#include <cstdio>

// Simple copy kernel - should achieve near-peak bandwidth
__global__ void copy_kernel(const float4* __restrict__ src, 
                            float4* __restrict__ dst, 
                            size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

// Read-only kernel (no writes, tests read bandwidth)
__global__ void read_kernel(const float4* __restrict__ src, 
                            float* __restrict__ result,
                            size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    if (idx < n) {
        float4 val = src[idx];
        sum = val.x + val.y + val.z + val.w;
    }
    // Prevent optimization - atomically add (only 1 thread does this)
    if (idx == 0) atomicAdd(result, sum);
}

struct BandwidthResults {
    float copy_bandwidth_gb_s;
    float read_bandwidth_gb_s;
    float theoretical_peak_gb_s;
};

BandwidthResults measure_bandwidth() {
    const size_t SIZE = 256 * 1024 * 1024;  // 256 MB
    const size_t N = SIZE / sizeof(float4);
    const int THREADS = 256;
    const int BLOCKS = (N + THREADS - 1) / THREADS;
    const int WARMUP = 10;
    const int ITERATIONS = 100;
    
    printf("\n=== Bandwidth Measurement ===\n");
    printf("Data size: %zu MB\n", SIZE / (1024 * 1024));
    
    // Allocate
    float4 *d_src, *d_dst;
    float *d_result;
    cudaMalloc(&d_src, SIZE);
    cudaMalloc(&d_dst, SIZE);
    cudaMalloc(&d_result, sizeof(float));
    cudaMemset(d_src, 1, SIZE);  // Initialize with some data
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    for (int i = 0; i < WARMUP; i++) {
        copy_kernel<<<BLOCKS, THREADS>>>(d_src, d_dst, N);
    }
    cudaDeviceSynchronize();
    
    // Measure copy (read + write)
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        copy_kernel<<<BLOCKS, THREADS>>>(d_src, d_dst, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float copy_ms;
    cudaEventElapsedTime(&copy_ms, start, stop);
    float copy_bw = (2.0f * SIZE * ITERATIONS / 1e9) / (copy_ms / 1e3);
    
    // Measure read-only
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        read_kernel<<<BLOCKS, THREADS>>>(d_src, d_result, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float read_ms;
    cudaEventElapsedTime(&read_ms, start, stop);
    float read_bw = (SIZE * ITERATIONS / 1e9) / (read_ms / 1e3);
    
    // Theoretical peak
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    float theoretical = 2.0f * props.memoryClockRate * (props.memoryBusWidth / 8) / 1e6;
    
    printf("Copy bandwidth:       %.1f GB/s\n", copy_bw);
    printf("Read bandwidth:       %.1f GB/s\n", read_bw);
    printf("Theoretical peak:     %.1f GB/s\n", theoretical);
    printf("Achieved:             %.1f%% of peak\n", 100.0f * copy_bw / theoretical);
    
    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_result);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return {copy_bw, read_bw, theoretical};
}
