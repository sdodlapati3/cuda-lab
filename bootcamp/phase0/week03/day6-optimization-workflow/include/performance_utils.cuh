#pragma once
/**
 * performance_utils.cuh - Performance Analysis Utilities
 * 
 * Reusable utilities for benchmarking and profiling.
 */

#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// ============================================================================
// Timer class for kernel benchmarking
// ============================================================================
class CudaTimer {
public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    
    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    
    void start() {
        CUDA_CHECK(cudaEventRecord(start_));
    }
    
    float stop() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_, stop_;
};

// ============================================================================
// Device info utilities
// ============================================================================
inline void printDeviceInfo() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    printf("Device: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("SMs: %d, Max threads/SM: %d\n", 
           prop.multiProcessorCount, prop.maxThreadsPerMultiProcessor);
    printf("Global Memory: %.1f GB\n", prop.totalGlobalMem / 1e9);
    printf("Memory Clock: %.0f MHz, Bus Width: %d bits\n",
           prop.memoryClockRate / 1000.0f, prop.memoryBusWidth);
    
    // Calculate peak bandwidth
    double peak_bw = 2.0 * prop.memoryClockRate * 1e3 * prop.memoryBusWidth / 8 / 1e9;
    printf("Peak Bandwidth: ~%.0f GB/s\n", peak_bw);
}

inline double getPeakBandwidthGB() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    return 2.0 * prop.memoryClockRate * 1e3 * prop.memoryBusWidth / 8 / 1e9;
}

// ============================================================================
// Performance calculation utilities
// ============================================================================
inline double calculateBandwidthGB(size_t bytes, double time_ms) {
    return bytes / time_ms / 1e6;  // GB/s
}

inline double calculateGFLOPS(double flops, double time_ms) {
    return flops / time_ms / 1e6;  // GFLOP/s
}

inline double calculateArithmeticIntensity(double flops, size_t bytes) {
    return flops / bytes;  // FLOP/Byte
}

// ============================================================================
// Occupancy helper
// ============================================================================
template<typename KernelFunc>
void printOccupancy(const char* name, KernelFunc kernel, int blockSize, 
                    size_t dynamicSmem = 0) {
    int maxActiveBlocks;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocks, kernel, blockSize, dynamicSmem));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    int maxWarpsPerSM = prop.maxThreadsPerMultiProcessor / 32;
    int activeWarps = maxActiveBlocks * (blockSize / 32);
    float occupancy = 100.0f * activeWarps / maxWarpsPerSM;
    
    printf("%s: Block=%d, Blocks/SM=%d, Occupancy=%.1f%%\n",
           name, blockSize, maxActiveBlocks, occupancy);
}
