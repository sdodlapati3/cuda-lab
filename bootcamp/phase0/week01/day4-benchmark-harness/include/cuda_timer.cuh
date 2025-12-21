#pragma once

#include <cuda_runtime.h>
#include <cstdio>

/**
 * CudaTimer - RAII wrapper for CUDA event-based timing
 * 
 * Usage:
 *   CudaTimer timer;
 *   timer.start();
 *   my_kernel<<<...>>>(...);
 *   timer.stop();
 *   float ms = timer.elapsed_ms();
 */
class CudaTimer {
public:
    CudaTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }
    
    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    
    // Non-copyable
    CudaTimer(const CudaTimer&) = delete;
    CudaTimer& operator=(const CudaTimer&) = delete;
    
    void start() {
        cudaEventRecord(start_);
    }
    
    void stop() {
        cudaEventRecord(stop_);
    }
    
    // Returns elapsed time in milliseconds
    // Blocks until kernel completes!
    float elapsed_ms() {
        cudaEventSynchronize(stop_);
        float ms;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }
    
    // Returns elapsed time in microseconds
    float elapsed_us() {
        return elapsed_ms() * 1000.0f;
    }

private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
};

/**
 * Helper function for timing a kernel with warmup + iterations
 * 
 * Returns: average time in milliseconds
 */
template <typename Func>
float time_kernel(Func kernel_fn, int warmup = 10, int iterations = 100) {
    // Warmup
    for (int i = 0; i < warmup; i++) {
        kernel_fn();
    }
    cudaDeviceSynchronize();
    
    // Timed runs
    CudaTimer timer;
    timer.start();
    for (int i = 0; i < iterations; i++) {
        kernel_fn();
    }
    timer.stop();
    
    return timer.elapsed_ms() / iterations;
}

/**
 * Calculate and print bandwidth
 */
inline void print_bandwidth(const char* name, float ms, size_t bytes) {
    float gb_per_s = (bytes / 1e9) / (ms / 1e3);
    printf("%-20s: %8.3f ms  |  %8.2f GB/s\n", name, ms, gb_per_s);
}

/**
 * Calculate and print GFLOPS
 */
inline void print_gflops(const char* name, float ms, size_t flops) {
    float gflops = (flops / 1e9) / (ms / 1e3);
    printf("%-20s: %8.3f ms  |  %8.2f GFLOPS\n", name, ms, gflops);
}
