#pragma once
/**
 * timer.h - High-precision timing utilities
 */

#include <cuda_runtime.h>
#include <chrono>

namespace myproject {

// CUDA event-based timer
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
    
    void start() {
        cudaEventRecord(start_);
    }
    
    void stop() {
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
    }
    
    float elapsed_ms() const {
        float ms;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }
    
private:
    cudaEvent_t start_, stop_;
};

// CPU timer
class CpuTimer {
public:
    void start() {
        start_ = std::chrono::high_resolution_clock::now();
    }
    
    void stop() {
        stop_ = std::chrono::high_resolution_clock::now();
    }
    
    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(stop_ - start_).count();
    }
    
private:
    std::chrono::high_resolution_clock::time_point start_, stop_;
};

}  // namespace myproject
