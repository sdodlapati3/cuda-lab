#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// ============================================================================
// Error Checking
// ============================================================================

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())

// Check for errors after kernel launch
#define CUDA_KERNEL_CHECK() do { \
    CUDA_CHECK(cudaGetLastError()); \
    CUDA_CHECK(cudaDeviceSynchronize()); \
} while(0)

// ============================================================================
// Device Info
// ============================================================================

struct GpuInfo {
    char name[256];
    int sm_count;
    int compute_major;
    int compute_minor;
    size_t global_mem_bytes;
    size_t shared_mem_per_block;
    int max_threads_per_block;
    int max_threads_per_sm;
    int warp_size;
    float peak_bandwidth_gb_s;
    
    void print() const {
        printf("\n=== GPU Information ===\n");
        printf("Device:           %s\n", name);
        printf("Compute:          %d.%d\n", compute_major, compute_minor);
        printf("SM Count:         %d\n", sm_count);
        printf("Global Memory:    %.1f GB\n", global_mem_bytes / 1e9);
        printf("Shared/Block:     %zu KB\n", shared_mem_per_block / 1024);
        printf("Max Threads/Block: %d\n", max_threads_per_block);
        printf("Max Threads/SM:   %d\n", max_threads_per_sm);
        printf("Warp Size:        %d\n", warp_size);
        printf("Peak Bandwidth:   %.0f GB/s\n", peak_bandwidth_gb_s);
        printf("========================\n\n");
    }
};

inline GpuInfo get_gpu_info(int device = 0) {
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    
    GpuInfo info;
    strncpy(info.name, props.name, sizeof(info.name));
    info.sm_count = props.multiProcessorCount;
    info.compute_major = props.major;
    info.compute_minor = props.minor;
    info.global_mem_bytes = props.totalGlobalMem;
    info.shared_mem_per_block = props.sharedMemPerBlock;
    info.max_threads_per_block = props.maxThreadsPerBlock;
    info.max_threads_per_sm = props.maxThreadsPerMultiProcessor;
    info.warp_size = props.warpSize;
    info.peak_bandwidth_gb_s = 2.0f * props.memoryClockRate * 
                                (props.memoryBusWidth / 8) / 1e6;
    
    return info;
}

// ============================================================================
// Memory Helpers
// ============================================================================

template <typename T>
T* cuda_malloc(size_t count) {
    T* ptr;
    CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
    return ptr;
}

template <typename T>
void cuda_memcpy_h2d(T* dst, const T* src, size_t count) {
    CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
void cuda_memcpy_d2h(T* dst, const T* src, size_t count) {
    CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyDeviceToHost));
}

// RAII wrapper for device memory
template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer(size_t count) : count_(count) {
        CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
    }
    
    ~DeviceBuffer() {
        if (ptr_) cudaFree(ptr_);
    }
    
    // Non-copyable
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    
    // Movable
    DeviceBuffer(DeviceBuffer&& other) noexcept 
        : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }
    
    T* data() { return ptr_; }
    const T* data() const { return ptr_; }
    size_t size() const { return count_; }
    size_t bytes() const { return count_ * sizeof(T); }
    
    void copy_from_host(const T* src) {
        cuda_memcpy_h2d(ptr_, src, count_);
    }
    
    void copy_to_host(T* dst) const {
        cuda_memcpy_d2h(dst, ptr_, count_);
    }

private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
};

// ============================================================================
// Timing (from cuda_timer.cuh)
// ============================================================================

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
    
    void start() { cudaEventRecord(start_); }
    void stop() { cudaEventRecord(stop_); }
    
    float elapsed_ms() {
        cudaEventSynchronize(stop_);
        float ms;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }

private:
    cudaEvent_t start_, stop_;
};
