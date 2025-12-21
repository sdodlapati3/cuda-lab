#pragma once
/**
 * cuda_utils.cuh - CUDA utility macros and helpers
 */

#include <cuda_runtime.h>
#include <cstdio>

namespace myproject {

// Error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        return; \
    } \
} while(0)

#define CUDA_CHECK_RETURN(call, retval) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        return retval; \
    } \
} while(0)

// Device memory RAII wrapper
template<typename T>
class DeviceBuffer {
public:
    DeviceBuffer() : ptr_(nullptr), size_(0) {}
    
    explicit DeviceBuffer(size_t count) : size_(count) {
        cudaMalloc(&ptr_, count * sizeof(T));
    }
    
    ~DeviceBuffer() {
        if (ptr_) cudaFree(ptr_);
    }
    
    // No copy
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    
    // Move
    DeviceBuffer(DeviceBuffer&& other) noexcept 
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFree(ptr_);
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    
    void allocate(size_t count) {
        if (ptr_) cudaFree(ptr_);
        size_ = count;
        cudaMalloc(&ptr_, count * sizeof(T));
    }
    
    void copyFrom(const T* host_data) {
        cudaMemcpy(ptr_, host_data, size_ * sizeof(T), cudaMemcpyHostToDevice);
    }
    
    void copyTo(T* host_data) const {
        cudaMemcpy(host_data, ptr_, size_ * sizeof(T), cudaMemcpyDeviceToHost);
    }
    
    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    size_t size() const { return size_; }
    
private:
    T* ptr_;
    size_t size_;
};

// Kernel launch helpers
inline int divUp(int a, int b) {
    return (a + b - 1) / b;
}

inline dim3 getGridDim(int n, int blockSize) {
    return dim3(divUp(n, blockSize));
}

inline dim3 getGridDim2D(int width, int height, dim3 blockDim) {
    return dim3(divUp(width, blockDim.x), divUp(height, blockDim.y));
}

}  // namespace myproject
