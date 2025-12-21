#pragma once
/**
 * debug_utils.cuh - Reusable Debug Infrastructure
 * 
 * Include this header in your CUDA projects for consistent
 * error handling and debugging support.
 */

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// ============================================================================
// Error Checking Macros
// ============================================================================

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Kernel check - synchronizes in debug mode
#ifdef DEBUG
    #define CUDA_CHECK_KERNEL() do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "Kernel launch error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
        err = cudaDeviceSynchronize(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "Kernel execution error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)
#else
    #define CUDA_CHECK_KERNEL() do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "Kernel launch error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)
#endif

// ============================================================================
// Debug Print (only in debug mode)
// ============================================================================

#ifdef DEBUG
    #define DEBUG_PRINT(...) printf(__VA_ARGS__)
#else
    #define DEBUG_PRINT(...) ((void)0)
#endif

// ============================================================================
// Device Assertions
// ============================================================================

#ifdef DEBUG
    #define CUDA_ASSERT(cond) do { \
        if (!(cond)) { \
            printf("Assertion failed at %s:%d: %s\n", \
                   __FILE__, __LINE__, #cond); \
        } \
    } while(0)
#else
    #define CUDA_ASSERT(cond) ((void)0)
#endif

// ============================================================================
// RAII Buffer Management
// ============================================================================

class DeviceBuffer {
public:
    DeviceBuffer() : ptr_(nullptr), size_(0) {}
    
    explicit DeviceBuffer(size_t size) : ptr_(nullptr), size_(size) {
        if (size > 0) {
            CUDA_CHECK(cudaMalloc(&ptr_, size));
        }
    }
    
    ~DeviceBuffer() {
        if (ptr_) {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
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
    
    void* get() { return ptr_; }
    const void* get() const { return ptr_; }
    
    template<typename T>
    T* as() { return static_cast<T*>(ptr_); }
    
    template<typename T>
    const T* as() const { return static_cast<const T*>(ptr_); }
    
    size_t size() const { return size_; }
    
    void copyFrom(const void* src, size_t bytes) {
        CUDA_CHECK(cudaMemcpy(ptr_, src, bytes, cudaMemcpyHostToDevice));
    }
    
    void copyTo(void* dst, size_t bytes) const {
        CUDA_CHECK(cudaMemcpy(dst, ptr_, bytes, cudaMemcpyDeviceToHost));
    }

private:
    void* ptr_;
    size_t size_;
};

// Pinned host memory
class PinnedBuffer {
public:
    PinnedBuffer() : ptr_(nullptr), size_(0) {}
    
    explicit PinnedBuffer(size_t size) : ptr_(nullptr), size_(size) {
        if (size > 0) {
            CUDA_CHECK(cudaMallocHost(&ptr_, size));
        }
    }
    
    ~PinnedBuffer() {
        if (ptr_) {
            cudaFreeHost(ptr_);
            ptr_ = nullptr;
        }
    }
    
    // No copy
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    
    void* get() { return ptr_; }
    const void* get() const { return ptr_; }
    
    template<typename T>
    T* as() { return static_cast<T*>(ptr_); }
    
    size_t size() const { return size_; }

private:
    void* ptr_;
    size_t size_;
};

// ============================================================================
// Device Info Utilities
// ============================================================================

inline void printDeviceInfo() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    printf("Device: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Memory: %.1f GB\n", prop.totalGlobalMem / 1e9);
    printf("SMs: %d, Max threads/block: %d\n", 
           prop.multiProcessorCount, prop.maxThreadsPerBlock);
}
