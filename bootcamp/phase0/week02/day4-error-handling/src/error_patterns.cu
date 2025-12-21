/**
 * Day 4: Error Handling Patterns
 * 
 * Production-quality error handling examples.
 */

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// ============================================================================
// Pattern 1: Simple macro (most common)
// ============================================================================
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// ============================================================================
// Pattern 2: Kernel check macro
// ============================================================================
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

// ============================================================================
// Pattern 3: Exception-based (C++)
// ============================================================================
#include <stdexcept>
#include <string>

class CudaException : public std::runtime_error {
public:
    CudaException(cudaError_t err, const char* file, int line)
        : std::runtime_error(make_message(err, file, line)), error_(err) {}
    
    cudaError_t error() const { return error_; }

private:
    static std::string make_message(cudaError_t err, const char* file, int line) {
        return std::string("CUDA error at ") + file + ":" + std::to_string(line) 
               + ": " + cudaGetErrorString(err);
    }
    cudaError_t error_;
};

#define CUDA_THROW(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw CudaException(err, __FILE__, __LINE__); \
    } \
} while(0)

// ============================================================================
// Pattern 4: Error code return (C-style)
// ============================================================================
typedef cudaError_t CudaResult;

CudaResult safe_malloc(void** ptr, size_t size) {
    return cudaMalloc(ptr, size);
}

CudaResult run_kernel_safe(int* data, int n) {
    // Kernel here
    return cudaGetLastError();
}

// ============================================================================
// Pattern 5: RAII cleanup on error
// ============================================================================
class CudaBuffer {
public:
    CudaBuffer(size_t size) : ptr_(nullptr), size_(size) {
        CUDA_CHECK(cudaMalloc(&ptr_, size));
    }
    
    ~CudaBuffer() {
        if (ptr_) cudaFree(ptr_);
    }
    
    // Non-copyable
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;
    
    // Movable
    CudaBuffer(CudaBuffer&& other) noexcept 
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
    }
    
    void* get() { return ptr_; }
    size_t size() const { return size_; }

private:
    void* ptr_;
    size_t size_;
};

// ============================================================================
// Simple test kernels
// ============================================================================
__global__ void good_kernel(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = idx;
}

__global__ void bad_kernel(int* data) {
    data[10000000] = 42;  // OOB
}

// ============================================================================
// Demonstrations
// ============================================================================
void demo_simple_pattern() {
    printf("\n=== Pattern 1: Simple Macro ===\n");
    
    int* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, 100 * sizeof(int)));
    
    good_kernel<<<1, 32>>>(d_data, 100);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("  Kernel executed successfully\n");
    
    CUDA_CHECK(cudaFree(d_data));
}

void demo_kernel_check_pattern() {
    printf("\n=== Pattern 2: Kernel Check ===\n");
    
    int* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, 100 * sizeof(int)));
    
    good_kernel<<<1, 32>>>(d_data, 100);
    CUDA_CHECK_KERNEL();  // Checks both launch and execution
    
    printf("  Kernel check passed\n");
    
    CUDA_CHECK(cudaFree(d_data));
}

void demo_exception_pattern() {
    printf("\n=== Pattern 3: Exception-Based ===\n");
    
    try {
        int* d_data;
        CUDA_THROW(cudaMalloc(&d_data, 100 * sizeof(int)));
        
        good_kernel<<<1, 32>>>(d_data, 100);
        CUDA_THROW(cudaGetLastError());
        CUDA_THROW(cudaDeviceSynchronize());
        
        printf("  Kernel executed successfully\n");
        
        CUDA_THROW(cudaFree(d_data));
    } catch (const CudaException& e) {
        printf("  Caught: %s\n", e.what());
    }
}

void demo_raii_pattern() {
    printf("\n=== Pattern 5: RAII ===\n");
    
    {
        CudaBuffer buf(100 * sizeof(int));
        printf("  Allocated %zu bytes\n", buf.size());
        
        good_kernel<<<1, 32>>>((int*)buf.get(), 100);
        CUDA_CHECK_KERNEL();
        
        printf("  Buffer auto-freed when going out of scope\n");
    }
    // buf is freed here
    
    printf("  RAII cleanup complete\n");
}

int main() {
    printf("CUDA Error Handling Patterns\n");
    printf("============================\n");
    
    demo_simple_pattern();
    demo_kernel_check_pattern();
    demo_exception_pattern();
    demo_raii_pattern();
    
    printf("\n=== Recommendations ===\n");
    printf("Development: Use CUDA_CHECK_KERNEL (sync after each kernel)\n");
    printf("Production:  Use CUDA_CHECK + periodic sync for performance\n");
    printf("Libraries:   Use exception-based for clean error propagation\n");
    printf("Memory:      Always use RAII for automatic cleanup\n");
    
    return 0;
}
