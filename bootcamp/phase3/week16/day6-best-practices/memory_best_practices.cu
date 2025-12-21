/**
 * memory_best_practices.cu - Production memory patterns
 * 
 * Learning objectives:
 * - Apply best practices
 * - Avoid common pitfalls
 * - Build robust memory management
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <memory>

// ============================================================================
// RAII Wrapper for CUDA Memory
// ============================================================================

template<typename T>
class CudaBuffer {
    T* ptr_ = nullptr;
    size_t size_ = 0;
    
public:
    CudaBuffer() = default;
    
    explicit CudaBuffer(size_t count) : size_(count) {
        cudaMalloc(&ptr_, count * sizeof(T));
    }
    
    ~CudaBuffer() {
        if (ptr_) cudaFree(ptr_);
    }
    
    // Move only
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;
    
    CudaBuffer(CudaBuffer&& other) noexcept 
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    
    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFree(ptr_);
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    
    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    size_t size() const { return size_; }
    
    void resize(size_t count) {
        if (count > size_) {
            if (ptr_) cudaFree(ptr_);
            cudaMalloc(&ptr_, count * sizeof(T));
            size_ = count;
        }
    }
};

// ============================================================================
// Memory Pool Wrapper
// ============================================================================

class StreamBuffer {
    void* ptr_ = nullptr;
    size_t size_ = 0;
    cudaStream_t stream_;
    
public:
    StreamBuffer(size_t bytes, cudaStream_t stream) 
        : size_(bytes), stream_(stream) {
        cudaMallocAsync(&ptr_, bytes, stream);
    }
    
    ~StreamBuffer() {
        if (ptr_) cudaFreeAsync(ptr_, stream_);
    }
    
    void* get() { return ptr_; }
};

// ============================================================================
// Simple kernel
// ============================================================================

__global__ void process(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] *= 2.0f;
}

int main() {
    printf("=== Memory Best Practices Demo ===\n\n");
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // ========================================================================
    // Part 1: RAII Pattern
    // ========================================================================
    {
        printf("1. RAII Memory Management\n");
        printf("─────────────────────────────────────────\n");
        
        // Automatic cleanup, exception-safe
        {
            CudaBuffer<float> buffer(1024);
            printf("   Allocated %zu floats at %p\n", buffer.size(), buffer.get());
            
            process<<<4, 256>>>(buffer.get(), 1024);
            cudaDeviceSynchronize();
            
            // Automatically freed when scope exits
        }
        printf("   Buffer automatically freed\n");
        
        // Resize pattern
        {
            CudaBuffer<float> buffer;
            
            for (int size = 1024; size <= 16384; size *= 2) {
                buffer.resize(size);
                printf("   Resized to %d (only reallocates if larger)\n", size);
            }
        }
        printf("\n");
    }
    
    // ========================================================================
    // Part 2: Allocation Reuse
    // ========================================================================
    {
        printf("2. Allocation Reuse Pattern\n");
        printf("─────────────────────────────────────────\n");
        
        const int N = 1 << 20;
        const int ITERATIONS = 100;
        
        // Bad: Allocate each iteration
        cudaEventRecord(start);
        for (int i = 0; i < ITERATIONS; i++) {
            float* ptr;
            cudaMalloc(&ptr, N * sizeof(float));
            process<<<(N+255)/256, 256>>>(ptr, N);
            cudaDeviceSynchronize();
            cudaFree(ptr);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float bad_ms;
        cudaEventElapsedTime(&bad_ms, start, stop);
        
        // Good: Allocate once, reuse
        float* persistent;
        cudaMalloc(&persistent, N * sizeof(float));
        
        cudaEventRecord(start);
        for (int i = 0; i < ITERATIONS; i++) {
            process<<<(N+255)/256, 256>>>(persistent, N);
            cudaDeviceSynchronize();
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float good_ms;
        cudaEventElapsedTime(&good_ms, start, stop);
        
        cudaFree(persistent);
        
        printf("   Allocate each time: %.2f ms\n", bad_ms);
        printf("   Reuse allocation:   %.2f ms\n", good_ms);
        printf("   Speedup: %.2fx\n\n", bad_ms / good_ms);
    }
    
    // ========================================================================
    // Part 3: Memory Pool Pattern
    // ========================================================================
    {
        printf("3. Memory Pool for Variable Sizes\n");
        printf("─────────────────────────────────────────\n");
        
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        
        size_t sizes[] = {1024, 4096, 16384, 65536, 262144};
        
        cudaEventRecord(start);
        for (int iter = 0; iter < 100; iter++) {
            for (auto size : sizes) {
                StreamBuffer buf(size * sizeof(float), stream);
                process<<<(size+255)/256, 256, 0, stream>>>(
                    static_cast<float*>(buf.get()), size);
            }
        }
        cudaStreamSynchronize(stream);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float pool_ms;
        cudaEventElapsedTime(&pool_ms, start, stop);
        
        printf("   500 variable-size allocations: %.2f ms\n", pool_ms);
        printf("   Per allocation: %.1f μs\n", pool_ms * 1000 / 500);
        
        cudaStreamDestroy(stream);
        printf("\n");
    }
    
    // ========================================================================
    // Part 4: Memory Monitoring
    // ========================================================================
    {
        printf("4. Memory Monitoring\n");
        printf("─────────────────────────────────────────\n");
        
        size_t free_start, total;
        cudaMemGetInfo(&free_start, &total);
        
        printf("   Total GPU memory: %.2f GB\n", total / 1e9);
        printf("   Free before:      %.2f GB\n", free_start / 1e9);
        
        // Allocate some memory
        std::vector<CudaBuffer<float>> buffers;
        for (int i = 0; i < 10; i++) {
            buffers.emplace_back(10 << 20);  // 10M floats = 40MB each
        }
        
        size_t free_after;
        cudaMemGetInfo(&free_after, &total);
        
        printf("   Free after 400MB: %.2f GB\n", free_after / 1e9);
        printf("   Allocated:        %.0f MB\n", (free_start - free_after) / 1e6);
        
        // Clear
        buffers.clear();
        
        cudaMemGetInfo(&free_after, &total);
        printf("   Free after clear: %.2f GB\n\n", free_after / 1e9);
    }
    
    // ========================================================================
    // Part 5: Alignment Best Practice
    // ========================================================================
    {
        printf("5. Aligned Allocations\n");
        printf("─────────────────────────────────────────\n");
        
        auto align_size = [](size_t size, size_t alignment) {
            return ((size + alignment - 1) / alignment) * alignment;
        };
        
        size_t sizes[] = {100, 1000, 10000, 100000};
        
        printf("   %-10s %-12s %-12s\n", "Original", "Aligned 256", "Overhead");
        for (auto size : sizes) {
            size_t aligned = align_size(size, 256);
            float overhead = 100.0f * (aligned - size) / size;
            printf("   %-10zu %-12zu %.1f%%\n", size, aligned, overhead);
        }
        printf("\n");
    }
    
    // ========================================================================
    // Summary
    // ========================================================================
    printf("═══════════════════════════════════════════════════════════\n");
    printf("MEMORY BEST PRACTICES SUMMARY\n");
    printf("═══════════════════════════════════════════════════════════\n\n");
    
    printf("DO:\n");
    printf("  ✓ Use RAII wrappers for automatic cleanup\n");
    printf("  ✓ Allocate early, reuse buffers\n");
    printf("  ✓ Use memory pools for frequent allocations\n");
    printf("  ✓ Monitor memory usage in production\n");
    printf("  ✓ Align allocations to 256 bytes\n");
    printf("  ✓ Use pinned memory for async transfers\n\n");
    
    printf("DON'T:\n");
    printf("  ✗ Allocate/free in hot loops\n");
    printf("  ✗ Forget cleanup on error paths\n");
    printf("  ✗ Use pageable memory with cudaMemcpyAsync\n");
    printf("  ✗ Over-allocate pinned memory\n");
    printf("  ✗ Ignore fragmentation in long-running apps\n\n");
    
    printf("ALLOCATION LATENCY GUIDE:\n");
    printf("  cudaMalloc:       ~1000 μs\n");
    printf("  cudaMallocAsync:  ~10 μs\n");
    printf("  Custom pool:      ~1 μs\n");
    printf("  cudaMallocHost:   ~1000 μs\n\n");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
