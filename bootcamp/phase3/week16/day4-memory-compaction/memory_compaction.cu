/**
 * memory_compaction.cu - Memory fragmentation and compaction
 * 
 * Learning objectives:
 * - Demonstrate fragmentation
 * - Simple pool allocator
 * - Memory pool benefits
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <queue>
#include <cstdlib>

// ============================================================================
// Simple Fixed-Size Block Allocator
// ============================================================================

class FixedBlockAllocator {
    char* base_;
    size_t block_size_;
    int num_blocks_;
    std::queue<char*> free_list_;
    
public:
    FixedBlockAllocator(size_t block_size, int num_blocks)
        : block_size_(block_size), num_blocks_(num_blocks) {
        cudaMalloc(&base_, block_size * num_blocks);
        
        // Initialize free list
        for (int i = 0; i < num_blocks; i++) {
            free_list_.push(base_ + i * block_size_);
        }
    }
    
    ~FixedBlockAllocator() {
        cudaFree(base_);
    }
    
    void* alloc() {
        if (free_list_.empty()) return nullptr;
        void* ptr = free_list_.front();
        free_list_.pop();
        return ptr;
    }
    
    void free(void* ptr) {
        free_list_.push(static_cast<char*>(ptr));
    }
    
    int available() const { return free_list_.size(); }
};

// ============================================================================
// Slab Allocator (Multiple Sizes)
// ============================================================================

class SlabAllocator {
    static constexpr int NUM_CLASSES = 4;
    size_t sizes_[NUM_CLASSES] = {256, 1024, 4096, 16384};
    FixedBlockAllocator* slabs_[NUM_CLASSES];
    
public:
    SlabAllocator(int blocks_per_class = 1000) {
        for (int i = 0; i < NUM_CLASSES; i++) {
            slabs_[i] = new FixedBlockAllocator(sizes_[i], blocks_per_class);
        }
    }
    
    ~SlabAllocator() {
        for (int i = 0; i < NUM_CLASSES; i++) {
            delete slabs_[i];
        }
    }
    
    void* alloc(size_t size) {
        for (int i = 0; i < NUM_CLASSES; i++) {
            if (size <= sizes_[i]) {
                return slabs_[i]->alloc();
            }
        }
        // Fallback to regular malloc for large sizes
        void* ptr;
        cudaMalloc(&ptr, size);
        return ptr;
    }
    
    void free(void* ptr, size_t size) {
        for (int i = 0; i < NUM_CLASSES; i++) {
            if (size <= sizes_[i]) {
                slabs_[i]->free(ptr);
                return;
            }
        }
        cudaFree(ptr);
    }
    
    void print_status() {
        for (int i = 0; i < NUM_CLASSES; i++) {
            printf("   Slab %5zu bytes: %d blocks available\n",
                   sizes_[i], slabs_[i]->available());
        }
    }
};

// Simple kernel
__global__ void touch_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = 1.0f;
}

int main() {
    printf("=== Memory Compaction Demo ===\n\n");
    
    // ========================================================================
    // Part 1: Demonstrate Fragmentation
    // ========================================================================
    {
        printf("1. Fragmentation Demonstration\n");
        printf("─────────────────────────────────────────\n");
        
        std::vector<void*> ptrs;
        
        // Allocate many small blocks
        printf("   Allocating 100 blocks of varying sizes...\n");
        for (int i = 0; i < 100; i++) {
            void* ptr;
            size_t size = (1 << (10 + rand() % 5)) * sizeof(float);  // 1KB to 16KB
            cudaMalloc(&ptr, size);
            ptrs.push_back(ptr);
        }
        
        // Free every other block
        printf("   Freeing every other block...\n");
        for (int i = 0; i < 100; i += 2) {
            cudaFree(ptrs[i]);
            ptrs[i] = nullptr;
        }
        
        // Try to allocate one large block
        void* large_ptr;
        size_t large_size = 50 << 20;  // 50 MB
        cudaError_t err = cudaMalloc(&large_ptr, large_size);
        
        printf("   Attempting 50MB allocation: %s\n",
               err == cudaSuccess ? "SUCCESS" : "FAILED (fragmented)");
        
        // Cleanup
        for (auto ptr : ptrs) {
            if (ptr) cudaFree(ptr);
        }
        if (err == cudaSuccess) cudaFree(large_ptr);
        cudaGetLastError();  // Clear error
        printf("\n");
    }
    
    // ========================================================================
    // Part 2: Fixed Block Allocator
    // ========================================================================
    {
        printf("2. Fixed Block Allocator\n");
        printf("─────────────────────────────────────────\n");
        
        const int BLOCK_SIZE = 4096;
        const int NUM_BLOCKS = 1000;
        
        FixedBlockAllocator allocator(BLOCK_SIZE, NUM_BLOCKS);
        
        printf("   Block size: %d bytes, Pool: %d blocks\n", BLOCK_SIZE, NUM_BLOCKS);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Allocate and free many times
        std::vector<void*> allocated;
        for (int iter = 0; iter < 10000; iter++) {
            // Allocate some
            for (int i = 0; i < 10; i++) {
                void* p = allocator.alloc();
                if (p) allocated.push_back(p);
            }
            // Free some
            while (allocated.size() > 5) {
                allocator.free(allocated.back());
                allocated.pop_back();
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(end - start).count();
        
        printf("   10000 iterations of alloc/free: %.0f μs\n", us);
        printf("   Per operation: %.1f ns\n", us * 1000 / 150000);  // ~15 ops per iter
        printf("   Blocks available: %d\n\n", allocator.available());
    }
    
    // ========================================================================
    // Part 3: Slab Allocator
    // ========================================================================
    {
        printf("3. Slab Allocator (Multiple Sizes)\n");
        printf("─────────────────────────────────────────\n");
        
        SlabAllocator allocator(500);
        
        printf("   Initial state:\n");
        allocator.print_status();
        
        // Simulate mixed workload
        std::vector<std::pair<void*, size_t>> allocated;
        
        for (int i = 0; i < 1000; i++) {
            size_t sizes[] = {100, 500, 2000, 10000};
            size_t size = sizes[rand() % 4];
            
            void* ptr = allocator.alloc(size);
            if (ptr) allocated.push_back({ptr, size});
            
            // Occasionally free
            if (allocated.size() > 200 && rand() % 2) {
                auto [p, s] = allocated.back();
                allocator.free(p, s);
                allocated.pop_back();
            }
        }
        
        printf("\n   After workload:\n");
        allocator.print_status();
        
        // Cleanup
        for (auto [p, s] : allocated) {
            allocator.free(p, s);
        }
        
        printf("\n   After cleanup:\n");
        allocator.print_status();
        printf("\n");
    }
    
    // ========================================================================
    // Part 4: CUDA Memory Pool (Built-in Solution)
    // ========================================================================
    {
        printf("4. CUDA Memory Pool (Stream-Ordered)\n");
        printf("─────────────────────────────────────────\n");
        
        int device;
        cudaGetDevice(&device);
        
        int poolSupport;
        cudaDeviceGetAttribute(&poolSupport, cudaDevAttrMemoryPoolsSupported, device);
        
        if (!poolSupport) {
            printf("   Memory pools not supported.\n\n");
        } else {
            cudaStream_t stream;
            cudaStreamCreate(&stream);
            
            cudaMemPool_t pool;
            cudaDeviceGetDefaultMemPool(&pool, device);
            
            // Many allocations through pool
            auto start = std::chrono::high_resolution_clock::now();
            
            for (int i = 0; i < 1000; i++) {
                float* ptr;
                cudaMallocAsync(&ptr, 4096, stream);
                touch_kernel<<<1, 256, 0, stream>>>(ptr, 1024);
                cudaFreeAsync(ptr, stream);
            }
            cudaStreamSynchronize(stream);
            
            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(end - start).count();
            
            printf("   1000 async alloc/free cycles: %.1f ms\n", ms);
            printf("   Per cycle: %.1f μs\n", ms * 1000 / 1000);
            
            // Pool stats
            size_t reserved, used;
            cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &reserved);
            cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used);
            
            printf("   Pool reserved: %.2f KB (reusable)\n", reserved / 1024.0f);
            
            cudaStreamDestroy(stream);
        }
        printf("\n");
    }
    
    printf("=== Key Points ===\n\n");
    printf("1. Random alloc/free leads to fragmentation\n");
    printf("2. Fixed-size pools eliminate fragmentation\n");
    printf("3. Slab allocator handles multiple sizes\n");
    printf("4. CUDA memory pools are the modern solution\n");
    printf("5. Stream-ordered allocs enable reuse\n");
    
    return 0;
}
