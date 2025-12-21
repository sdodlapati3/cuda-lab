/**
 * memory_pools.cu - Stream-ordered memory allocation
 * 
 * Learning objectives:
 * - Compare cudaMalloc vs cudaMallocAsync
 * - Configure memory pools
 * - Measure allocation overhead
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

// Simple kernel for testing
__global__ void dummy_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}

int main() {
    printf("=== Memory Pools Demo ===\n\n");
    
    const int N = 1 << 20;  // 1M elements
    const size_t size = N * sizeof(float);
    const int NUM_ITERATIONS = 100;
    
    // Check CUDA version supports memory pools
    int device;
    cudaGetDevice(&device);
    
    int memPoolSupport;
    cudaDeviceGetAttribute(&memPoolSupport, 
                           cudaDevAttrMemoryPoolsSupported, device);
    
    if (!memPoolSupport) {
        printf("Memory pools not supported on this device.\n");
        return 1;
    }
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // ========================================================================
    // Part 1: Compare cudaMalloc vs cudaMallocAsync
    // ========================================================================
    {
        printf("1. Allocation Latency Comparison\n");
        printf("─────────────────────────────────────────\n");
        
        float* ptr;
        
        // Warmup
        cudaMalloc(&ptr, size);
        cudaFree(ptr);
        
        // Measure cudaMalloc
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            cudaMalloc(&ptr, size);
            cudaFree(ptr);
        }
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        
        double malloc_us = std::chrono::duration<double, std::micro>(end - start).count() 
                           / NUM_ITERATIONS;
        
        // Warmup pool
        cudaMallocAsync(&ptr, size, stream);
        cudaFreeAsync(ptr, stream);
        cudaStreamSynchronize(stream);
        
        // Measure cudaMallocAsync
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            cudaMallocAsync(&ptr, size, stream);
            cudaFreeAsync(ptr, stream);
        }
        cudaStreamSynchronize(stream);
        end = std::chrono::high_resolution_clock::now();
        
        double async_us = std::chrono::duration<double, std::micro>(end - start).count()
                          / NUM_ITERATIONS;
        
        printf("   cudaMalloc/Free:      %8.1f μs per pair\n", malloc_us);
        printf("   cudaMallocAsync/Free: %8.1f μs per pair\n", async_us);
        printf("   Speedup: %.1fx\n\n", malloc_us / async_us);
    }
    
    // ========================================================================
    // Part 2: Memory Pool with Kernels
    // ========================================================================
    {
        printf("2. Pool Allocation with Kernel Execution\n");
        printf("─────────────────────────────────────────\n");
        
        float* ptr;
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        // Traditional pattern
        cudaEventRecord(start);
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            cudaMalloc(&ptr, size);
            dummy_kernel<<<(N + 255) / 256, 256>>>(ptr, N);
            cudaDeviceSynchronize();
            cudaFree(ptr);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float malloc_ms;
        cudaEventElapsedTime(&malloc_ms, start, stop);
        
        // Pool pattern
        cudaEventRecord(start);
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            cudaMallocAsync(&ptr, size, stream);
            dummy_kernel<<<(N + 255) / 256, 256, 0, stream>>>(ptr, N);
            cudaFreeAsync(ptr, stream);
        }
        cudaStreamSynchronize(stream);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float pool_ms;
        cudaEventElapsedTime(&pool_ms, start, stop);
        
        printf("   With cudaMalloc:      %.2f ms total\n", malloc_ms);
        printf("   With cudaMallocAsync: %.2f ms total\n", pool_ms);
        printf("   Speedup: %.2fx\n\n", malloc_ms / pool_ms);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    // ========================================================================
    // Part 3: Pool Configuration
    // ========================================================================
    {
        printf("3. Memory Pool Configuration\n");
        printf("─────────────────────────────────────────\n");
        
        cudaMemPool_t pool;
        cudaDeviceGetDefaultMemPool(&pool, device);
        
        // Get pool statistics
        size_t reservedMem, usedMem;
        cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &reservedMem);
        cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &usedMem);
        
        printf("   Pool reserved: %.2f MB\n", reservedMem / 1e6);
        printf("   Pool used:     %.2f MB\n", usedMem / 1e6);
        
        // Allocate some memory
        float* ptrs[10];
        for (int i = 0; i < 10; i++) {
            cudaMallocAsync(&ptrs[i], size, stream);
        }
        cudaStreamSynchronize(stream);
        
        cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &reservedMem);
        cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &usedMem);
        
        printf("   After 10 allocs:\n");
        printf("     Reserved: %.2f MB\n", reservedMem / 1e6);
        printf("     Used:     %.2f MB\n", usedMem / 1e6);
        
        // Free all
        for (int i = 0; i < 10; i++) {
            cudaFreeAsync(ptrs[i], stream);
        }
        cudaStreamSynchronize(stream);
        
        cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &reservedMem);
        cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &usedMem);
        
        printf("   After freeing:\n");
        printf("     Reserved: %.2f MB (cached for reuse)\n", reservedMem / 1e6);
        printf("     Used:     %.2f MB\n", usedMem / 1e6);
        
        // Trim pool
        cudaMemPoolTrimTo(pool, 0);
        
        cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &reservedMem);
        printf("   After trim:   %.2f MB reserved\n\n", reservedMem / 1e6);
    }
    
    // ========================================================================
    // Part 4: Varying Allocation Sizes
    // ========================================================================
    {
        printf("4. Varying Allocation Sizes\n");
        printf("─────────────────────────────────────────\n");
        
        size_t sizes[] = {1 << 10, 1 << 15, 1 << 20, 1 << 25};  // 1KB to 32MB
        
        for (auto alloc_size : sizes) {
            float* ptr;
            
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < 100; i++) {
                cudaMallocAsync(&ptr, alloc_size, stream);
                cudaFreeAsync(ptr, stream);
            }
            cudaStreamSynchronize(stream);
            auto end = std::chrono::high_resolution_clock::now();
            
            double us = std::chrono::duration<double, std::micro>(end - start).count() / 100;
            
            printf("   Size %8zu bytes: %.1f μs per alloc/free\n", alloc_size, us);
        }
        printf("\n");
    }
    
    printf("=== Key Points ===\n\n");
    printf("1. cudaMallocAsync is 10-100x faster than cudaMalloc\n");
    printf("2. Memory stays in pool after FreeAsync (cached)\n");
    printf("3. Use cudaMemPoolTrimTo to release cached memory\n");
    printf("4. Pool allocations are stream-ordered (no sync needed)\n");
    
    cudaStreamDestroy(stream);
    
    return 0;
}
