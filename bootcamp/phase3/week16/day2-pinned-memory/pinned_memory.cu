/**
 * pinned_memory.cu - Page-locked host memory for fast transfers
 * 
 * Learning objectives:
 * - Compare pageable vs pinned transfer speeds
 * - Use async transfers with pinned memory
 * - Overlap transfer with computation
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

// Kernel that takes some time
__global__ void compute_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        for (int i = 0; i < 100; i++) {
            val = sinf(val) + cosf(val);
        }
        data[idx] = val;
    }
}

int main() {
    printf("=== Pinned Memory Demo ===\n\n");
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // ========================================================================
    // Part 1: Bandwidth Comparison
    // ========================================================================
    {
        printf("1. Pageable vs Pinned Transfer Bandwidth\n");
        printf("─────────────────────────────────────────\n");
        
        const size_t sizes[] = {1 << 20, 1 << 24, 1 << 26};  // 1MB, 16MB, 64MB
        
        for (auto size : sizes) {
            // Pageable memory
            float* h_pageable = new float[size / sizeof(float)];
            memset(h_pageable, 0, size);
            
            // Pinned memory
            float* h_pinned;
            cudaMallocHost(&h_pinned, size);
            memset(h_pinned, 0, size);
            
            // Device memory
            float* d_data;
            cudaMalloc(&d_data, size);
            
            // Warmup
            cudaMemcpy(d_data, h_pageable, size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_data, h_pinned, size, cudaMemcpyHostToDevice);
            
            // Measure pageable
            cudaEventRecord(start);
            for (int i = 0; i < 10; i++) {
                cudaMemcpy(d_data, h_pageable, size, cudaMemcpyHostToDevice);
            }
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            
            float pageable_ms;
            cudaEventElapsedTime(&pageable_ms, start, stop);
            pageable_ms /= 10;
            
            // Measure pinned
            cudaEventRecord(start);
            for (int i = 0; i < 10; i++) {
                cudaMemcpy(d_data, h_pinned, size, cudaMemcpyHostToDevice);
            }
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            
            float pinned_ms;
            cudaEventElapsedTime(&pinned_ms, start, stop);
            pinned_ms /= 10;
            
            float pageable_gbps = size / (pageable_ms / 1000) / 1e9;
            float pinned_gbps = size / (pinned_ms / 1000) / 1e9;
            
            printf("   %4zu MB: Pageable %.1f GB/s, Pinned %.1f GB/s (%.1fx)\n",
                   size >> 20, pageable_gbps, pinned_gbps, pinned_gbps / pageable_gbps);
            
            delete[] h_pageable;
            cudaFreeHost(h_pinned);
            cudaFree(d_data);
        }
        printf("\n");
    }
    
    // ========================================================================
    // Part 2: Async Transfer with Overlap
    // ========================================================================
    {
        printf("2. Async Transfer + Compute Overlap\n");
        printf("─────────────────────────────────────────\n");
        
        const int N = 1 << 24;  // 16M elements
        const size_t size = N * sizeof(float);
        const int CHUNKS = 4;
        const int chunk_n = N / CHUNKS;
        const size_t chunk_size = chunk_n * sizeof(float);
        
        // Allocate
        float* h_pinned;
        cudaMallocHost(&h_pinned, size);
        for (int i = 0; i < N; i++) h_pinned[i] = 1.0f;
        
        float* d_data;
        cudaMalloc(&d_data, size);
        
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        
        // Synchronous: transfer all, then compute all
        cudaEventRecord(start);
        cudaMemcpy(d_data, h_pinned, size, cudaMemcpyHostToDevice);
        compute_kernel<<<(N + 255) / 256, 256>>>(d_data, N);
        cudaMemcpy(h_pinned, d_data, size, cudaMemcpyDeviceToHost);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float sync_ms;
        cudaEventElapsedTime(&sync_ms, start, stop);
        
        // Overlapped: pipeline chunks
        cudaEventRecord(start);
        for (int c = 0; c < CHUNKS; c++) {
            int offset = c * chunk_n;
            cudaMemcpyAsync(d_data + offset, h_pinned + offset, chunk_size,
                           cudaMemcpyHostToDevice, stream);
            compute_kernel<<<(chunk_n + 255) / 256, 256, 0, stream>>>(
                d_data + offset, chunk_n);
            cudaMemcpyAsync(h_pinned + offset, d_data + offset, chunk_size,
                           cudaMemcpyDeviceToHost, stream);
        }
        cudaStreamSynchronize(stream);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float overlap_ms;
        cudaEventElapsedTime(&overlap_ms, start, stop);
        
        printf("   Synchronous:  %.2f ms\n", sync_ms);
        printf("   Overlapped:   %.2f ms (%d chunks)\n", overlap_ms, CHUNKS);
        printf("   Speedup: %.2fx\n\n", sync_ms / overlap_ms);
        
        cudaStreamDestroy(stream);
        cudaFreeHost(h_pinned);
        cudaFree(d_data);
    }
    
    // ========================================================================
    // Part 3: Multi-Stream Pipeline
    // ========================================================================
    {
        printf("3. Multi-Stream Pipeline\n");
        printf("─────────────────────────────────────────\n");
        
        const int N = 1 << 24;
        const size_t size = N * sizeof(float);
        const int NUM_STREAMS = 4;
        const int chunk_n = N / NUM_STREAMS;
        const size_t chunk_size = chunk_n * sizeof(float);
        
        float* h_pinned;
        cudaMallocHost(&h_pinned, size);
        for (int i = 0; i < N; i++) h_pinned[i] = 1.0f;
        
        float* d_data;
        cudaMalloc(&d_data, size);
        
        cudaStream_t streams[NUM_STREAMS];
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaStreamCreate(&streams[i]);
        }
        
        cudaEventRecord(start);
        
        // Each stream handles one chunk
        for (int s = 0; s < NUM_STREAMS; s++) {
            int offset = s * chunk_n;
            cudaMemcpyAsync(d_data + offset, h_pinned + offset, chunk_size,
                           cudaMemcpyHostToDevice, streams[s]);
            compute_kernel<<<(chunk_n + 255) / 256, 256, 0, streams[s]>>>(
                d_data + offset, chunk_n);
            cudaMemcpyAsync(h_pinned + offset, d_data + offset, chunk_size,
                           cudaMemcpyDeviceToHost, streams[s]);
        }
        
        for (int s = 0; s < NUM_STREAMS; s++) {
            cudaStreamSynchronize(streams[s]);
        }
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float multi_ms;
        cudaEventElapsedTime(&multi_ms, start, stop);
        
        printf("   %d streams in parallel: %.2f ms\n\n", NUM_STREAMS, multi_ms);
        
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaStreamDestroy(streams[i]);
        }
        cudaFreeHost(h_pinned);
        cudaFree(d_data);
    }
    
    // ========================================================================
    // Part 4: Write-Combining Memory
    // ========================================================================
    {
        printf("4. Write-Combining Memory\n");
        printf("─────────────────────────────────────────\n");
        
        const size_t size = 64 << 20;  // 64 MB
        
        // Regular pinned
        float* h_regular;
        cudaMallocHost(&h_regular, size);
        
        // Write-combining (good for sequential writes)
        float* h_wc;
        cudaHostAlloc(&h_wc, size, cudaHostAllocWriteCombined);
        
        float* d_data;
        cudaMalloc(&d_data, size);
        
        // Measure H2D transfer (write-combining shines here)
        cudaEventRecord(start);
        for (int i = 0; i < 5; i++) {
            cudaMemcpy(d_data, h_regular, size, cudaMemcpyHostToDevice);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float regular_ms;
        cudaEventElapsedTime(&regular_ms, start, stop);
        
        cudaEventRecord(start);
        for (int i = 0; i < 5; i++) {
            cudaMemcpy(d_data, h_wc, size, cudaMemcpyHostToDevice);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float wc_ms;
        cudaEventElapsedTime(&wc_ms, start, stop);
        
        printf("   Regular pinned H2D: %.1f GB/s\n", 
               5 * size / (regular_ms / 1000) / 1e9);
        printf("   Write-combining H2D: %.1f GB/s\n\n",
               5 * size / (wc_ms / 1000) / 1e9);
        
        cudaFreeHost(h_regular);
        cudaFreeHost(h_wc);
        cudaFree(d_data);
    }
    
    printf("=== Key Points ===\n\n");
    printf("1. Pinned memory enables DMA and async transfers\n");
    printf("2. ~2x bandwidth improvement over pageable\n");
    printf("3. Overlap H2D, compute, D2H with chunked pipeline\n");
    printf("4. Write-combining for sequential host writes\n");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
