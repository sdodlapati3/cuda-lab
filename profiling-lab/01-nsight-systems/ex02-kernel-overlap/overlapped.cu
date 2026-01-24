/*
 * overlapped.cu - YOUR TASK: Implement stream-based overlap
 * 
 * Goal: Use multiple CUDA streams to overlap:
 * 1. H2D transfers with kernel execution
 * 2. Kernel execution with D2H transfers
 * 3. Multiple kernels (if resources allow)
 * 
 * TODO: 
 * - Create multiple streams
 * - Use pinned (page-locked) host memory
 * - Use cudaMemcpyAsync instead of cudaMemcpy
 * - Launch kernels in different streams
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define NUM_CHUNKS 4
#define NUM_STREAMS 4  // Try different values: 2, 4, 8
#define CHUNK_SIZE (1 << 22)  // 4M elements per chunk
#define BLOCK_SIZE 256

// Same kernel as sequential version
__global__ void compute_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        for (int i = 0; i < 100; i++) {
            val = val * val + 0.1f;
            val = sqrtf(val);
        }
        data[idx] = val;
    }
}

int main() {
    printf("Overlapped execution with %d streams\n", NUM_STREAMS);
    printf("Chunks: %d, Elements per chunk: %d\n", NUM_CHUNKS, CHUNK_SIZE);
    
    size_t chunk_bytes = CHUNK_SIZE * sizeof(float);
    
    // TODO 1: Create CUDA streams
    // cudaStream_t streams[NUM_STREAMS];
    // for (int i = 0; i < NUM_STREAMS; i++) {
    //     CHECK_CUDA(cudaStreamCreate(&streams[i]));
    // }
    
    // TODO 2: Allocate PINNED host memory (required for async transfers)
    // Use cudaMallocHost instead of malloc
    float* h_data[NUM_CHUNKS];
    for (int i = 0; i < NUM_CHUNKS; i++) {
        // CHANGE THIS: h_data[i] = (float*)malloc(chunk_bytes);
        h_data[i] = (float*)malloc(chunk_bytes);  // Replace with cudaMallocHost
        
        // Initialize
        for (int j = 0; j < CHUNK_SIZE; j++) {
            h_data[i][j] = (float)(i * CHUNK_SIZE + j) / (NUM_CHUNKS * CHUNK_SIZE);
        }
    }
    
    // Allocate device memory
    float* d_data[NUM_CHUNKS];
    for (int i = 0; i < NUM_CHUNKS; i++) {
        CHECK_CUDA(cudaMalloc(&d_data[i], chunk_bytes));
    }
    
    // Timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    int grid_size = (CHUNK_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    CHECK_CUDA(cudaEventRecord(start));
    
    // TODO 3: Implement overlapped execution
    // Strategy: Issue all H2D, then all kernels, then all D2H
    // Each chunk uses stream[i % NUM_STREAMS]
    
    // Current: Sequential (REPLACE THIS)
    for (int i = 0; i < NUM_CHUNKS; i++) {
        // TODO: Use cudaMemcpyAsync with streams[i % NUM_STREAMS]
        CHECK_CUDA(cudaMemcpy(d_data[i], h_data[i], chunk_bytes, cudaMemcpyHostToDevice));
        
        // TODO: Launch kernel in streams[i % NUM_STREAMS]
        compute_kernel<<<grid_size, BLOCK_SIZE>>>(d_data[i], CHUNK_SIZE);
        
        // TODO: Use cudaMemcpyAsync with streams[i % NUM_STREAMS]
        CHECK_CUDA(cudaMemcpy(h_data[i], d_data[i], chunk_bytes, cudaMemcpyDeviceToHost));
    }
    
    // TODO 4: Synchronize all streams
    // for (int i = 0; i < NUM_STREAMS; i++) {
    //     CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    // }
    
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    
    printf("Total time: %.2f ms\n", ms);
    printf("Per chunk: %.2f ms\n", ms / NUM_CHUNKS);
    
    // Verify results
    printf("Sample output[0]: %.6f\n", h_data[0][0]);
    
    // Cleanup
    for (int i = 0; i < NUM_CHUNKS; i++) {
        // TODO: Use cudaFreeHost for pinned memory
        free(h_data[i]);
        CHECK_CUDA(cudaFree(d_data[i]));
    }
    
    // TODO: Destroy streams
    // for (int i = 0; i < NUM_STREAMS; i++) {
    //     CHECK_CUDA(cudaStreamDestroy(streams[i]));
    // }
    
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    return 0;
}
