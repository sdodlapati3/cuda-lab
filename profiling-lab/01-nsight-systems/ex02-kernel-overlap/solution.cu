/*
 * solution.cu - Reference solution for stream overlap
 * 
 * Demonstrates proper use of:
 * - Multiple CUDA streams for concurrency
 * - Pinned memory for async transfers
 * - Overlap of H2D, compute, and D2H
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
#define NUM_STREAMS 4
#define CHUNK_SIZE (1 << 22)
#define BLOCK_SIZE 256

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
    printf("Overlapped execution with %d streams (SOLUTION)\n", NUM_STREAMS);
    printf("Chunks: %d, Elements per chunk: %d\n", NUM_CHUNKS, CHUNK_SIZE);
    
    size_t chunk_bytes = CHUNK_SIZE * sizeof(float);
    
    // Create streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
    }
    
    // Allocate PINNED host memory (critical for async transfers!)
    float* h_data[NUM_CHUNKS];
    for (int i = 0; i < NUM_CHUNKS; i++) {
        CHECK_CUDA(cudaMallocHost(&h_data[i], chunk_bytes));
        for (int j = 0; j < CHUNK_SIZE; j++) {
            h_data[i][j] = (float)(i * CHUNK_SIZE + j) / (NUM_CHUNKS * CHUNK_SIZE);
        }
    }
    
    // Allocate device memory
    float* d_data[NUM_CHUNKS];
    for (int i = 0; i < NUM_CHUNKS; i++) {
        CHECK_CUDA(cudaMalloc(&d_data[i], chunk_bytes));
    }
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    int grid_size = (CHUNK_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    CHECK_CUDA(cudaEventRecord(start));
    
    // Strategy 1: Breadth-first (issue all of one type, then next)
    // This maximizes overlap opportunity
    
    // Issue all H2D transfers
    for (int i = 0; i < NUM_CHUNKS; i++) {
        int stream_idx = i % NUM_STREAMS;
        CHECK_CUDA(cudaMemcpyAsync(d_data[i], h_data[i], chunk_bytes, 
                                   cudaMemcpyHostToDevice, streams[stream_idx]));
    }
    
    // Issue all kernels
    for (int i = 0; i < NUM_CHUNKS; i++) {
        int stream_idx = i % NUM_STREAMS;
        compute_kernel<<<grid_size, BLOCK_SIZE, 0, streams[stream_idx]>>>(d_data[i], CHUNK_SIZE);
    }
    
    // Issue all D2H transfers
    for (int i = 0; i < NUM_CHUNKS; i++) {
        int stream_idx = i % NUM_STREAMS;
        CHECK_CUDA(cudaMemcpyAsync(h_data[i], d_data[i], chunk_bytes,
                                   cudaMemcpyDeviceToHost, streams[stream_idx]));
    }
    
    // Synchronize all streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }
    
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    
    printf("Total time: %.2f ms\n", ms);
    printf("Per chunk (amortized): %.2f ms\n", ms / NUM_CHUNKS);
    
    printf("Sample output[0]: %.6f\n", h_data[0][0]);
    
    // Cleanup
    for (int i = 0; i < NUM_CHUNKS; i++) {
        CHECK_CUDA(cudaFreeHost(h_data[i]));
        CHECK_CUDA(cudaFree(d_data[i]));
    }
    for (int i = 0; i < NUM_STREAMS; i++) {
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
    }
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    return 0;
}
