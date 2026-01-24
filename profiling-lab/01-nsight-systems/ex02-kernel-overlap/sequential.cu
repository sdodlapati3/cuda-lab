/*
 * sequential.cu - Baseline: All operations in default stream
 * 
 * This demonstrates poor GPU utilization due to sequential execution.
 * Profile with: nsys profile --stats=true ./sequential
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
#define CHUNK_SIZE (1 << 22)  // 4M elements per chunk
#define BLOCK_SIZE 256

// Simple compute kernel - square each element
__global__ void compute_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Do some work to make kernel visible in profiler
        float val = data[idx];
        for (int i = 0; i < 100; i++) {
            val = val * val + 0.1f;
            val = sqrtf(val);
        }
        data[idx] = val;
    }
}

int main() {
    printf("Sequential execution (default stream)\n");
    printf("Chunks: %d, Elements per chunk: %d\n", NUM_CHUNKS, CHUNK_SIZE);
    
    size_t chunk_bytes = CHUNK_SIZE * sizeof(float);
    
    // Allocate host memory (regular, not pinned)
    float* h_data[NUM_CHUNKS];
    for (int i = 0; i < NUM_CHUNKS; i++) {
        h_data[i] = (float*)malloc(chunk_bytes);
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
    
    // Sequential execution - all in default stream
    for (int i = 0; i < NUM_CHUNKS; i++) {
        // H2D transfer
        CHECK_CUDA(cudaMemcpy(d_data[i], h_data[i], chunk_bytes, cudaMemcpyHostToDevice));
        
        // Compute
        compute_kernel<<<grid_size, BLOCK_SIZE>>>(d_data[i], CHUNK_SIZE);
        
        // D2H transfer
        CHECK_CUDA(cudaMemcpy(h_data[i], d_data[i], chunk_bytes, cudaMemcpyDeviceToHost));
    }
    
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    
    printf("Total time: %.2f ms\n", ms);
    printf("Per chunk: %.2f ms\n", ms / NUM_CHUNKS);
    
    // Verify results (spot check)
    printf("Sample output[0]: %.6f\n", h_data[0][0]);
    
    // Cleanup
    for (int i = 0; i < NUM_CHUNKS; i++) {
        free(h_data[i]);
        CHECK_CUDA(cudaFree(d_data[i]));
    }
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    return 0;
}
