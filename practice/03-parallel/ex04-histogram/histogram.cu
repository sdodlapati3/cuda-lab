#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

#define NUM_BINS 256
#define BLOCK_SIZE 256

// =============================================================================
// TODO 1: Naive histogram with global atomics
// =============================================================================
__global__ void histogram_global(unsigned int* hist, const unsigned char* data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // TODO: Each thread atomically increments global histogram
    // if (idx < n) {
    //     atomicAdd(&hist[data[idx]], 1);
    // }
}

// =============================================================================
// TODO 2: Privatized histogram with shared memory
// =============================================================================
__global__ void histogram_shared(unsigned int* hist, const unsigned char* data, int n) {
    __shared__ unsigned int local_hist[NUM_BINS];
    
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // TODO: Initialize local histogram to 0
    // if (tid < NUM_BINS) local_hist[tid] = 0;
    // __syncthreads();
    
    // TODO: Each thread updates local histogram
    // if (idx < n) atomicAdd(&local_hist[data[idx]], 1);
    // __syncthreads();
    
    // TODO: Merge local histogram to global
    // if (tid < NUM_BINS) atomicAdd(&hist[tid], local_hist[tid]);
}

void cpu_histogram(unsigned int* hist, const unsigned char* data, int n) {
    for (int i = 0; i < NUM_BINS; i++) hist[i] = 0;
    for (int i = 0; i < n; i++) hist[data[i]]++;
}

int main() {
    const int N = 64 * 1024 * 1024;  // 64 MB of data
    const int iterations = 100;
    
    printf("Parallel Histogram Exercise\n");
    printf("===========================\n");
    printf("Data size: %d bytes, Bins: %d\n\n", N, NUM_BINS);
    
    unsigned char *h_data;
    unsigned int *h_hist, *h_ref;
    unsigned char *d_data;
    unsigned int *d_hist;
    
    h_data = (unsigned char*)malloc(N);
    h_hist = (unsigned int*)malloc(NUM_BINS * sizeof(unsigned int));
    h_ref = (unsigned int*)malloc(NUM_BINS * sizeof(unsigned int));
    
    // Initialize with random data
    for (int i = 0; i < N; i++) h_data[i] = rand() % NUM_BINS;
    cpu_histogram(h_ref, h_data, N);
    
    CHECK_CUDA(cudaMalloc(&d_data, N));
    CHECK_CUDA(cudaMalloc(&d_hist, NUM_BINS * sizeof(unsigned int)));
    CHECK_CUDA(cudaMemcpy(d_data, h_data, N, cudaMemcpyHostToDevice));
    
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    // Benchmark global atomics
    CHECK_CUDA(cudaMemset(d_hist, 0, NUM_BINS * sizeof(unsigned int)));
    histogram_global<<<numBlocks, BLOCK_SIZE>>>(d_hist, d_data, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        CHECK_CUDA(cudaMemset(d_hist, 0, NUM_BINS * sizeof(unsigned int)));
        histogram_global<<<numBlocks, BLOCK_SIZE>>>(d_hist, d_data, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float time_global;
    CHECK_CUDA(cudaEventElapsedTime(&time_global, start, stop));
    time_global /= iterations;
    
    CHECK_CUDA(cudaMemcpy(h_hist, d_hist, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    bool global_correct = true;
    for (int i = 0; i < NUM_BINS; i++) {
        if (h_hist[i] != h_ref[i]) { global_correct = false; break; }
    }
    printf("Global Atomics: %.3f ms [%s]\n", time_global, global_correct ? "PASS" : "FAIL");
    
    // Benchmark shared memory
    CHECK_CUDA(cudaMemset(d_hist, 0, NUM_BINS * sizeof(unsigned int)));
    histogram_shared<<<numBlocks, BLOCK_SIZE>>>(d_hist, d_data, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        CHECK_CUDA(cudaMemset(d_hist, 0, NUM_BINS * sizeof(unsigned int)));
        histogram_shared<<<numBlocks, BLOCK_SIZE>>>(d_hist, d_data, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float time_shared;
    CHECK_CUDA(cudaEventElapsedTime(&time_shared, start, stop));
    time_shared /= iterations;
    
    CHECK_CUDA(cudaMemcpy(h_hist, d_hist, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    bool shared_correct = true;
    for (int i = 0; i < NUM_BINS; i++) {
        if (h_hist[i] != h_ref[i]) { shared_correct = false; break; }
    }
    printf("Shared Memory: %.3f ms [%s]\n", time_shared, shared_correct ? "PASS" : "FAIL");
    printf("\nSpeedup: %.2fx\n", time_global / time_shared);
    
    free(h_data); free(h_hist); free(h_ref);
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_hist));
    
    return 0;
}
