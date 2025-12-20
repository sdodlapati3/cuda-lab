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

#define NUM_STREAMS 4

__global__ void process_kernel(float* out, const float* in, int n, int offset) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        // Some computation
        float val = in[idx];
        for (int i = 0; i < 100; i++) {
            val = sinf(val) * cosf(val);
        }
        out[idx] = val;
    }
}

// =============================================================================
// TODO 1: Sequential version (single stream)
// =============================================================================
float run_sequential(float* d_in, float* d_out, float* h_in, float* h_out, int N) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    CHECK_CUDA(cudaEventRecord(start));
    
    // TODO: Copy all data to device
    // TODO: Launch kernel for all data
    // TODO: Copy all results back
    
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    return ms;
}

// =============================================================================
// TODO 2: Overlapped version (multiple streams)
// =============================================================================
float run_overlapped(float* d_in, float* d_out, float* h_in, float* h_out, 
                     int N, int numStreams) {
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < numStreams; i++) {
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
    }
    
    int chunkSize = N / numStreams;
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    CHECK_CUDA(cudaEventRecord(start));
    
    // TODO: For each stream:
    // 1. cudaMemcpyAsync H2D for chunk
    // 2. Launch kernel for chunk in stream
    // 3. cudaMemcpyAsync D2H for chunk
    
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    
    for (int i = 0; i < numStreams; i++) {
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
    }
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    return ms;
}

int main() {
    const int N = 4 * 1024 * 1024;  // 4M elements
    
    printf("CUDA Streams Exercise\n");
    printf("=====================\n");
    printf("Array size: %d elements\n\n", N);
    
    // Allocate pinned host memory (required for async transfers)
    float *h_in, *h_out;
    CHECK_CUDA(cudaMallocHost(&h_in, N * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&h_out, N * sizeof(float)));
    
    // Initialize
    for (int i = 0; i < N; i++) h_in[i] = (float)i;
    
    // Allocate device memory
    float *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, N * sizeof(float)));
    
    // Run sequential
    float time_seq = run_sequential(d_in, d_out, h_in, h_out, N);
    printf("Sequential: %.2f ms\n", time_seq);
    
    // Run overlapped
    float time_overlap = run_overlapped(d_in, d_out, h_in, h_out, N, NUM_STREAMS);
    printf("Overlapped (%d streams): %.2f ms\n", NUM_STREAMS, time_overlap);
    printf("Speedup: %.2fx\n", time_seq / time_overlap);
    
    CHECK_CUDA(cudaFreeHost(h_in));
    CHECK_CUDA(cudaFreeHost(h_out));
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    
    return 0;
}
