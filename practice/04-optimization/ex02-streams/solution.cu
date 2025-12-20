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
#define BLOCK_SIZE 256

__global__ void process_kernel(float* out, const float* in, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        float val = in[idx];
        for (int i = 0; i < 100; i++) {
            val = sinf(val) * cosf(val);
        }
        out[idx] = val;
    }
}

// SOLUTION: Sequential version
float run_sequential(float* d_in, float* d_out, float* h_in, float* h_out, int N) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    CHECK_CUDA(cudaEventRecord(start));
    
    CHECK_CUDA(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));
    process_kernel<<<numBlocks, BLOCK_SIZE>>>(d_out, d_in, N);
    CHECK_CUDA(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    return ms;
}

// SOLUTION: Overlapped version
float run_overlapped(float* d_in, float* d_out, float* h_in, float* h_out, 
                     int N, int numStreams) {
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < numStreams; i++) {
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
    }
    
    int chunkSize = N / numStreams;
    int numBlocks = (chunkSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    CHECK_CUDA(cudaEventRecord(start));
    
    for (int i = 0; i < numStreams; i++) {
        int offset = i * chunkSize;
        
        // H2D for this chunk
        CHECK_CUDA(cudaMemcpyAsync(d_in + offset, h_in + offset, 
                                   chunkSize * sizeof(float),
                                   cudaMemcpyHostToDevice, streams[i]));
        
        // Kernel for this chunk
        process_kernel<<<numBlocks, BLOCK_SIZE, 0, streams[i]>>>(
            d_out + offset, d_in + offset, chunkSize);
        
        // D2H for this chunk
        CHECK_CUDA(cudaMemcpyAsync(h_out + offset, d_out + offset,
                                   chunkSize * sizeof(float),
                                   cudaMemcpyDeviceToHost, streams[i]));
    }
    
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
    const int N = 4 * 1024 * 1024;
    
    printf("CUDA Streams - SOLUTION\n");
    printf("=======================\n");
    printf("Array size: %d elements\n\n", N);
    
    float *h_in, *h_out;
    CHECK_CUDA(cudaMallocHost(&h_in, N * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&h_out, N * sizeof(float)));
    
    for (int i = 0; i < N; i++) h_in[i] = (float)i;
    
    float *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, N * sizeof(float)));
    
    // Warmup
    run_sequential(d_in, d_out, h_in, h_out, N);
    
    // Benchmark
    float time_seq = run_sequential(d_in, d_out, h_in, h_out, N);
    printf("Sequential: %.2f ms\n", time_seq);
    
    for (int ns = 2; ns <= NUM_STREAMS; ns++) {
        float time_overlap = run_overlapped(d_in, d_out, h_in, h_out, N, ns);
        printf("Overlapped (%d streams): %.2f ms (%.2fx speedup)\n", 
               ns, time_overlap, time_seq / time_overlap);
    }
    
    CHECK_CUDA(cudaFreeHost(h_in));
    CHECK_CUDA(cudaFreeHost(h_out));
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    
    printf("\n✓ Solution complete!\n");
    return 0;
}
