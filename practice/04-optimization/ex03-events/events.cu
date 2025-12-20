#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

__global__ void producer_kernel(float* data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        data[idx] = sinf((float)idx) * 100.0f;
    }
}

__global__ void consumer_kernel(float* out, const float* data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        out[idx] = data[idx] * 2.0f;
    }
}

// =============================================================================
// TODO 1: Accurate timing with CUDA events
// =============================================================================
void benchmark_with_events(float* d_out, float* d_in, int N, int iterations) {
    cudaEvent_t start, stop;
    
    // TODO: Create events
    // TODO: Record start event
    // TODO: Run kernel(s) multiple times
    // TODO: Record stop event
    // TODO: Synchronize and calculate time
    // TODO: Destroy events
    
    printf("Event timing: (implement)\n");
}

// =============================================================================
// TODO 2: Compare with CPU timing
// =============================================================================
void benchmark_with_cpu(float* d_out, float* d_in, int N, int iterations) {
    // Synchronize before timing
    CHECK_CUDA(cudaDeviceSynchronize());
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // TODO: Run kernel(s)
    
    CHECK_CUDA(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    printf("CPU timing: %.3f ms\n", duration.count() / 1000.0f);
}

// =============================================================================
// TODO 3: Producer-consumer with event synchronization
// =============================================================================
void producer_consumer_pattern(float* d_data, float* d_out, int N) {
    cudaStream_t producer_stream, consumer_stream;
    cudaEvent_t data_ready;
    
    CHECK_CUDA(cudaStreamCreate(&producer_stream));
    CHECK_CUDA(cudaStreamCreate(&consumer_stream));
    CHECK_CUDA(cudaEventCreate(&data_ready));
    
    int numBlocks = (N + 255) / 256;
    
    // TODO: Producer launches in producer_stream
    // producer_kernel<<<numBlocks, 256, 0, producer_stream>>>(d_data, N);
    
    // TODO: Record event after producer
    // cudaEventRecord(data_ready, producer_stream);
    
    // TODO: Consumer waits for data_ready before running
    // cudaStreamWaitEvent(consumer_stream, data_ready);
    // consumer_kernel<<<numBlocks, 256, 0, consumer_stream>>>(d_out, d_data, N);
    
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaStreamDestroy(producer_stream));
    CHECK_CUDA(cudaStreamDestroy(consumer_stream));
    CHECK_CUDA(cudaEventDestroy(data_ready));
    
    printf("Producer-consumer pattern: (implement)\n");
}

int main() {
    const int N = 1024 * 1024;
    const int iterations = 100;
    
    printf("CUDA Events Exercise\n");
    printf("====================\n\n");
    
    float *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, N * sizeof(float)));
    
    printf("Timing Comparison:\n");
    benchmark_with_events(d_out, d_in, N, iterations);
    benchmark_with_cpu(d_out, d_in, N, iterations);
    
    printf("\nProducer-Consumer:\n");
    producer_consumer_pattern(d_in, d_out, N);
    
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    
    return 0;
}
