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

// SOLUTION: Accurate timing with CUDA events
void benchmark_with_events(float* d_out, float* d_in, int N, int iterations) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    int numBlocks = (N + 255) / 256;
    
    // Warmup
    producer_kernel<<<numBlocks, 256>>>(d_in, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        producer_kernel<<<numBlocks, 256>>>(d_in, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    
    printf("Event timing: %.3f ms (avg %.3f ms per iteration)\n", ms, ms / iterations);
    
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

// SOLUTION: CPU timing comparison
void benchmark_with_cpu(float* d_out, float* d_in, int N, int iterations) {
    int numBlocks = (N + 255) / 256;
    
    // Warmup
    producer_kernel<<<numBlocks, 256>>>(d_in, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        producer_kernel<<<numBlocks, 256>>>(d_in, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    float ms = duration.count() / 1000.0f;
    
    printf("CPU timing:   %.3f ms (avg %.3f ms per iteration)\n", ms, ms / iterations);
}

// SOLUTION: Producer-consumer with event sync
void producer_consumer_pattern(float* d_data, float* d_out, int N) {
    cudaStream_t producer_stream, consumer_stream;
    cudaEvent_t data_ready, start, stop;
    
    CHECK_CUDA(cudaStreamCreate(&producer_stream));
    CHECK_CUDA(cudaStreamCreate(&consumer_stream));
    CHECK_CUDA(cudaEventCreate(&data_ready));
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    int numBlocks = (N + 255) / 256;
    
    CHECK_CUDA(cudaEventRecord(start));
    
    // Producer runs in producer_stream
    producer_kernel<<<numBlocks, 256, 0, producer_stream>>>(d_data, N);
    
    // Record event when producer is done
    CHECK_CUDA(cudaEventRecord(data_ready, producer_stream));
    
    // Consumer waits for producer
    CHECK_CUDA(cudaStreamWaitEvent(consumer_stream, data_ready));
    consumer_kernel<<<numBlocks, 256, 0, consumer_stream>>>(d_out, d_data, N);
    
    CHECK_CUDA(cudaEventRecord(stop, consumer_stream));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("Producer-consumer completed in %.3f ms\n", ms);
    
    // Verify
    float* h_out = (float*)malloc(N * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    float expected = sinf(0.0f) * 100.0f * 2.0f;
    printf("Verification: d_out[0] = %.3f (expected %.3f) [%s]\n", 
           h_out[0], expected, (fabs(h_out[0] - expected) < 0.01) ? "PASS" : "FAIL");
    free(h_out);
    
    CHECK_CUDA(cudaStreamDestroy(producer_stream));
    CHECK_CUDA(cudaStreamDestroy(consumer_stream));
    CHECK_CUDA(cudaEventDestroy(data_ready));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

int main() {
    const int N = 1024 * 1024;
    const int iterations = 100;
    
    printf("CUDA Events - SOLUTION\n");
    printf("======================\n\n");
    
    float *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, N * sizeof(float)));
    
    printf("Timing Comparison (%d iterations):\n", iterations);
    benchmark_with_events(d_out, d_in, N, iterations);
    benchmark_with_cpu(d_out, d_in, N, iterations);
    
    printf("\nProducer-Consumer Pattern:\n");
    producer_consumer_pattern(d_in, d_out, N);
    
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    
    printf("\n✓ Solution complete!\n");
    return 0;
}
