/*
 * Day 2: Batched Inference
 * 
 * Dynamic batching and batch size optimization.
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <vector>
#include <algorithm>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CHECK_CUBLAS(call) \
    do { \
        cublasStatus_t err = call; \
        if (err != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d\n", __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Simple linear layer forward
void linearForward(cublasHandle_t handle,
                   const float* input, const float* weights, float* output,
                   int batchSize, int inputSize, int outputSize) {
    const float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              outputSize, batchSize, inputSize,
                              &alpha, weights, outputSize,
                              input, inputSize,
                              &beta, output, outputSize));
}

// Benchmark different batch sizes
void benchmarkBatchSize(cublasHandle_t handle,
                        float* d_input, float* d_weights, float* d_output,
                        int inputSize, int outputSize,
                        int batchSize, int iterations) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    // Warmup
    linearForward(handle, d_input, d_weights, d_output, batchSize, inputSize, outputSize);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        linearForward(handle, d_input, d_weights, d_output, batchSize, inputSize, outputSize);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    float msPerBatch = ms / iterations;
    float samplesPerSec = batchSize * 1000.0f / msPerBatch;
    float tflops = 2.0f * batchSize * inputSize * outputSize / (msPerBatch * 1e9f);
    
    printf("%8d %12.3f %15.0f %12.2f\n", 
           batchSize, msPerBatch, samplesPerSec, tflops);
    
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

int main() {
    printf("=== Day 2: Batched Inference ===\n\n");
    
    const int inputSize = 1024;
    const int outputSize = 1024;
    const int maxBatchSize = 512;
    const int iterations = 100;
    
    printf("Layer: %d -> %d\n\n", inputSize, outputSize);
    
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    
    // Allocate for max batch size
    float *d_input, *d_weights, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, maxBatchSize * inputSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_weights, inputSize * outputSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, maxBatchSize * outputSize * sizeof(float)));
    
    // Initialize
    std::vector<float> h_weights(inputSize * outputSize);
    for (auto& w : h_weights) w = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    CHECK_CUDA(cudaMemcpy(d_weights, h_weights.data(), h_weights.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    
    // Benchmark different batch sizes
    printf("--- Batch Size vs Performance ---\n");
    printf("%8s %12s %15s %12s\n", "Batch", "Latency(ms)", "Throughput", "TFLOPS");
    printf("%8s %12s %15s %12s\n", "-----", "-----------", "----------", "------");
    
    std::vector<int> batchSizes = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
    for (int bs : batchSizes) {
        benchmarkBatchSize(handle, d_input, d_weights, d_output,
                          inputSize, outputSize, bs, iterations);
    }
    
    printf("\n--- Key Observations ---\n");
    printf("- Latency: ~constant for small batches (launch overhead)\n");
    printf("- Throughput: Increases with batch size (better GPU utilization)\n");
    printf("- TFLOPS: Plateaus at large batches (compute-bound)\n");
    printf("\nOptimal batch size depends on latency requirements.\n");
    
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_weights));
    CHECK_CUDA(cudaFree(d_output));
    
    printf("\n=== Day 2 Complete ===\n");
    return 0;
}
