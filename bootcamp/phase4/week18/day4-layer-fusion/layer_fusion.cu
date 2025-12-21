/*
 * Day 4: Layer Fusion
 * 
 * Fusing Linear + Bias + ReLU for inference optimization.
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>

#define CHECK_CUDA(call) do { cudaError_t e = call; if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); exit(1); } } while(0)
#define CHECK_CUBLAS(call) do { cublasStatus_t s = call; if (s != CUBLAS_STATUS_SUCCESS) { printf("cuBLAS error\n"); exit(1); } } while(0)

// Separate kernels
__global__ void add_bias(float* x, const float* bias, int n, int features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) x[idx] += bias[idx % features];
}

__global__ void relu(float* x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) x[idx] = fmaxf(0.0f, x[idx]);
}

// Fused bias + relu
__global__ void fused_bias_relu(float* x, const float* bias, int n, int features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] = fmaxf(0.0f, x[idx] + bias[idx % features]);
    }
}

int main() {
    printf("=== Day 4: Layer Fusion ===\n\n");
    
    const int batchSize = 64;
    const int features = 1024;
    const int n = batchSize * features;
    const int iterations = 1000;
    
    float *d_x, *d_bias;
    CHECK_CUDA(cudaMalloc(&d_x, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_bias, features * sizeof(float)));
    
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    // Separate kernels
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        add_bias<<<grid, block>>>(d_x, d_bias, n, features);
        relu<<<grid, block>>>(d_x, n);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms_separate;
    CHECK_CUDA(cudaEventElapsedTime(&ms_separate, start, stop));
    
    // Fused kernel
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        fused_bias_relu<<<grid, block>>>(d_x, d_bias, n, features);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms_fused;
    CHECK_CUDA(cudaEventElapsedTime(&ms_fused, start, stop));
    
    printf("Separate kernels: %.3f ms\n", ms_separate / iterations);
    printf("Fused kernel:     %.3f ms (%.2fx faster)\n", 
           ms_fused / iterations, ms_separate / ms_fused);
    
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_bias));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    printf("\n=== Day 4 Complete ===\n");
    return 0;
}
