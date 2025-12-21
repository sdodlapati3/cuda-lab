/**
 * Week 30, Day 2: Custom Forward Pass
 * Example: Fused GELU activation.
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
__global__ void geluForwardKernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        output[idx] = x * cdf;
    }
}

// Faster approximate GELU using sigmoid
__global__ void geluFastKernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        output[idx] = x * (1.0f / (1.0f + expf(-1.702f * x)));
    }
}

/*
 * PyTorch Integration (pseudocode):
 *
 * torch::Tensor gelu_forward_cuda(torch::Tensor input) {
 *     auto output = torch::empty_like(input);
 *     int n = input.numel();
 *     int threads = 256;
 *     int blocks = (n + threads - 1) / threads;
 *     geluForwardKernel<<<blocks, threads>>>(
 *         input.data_ptr<float>(),
 *         output.data_ptr<float>(),
 *         n
 *     );
 *     return output;
 * }
 */

int main() {
    printf("Week 30 Day 2: Custom Forward Pass\n\n");
    
    const int N = 1 << 20;  // 1M elements
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    
    // Initialize with test data
    float* h_input = new float[N];
    for (int i = 0; i < N; i++) {
        h_input[i] = (i % 100) / 50.0f - 1.0f;  // [-1, 1]
    }
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    int iterations = 100;
    
    // Warmup
    geluForwardKernel<<<blocks, threads>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    // Exact GELU
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        geluForwardKernel<<<blocks, threads>>>(d_input, d_output, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float exactMs;
    cudaEventElapsedTime(&exactMs, start, stop);
    exactMs /= iterations;
    
    // Fast GELU
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        geluFastKernel<<<blocks, threads>>>(d_input, d_output, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float fastMs;
    cudaEventElapsedTime(&fastMs, start, stop);
    fastMs /= iterations;
    
    printf("GELU Forward Performance (%d elements):\n", N);
    printf("  Exact GELU (tanh): %.3f ms (%.2f GB/s)\n", 
           exactMs, 2.0f * N * sizeof(float) / exactMs / 1e6);
    printf("  Fast GELU (sigmoid): %.3f ms (%.2f GB/s)\n",
           fastMs, 2.0f * N * sizeof(float) / fastMs / 1e6);
    printf("  Speedup: %.2fx\n", exactMs / fastMs);
    
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    
    return 0;
}
