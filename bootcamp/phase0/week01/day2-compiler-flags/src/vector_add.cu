/**
 * Day 2: Vector Add - For compiler flag experiments
 * 
 * This simple kernel is good for seeing optimization effects
 * because it's memory-bound and easy to understand.
 */

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Simple vector addition kernel
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

float benchmark(const float* d_a, const float* d_b, float* d_c, 
                int n, int iterations) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Timed runs
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return ms / iterations;  // ms per kernel
}

int main() {
    const int N = 1 << 24;  // 16M elements
    const int iterations = 100;
    
    printf("Vector Add Benchmark\n");
    printf("N = %d (%.1f MB per array)\n", N, N * sizeof(float) / 1e6);
    printf("Iterations: %d\n\n", iterations);
    
    // Allocate host memory
    float* h_a = (float*)malloc(N * sizeof(float));
    float* h_b = (float*)malloc(N * sizeof(float));
    float* h_c = (float*)malloc(N * sizeof(float));
    
    // Initialize
    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }
    
    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float)));
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));
    
    // Benchmark
    float ms = benchmark(d_a, d_b, d_c, N, iterations);
    
    // Calculate bandwidth
    // Read: 2 arrays, Write: 1 array = 3 * N * sizeof(float) bytes
    float bytes = 3.0f * N * sizeof(float);
    float gb_per_s = (bytes / 1e9) / (ms / 1e3);
    
    printf("Time per kernel: %.3f ms\n", ms);
    printf("Effective bandwidth: %.1f GB/s\n", gb_per_s);
    
    // Get theoretical peak for comparison
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    float peak_bw = 2.0f * props.memoryClockRate * (props.memoryBusWidth / 8) / 1e6;
    printf("Theoretical peak: %.0f GB/s\n", peak_bw);
    printf("Achieved: %.1f%% of peak\n", 100.0f * gb_per_s / peak_bw);
    
    // Verify correctness
    CUDA_CHECK(cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost));
    bool correct = true;
    for (int i = 0; i < N && correct; i++) {
        if (fabs(h_c[i] - 3.0f) > 1e-5) correct = false;
    }
    printf("\nCorrectness: %s\n", correct ? "PASS" : "FAIL");
    
    // Cleanup
    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    
    return 0;
}
