/**
 * Day 4: Benchmark Harness - Main driver
 * 
 * Demonstrates how to use the benchmark framework.
 */

#include <cstdio>
#include <cstdlib>
#include "cuda_timer.cuh"
#include "benchmark.cuh"

// External kernel declarations (from kernels.cu)
extern __global__ void vector_add_naive(const float*, const float*, float*, int);
extern __global__ void vector_add_vec4(const float4*, const float4*, float4*, int);
extern __global__ void reduce_naive(const float*, float*, int);
extern __global__ void reduce_sequential(const float*, float*, int);
extern __global__ void reduce_warp_shuffle(const float*, float*, int);

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ============================================================================
// Vector Add Benchmark
// ============================================================================

void benchmark_vector_add() {
    const int N = 1 << 24;  // 16M elements
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    const size_t bytes = 3ULL * N * sizeof(float);  // Read 2, write 1
    
    // Allocate
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float)));
    
    // Initialize with random data
    float* h_data = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) h_data[i] = (float)rand() / RAND_MAX;
    CUDA_CHECK(cudaMemcpy(d_a, h_data, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_data, N * sizeof(float), cudaMemcpyHostToDevice));
    free(h_data);
    
    clear_results();
    
    // Benchmark naive version
    benchmark_detailed("naive (float)", [&]() {
        vector_add_naive<<<blocks, threads>>>(d_a, d_b, d_c, N);
    }, bytes);
    
    // Benchmark vectorized version
    int N4 = N / 4;
    int blocks4 = (N4 + threads - 1) / threads;
    benchmark_detailed("vectorized (float4)", [&]() {
        vector_add_vec4<<<blocks4, threads>>>(
            (float4*)d_a, (float4*)d_b, (float4*)d_c, N4);
    }, bytes);
    
    print_results("Vector Add (16M elements)");
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

// ============================================================================
// Reduction Benchmark
// ============================================================================

void benchmark_reduction() {
    const int N = 1 << 24;  // 16M elements
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    const size_t bytes = N * sizeof(float);  // Read input once
    const size_t smem = threads * sizeof(float);
    
    // Allocate
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, blocks * sizeof(float)));
    
    // Initialize
    float* h_data = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) h_data[i] = 1.0f;  // Sum should be N
    CUDA_CHECK(cudaMemcpy(d_input, h_data, N * sizeof(float), cudaMemcpyHostToDevice));
    free(h_data);
    
    clear_results();
    
    // Benchmark different reduction strategies
    benchmark_detailed("naive (divergent)", [&]() {
        reduce_naive<<<blocks, threads, smem>>>(d_input, d_output, N);
    }, bytes);
    
    benchmark_detailed("sequential addr", [&]() {
        reduce_sequential<<<blocks, threads, smem>>>(d_input, d_output, N);
    }, bytes);
    
    benchmark_detailed("warp shuffle", [&]() {
        reduce_warp_shuffle<<<blocks, threads>>>(d_input, d_output, N);
    }, bytes);
    
    print_results("Reduction (16M elements, first pass)");
    
    cudaFree(d_input);
    cudaFree(d_output);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    // Print device info
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    printf("\n");
    printf("════════════════════════════════════════════════════════════════════\n");
    printf("  CUDA Benchmark Harness - Day 4\n");
    printf("════════════════════════════════════════════════════════════════════\n");
    printf("  Device: %s\n", props.name);
    printf("  Compute: %d.%d\n", props.major, props.minor);
    printf("  Peak BW: %.0f GB/s\n", 
           2.0 * props.memoryClockRate * (props.memoryBusWidth / 8) / 1e6);
    printf("════════════════════════════════════════════════════════════════════\n");
    
    benchmark_vector_add();
    benchmark_reduction();
    
    return 0;
}
