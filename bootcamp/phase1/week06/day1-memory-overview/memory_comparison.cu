/**
 * memory_comparison.cu - Benchmark different memory types
 * 
 * Learning objectives:
 * - Compare memory access speeds
 * - See real performance differences
 */

#include <cuda_runtime.h>
#include <cstdio>

__constant__ float d_const_data[1024];

// Global memory read benchmark
__global__ void benchmark_global(const float* data, float* output, int n, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    float sum = 0.0f;
    for (int iter = 0; iter < iterations; iter++) {
        for (int i = idx; i < n; i += stride) {
            sum += data[i];
        }
    }
    if (idx == 0) output[0] = sum;
}

// Shared memory read benchmark
__global__ void benchmark_shared(const float* data, float* output, int n, int iterations) {
    __shared__ float smem[1024];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load to shared once
    if (idx < n) smem[tid] = data[idx];
    __syncthreads();
    
    // Read from shared many times
    float sum = 0.0f;
    for (int iter = 0; iter < iterations; iter++) {
        if (tid < n) {
            sum += smem[tid];
        }
    }
    if (idx == 0) output[0] = sum;
}

// Constant memory read benchmark
__global__ void benchmark_constant(float* output, int n, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    for (int iter = 0; iter < iterations; iter++) {
        // All threads read same location - broadcast
        sum += d_const_data[0];
    }
    if (idx == 0) output[0] = sum;
}

// Register read benchmark
__global__ void benchmark_register(float* output, int n, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float reg = (float)idx;  // In register
    float sum = 0.0f;
    
    for (int iter = 0; iter < iterations; iter++) {
        sum += reg;
    }
    if (idx == 0) output[0] = sum;
}

int main() {
    printf("=== Memory Bandwidth/Latency Comparison ===\n\n");
    
    const int N = 1024;
    const int ITERATIONS = 10000;
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = 64;
    
    float* d_data, *d_output;
    float* h_data = new float[N];
    
    for (int i = 0; i < N; i++) h_data[i] = 1.0f;
    
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_const_data, h_data, N * sizeof(float));
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;
    
    // Warmup
    benchmark_global<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_data, d_output, N, 10);
    cudaDeviceSynchronize();
    
    // Benchmark global memory
    cudaEventRecord(start);
    benchmark_global<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_data, d_output, N, ITERATIONS);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("Global memory:   %8.3f ms\n", ms);
    float global_time = ms;
    
    // Benchmark shared memory
    cudaEventRecord(start);
    benchmark_shared<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_data, d_output, N, ITERATIONS);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("Shared memory:   %8.3f ms (%.1fx faster)\n", ms, global_time / ms);
    
    // Benchmark constant memory
    cudaEventRecord(start);
    benchmark_constant<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_output, N, ITERATIONS);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("Constant memory: %8.3f ms (%.1fx faster, broadcast)\n", ms, global_time / ms);
    
    // Benchmark registers
    cudaEventRecord(start);
    benchmark_register<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_output, N, ITERATIONS);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("Registers:       %8.3f ms (%.1fx faster)\n", ms, global_time / ms);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    cudaFree(d_output);
    delete[] h_data;
    
    printf("\n=== Key Insight ===\n");
    printf("Register > Constant (broadcast) > Shared >> Global\n");
    printf("Use the fastest memory type appropriate for your access pattern.\n");
    
    return 0;
}
