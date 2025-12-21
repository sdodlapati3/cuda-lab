/**
 * constant_memory.cu - Demonstrate constant memory usage
 * 
 * Learning objectives:
 * - Use constant memory for broadcast data
 * - See performance characteristics
 * - Know when to use vs avoid
 */

#include <cuda_runtime.h>
#include <cstdio>

#define FILTER_SIZE 9

// Constant memory for filter coefficients
__constant__ float d_filter[FILTER_SIZE];

// Global memory for comparison
__device__ float d_filter_global[FILTER_SIZE];

// Convolution using constant memory (good: all threads read same coefficients)
__global__ void convolve_constant(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n - FILTER_SIZE + 1) return;
    
    float sum = 0.0f;
    for (int i = 0; i < FILTER_SIZE; i++) {
        sum += input[idx + i] * d_filter[i];  // Broadcast access pattern
    }
    output[idx] = sum;
}

// Convolution using global memory (for comparison)
__global__ void convolve_global(const float* input, float* output, 
                                 const float* filter, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n - FILTER_SIZE + 1) return;
    
    float sum = 0.0f;
    for (int i = 0; i < FILTER_SIZE; i++) {
        sum += input[idx + i] * filter[i];  // Global memory access
    }
    output[idx] = sum;
}

// Bad use: scattered access (serialized!)
__global__ void bad_constant_access(float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // BAD: Each thread reads different constant memory location
    // This serializes access and is slow!
    output[idx] = d_filter[idx % FILTER_SIZE];
}

// Good use: broadcast access
__global__ void good_constant_access(float* output, int n, int coeff_idx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // GOOD: All threads read same location → broadcast
    output[idx] = output[idx] * d_filter[coeff_idx];
}

int main() {
    printf("=== Constant Memory Demo ===\n\n");
    
    printf("Constant memory: 64 KB, cached, broadcast-optimized\n\n");
    
    // Setup filter
    float h_filter[FILTER_SIZE];
    for (int i = 0; i < FILTER_SIZE; i++) {
        h_filter[i] = 1.0f / FILTER_SIZE;  // Simple averaging filter
    }
    
    // Copy to constant memory
    cudaMemcpyToSymbol(d_filter, h_filter, sizeof(h_filter));
    
    // Also copy to global memory for comparison
    float* d_filter_ptr;
    cudaMalloc(&d_filter_ptr, sizeof(h_filter));
    cudaMemcpy(d_filter_ptr, h_filter, sizeof(h_filter), cudaMemcpyHostToDevice);
    
    const int N = 1 << 20;
    float* d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    
    // Initialize
    float* h_input = new float[N];
    for (int i = 0; i < N; i++) h_input[i] = (float)i;
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    int block_size = 256;
    int num_blocks = (N + block_size - 1) / block_size;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;
    
    printf("=== Convolution Benchmark ===\n\n");
    
    // Warmup
    convolve_constant<<<num_blocks, block_size>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    // Constant memory convolution
    cudaEventRecord(start);
    for (int t = 0; t < 100; t++) {
        convolve_constant<<<num_blocks, block_size>>>(d_input, d_output, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("Constant memory: %.3f ms\n", ms / 100);
    float const_time = ms / 100;
    
    // Global memory convolution
    cudaEventRecord(start);
    for (int t = 0; t < 100; t++) {
        convolve_global<<<num_blocks, block_size>>>(d_input, d_output, d_filter_ptr, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("Global memory:   %.3f ms (%.2fx)\n", ms / 100, (ms / 100) / const_time);
    
    // Verify
    float* h_output = new float[N];
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("\nResult sample: output[100] = %.6f (expected: %.6f)\n", 
           h_output[100], (100 + 101 + 102 + 103 + 104 + 105 + 106 + 107 + 108) / 9.0f);
    
    printf("\n=== Access Pattern Comparison ===\n\n");
    
    // Bad scattered access
    cudaEventRecord(start);
    for (int t = 0; t < 100; t++) {
        bad_constant_access<<<num_blocks, block_size>>>(d_output, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("Scattered access (bad): %.3f ms\n", ms / 100);
    
    // Good broadcast access
    cudaEventRecord(start);
    for (int t = 0; t < 100; t++) {
        good_constant_access<<<num_blocks, block_size>>>(d_output, N, 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("Broadcast access (good): %.3f ms\n", ms / 100);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_filter_ptr);
    delete[] h_input;
    delete[] h_output;
    
    printf("\n=== Constant Memory Guidelines ===\n");
    printf("✓ Use for: Small, read-only data with broadcast access\n");
    printf("✓ Good for: Filter coefficients, lookup tables, constants\n");
    printf("✗ Avoid: Scattered access (different threads → different addresses)\n");
    printf("✗ Limited: 64 KB total\n");
    
    return 0;
}
