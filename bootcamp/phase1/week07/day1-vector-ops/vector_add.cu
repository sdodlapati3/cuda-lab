/**
 * vector_add.cu - The "Hello World" of CUDA
 * 
 * Learning objectives:
 * - Implement simplest useful kernel
 * - Measure memory bandwidth
 * - Establish baseline performance
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// Basic vector add kernel
__global__ void vector_add(const float* A, const float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

// Grid-stride loop version
__global__ void vector_add_stride(const float* A, const float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        C[i] = A[i] + B[i];
    }
}

void verify(const float* A, const float* B, const float* C, int n) {
    for (int i = 0; i < n; i++) {
        float expected = A[i] + B[i];
        if (fabsf(C[i] - expected) > 1e-5) {
            printf("Verification FAILED at %d: %.6f != %.6f\n", i, C[i], expected);
            return;
        }
    }
    printf("Verification PASSED\n");
}

int main() {
    printf("=== Vector Addition ===\n\n");
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    
    float peak_bandwidth = prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8) * 2 / 1e9;
    printf("Theoretical peak bandwidth: %.0f GB/s\n\n", peak_bandwidth);
    
    // Test different sizes
    int sizes[] = {1 << 16, 1 << 18, 1 << 20, 1 << 22, 1 << 24, 1 << 26};
    
    printf("%-15s %-12s %-15s %-12s\n", "N", "Time (ms)", "Bandwidth (GB/s)", "Efficiency");
    printf("------------------------------------------------------------\n");
    
    for (int n : sizes) {
        size_t bytes = n * sizeof(float);
        
        // Allocate
        float *h_A = new float[n];
        float *h_B = new float[n];
        float *h_C = new float[n];
        
        for (int i = 0; i < n; i++) {
            h_A[i] = rand() / (float)RAND_MAX;
            h_B[i] = rand() / (float)RAND_MAX;
        }
        
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, bytes);
        cudaMalloc(&d_B, bytes);
        cudaMalloc(&d_C, bytes);
        
        cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
        
        int block_size = 256;
        int num_blocks = (n + block_size - 1) / block_size;
        
        // Warmup
        vector_add<<<num_blocks, block_size>>>(d_A, d_B, d_C, n);
        cudaDeviceSynchronize();
        
        // Benchmark
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        const int TRIALS = 100;
        cudaEventRecord(start);
        for (int t = 0; t < TRIALS; t++) {
            vector_add<<<num_blocks, block_size>>>(d_A, d_B, d_C, n);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        ms /= TRIALS;
        
        // Calculate bandwidth: 2 reads + 1 write = 3 * N * sizeof(float)
        float bandwidth = 3.0f * bytes / ms / 1e6;  // GB/s
        float efficiency = 100.0f * bandwidth / peak_bandwidth;
        
        printf("%-15d %-12.3f %-15.1f %-12.1f%%\n", n, ms, bandwidth, efficiency);
        
        // Verify (once)
        if (n == sizes[0]) {
            cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
            verify(h_A, h_B, h_C, n);
        }
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        delete[] h_A;
        delete[] h_B;
        delete[] h_C;
    }
    
    printf("\n=== Key Insights ===\n");
    printf("1. Vector add is memory-bound (1 FLOP per 12 bytes)\n");
    printf("2. Bandwidth should approach peak for large arrays\n");
    printf("3. Small arrays don't saturate the GPU\n");
    printf("4. This is the baseline for all memory-bound kernels\n");
    
    return 0;
}
