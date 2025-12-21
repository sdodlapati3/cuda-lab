/**
 * Example Kernels - Position on Roofline
 * 
 * Different kernels with known arithmetic intensity.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

// Structure to hold kernel measurement
struct KernelPoint {
    const char* name;
    float ai;           // Arithmetic Intensity (FLOP/Byte)
    float achieved_gflops;
    float theoretical_gflops;  // Based on roofline
};

// Vector add: AI = 1 FLOP / 12 Bytes = 0.083
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];  // 1 FLOP, 12 bytes
    }
}

// SAXPY: AI = 2 FLOPS / 12 Bytes = 0.167
__global__ void saxpy(float alpha, const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = alpha * x[idx] + y[idx];  // 2 FLOPs, 12 bytes
    }
}

// Dot product: AI = 2 FLOPS / 8 Bytes = 0.25 (naive)
__global__ void dot_product(const float* a, const float* b, float* result, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    if (idx < n) {
        sum = a[idx] * b[idx];  // 2 FLOPs
    }
    sdata[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    
    if (tid == 0) atomicAdd(result, sdata[0]);
}

// Polynomial evaluation: AI = 10 FLOPS / 8 Bytes = 1.25
__global__ void poly_eval(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx];
        // 5th degree polynomial: 10 FLOPs (5 mul + 5 add via Horner)
        float result = 1.0f;
        result = result * val + 2.0f;
        result = result * val + 3.0f;
        result = result * val + 4.0f;
        result = result * val + 5.0f;
        result = result * val + 6.0f;
        y[idx] = result;
    }
}

std::vector<KernelPoint> measure_example_kernels(float peak_bw, float peak_gflops) {
    const int N = 1 << 22;  // 4M elements
    const int THREADS = 256;
    const int BLOCKS = (N + THREADS - 1) / THREADS;
    const int WARMUP = 10;
    const int ITERATIONS = 100;
    
    printf("\n=== Example Kernels ===\n");
    
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    std::vector<KernelPoint> points;
    
    // Measure vector_add
    {
        float ai = 1.0f / 12.0f;  // 1 FLOP / 12 bytes
        size_t flops = (size_t)N * 1;
        
        for (int i = 0; i < WARMUP; i++) 
            vector_add<<<BLOCKS, THREADS>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
        
        cudaEventRecord(start);
        for (int i = 0; i < ITERATIONS; i++)
            vector_add<<<BLOCKS, THREADS>>>(d_a, d_b, d_c, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms; cudaEventElapsedTime(&ms, start, stop);
        float achieved = (flops * ITERATIONS / 1e9) / (ms / 1e3);
        float theoretical = fmin(peak_gflops, peak_bw * ai);
        
        points.push_back({"vector_add", ai, achieved, theoretical});
        printf("vector_add:   AI=%.3f  Achieved=%.2f GFLOPS  Theoretical=%.2f GFLOPS\n", 
               ai, achieved, theoretical);
    }
    
    // Measure saxpy
    {
        float ai = 2.0f / 12.0f;
        size_t flops = (size_t)N * 2;
        
        for (int i = 0; i < WARMUP; i++)
            saxpy<<<BLOCKS, THREADS>>>(2.0f, d_a, d_b, N);
        cudaDeviceSynchronize();
        
        cudaEventRecord(start);
        for (int i = 0; i < ITERATIONS; i++)
            saxpy<<<BLOCKS, THREADS>>>(2.0f, d_a, d_b, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms; cudaEventElapsedTime(&ms, start, stop);
        float achieved = (flops * ITERATIONS / 1e9) / (ms / 1e3);
        float theoretical = fmin(peak_gflops, peak_bw * ai);
        
        points.push_back({"saxpy", ai, achieved, theoretical});
        printf("saxpy:        AI=%.3f  Achieved=%.2f GFLOPS  Theoretical=%.2f GFLOPS\n",
               ai, achieved, theoretical);
    }
    
    // Measure poly_eval
    {
        float ai = 10.0f / 8.0f;  // 10 FLOPs / 8 bytes
        size_t flops = (size_t)N * 10;
        
        for (int i = 0; i < WARMUP; i++)
            poly_eval<<<BLOCKS, THREADS>>>(d_a, d_b, N);
        cudaDeviceSynchronize();
        
        cudaEventRecord(start);
        for (int i = 0; i < ITERATIONS; i++)
            poly_eval<<<BLOCKS, THREADS>>>(d_a, d_b, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms; cudaEventElapsedTime(&ms, start, stop);
        float achieved = (flops * ITERATIONS / 1e9) / (ms / 1e3);
        float theoretical = fmin(peak_gflops, peak_bw * ai);
        
        points.push_back({"poly_eval", ai, achieved, theoretical});
        printf("poly_eval:    AI=%.3f  Achieved=%.2f GFLOPS  Theoretical=%.2f GFLOPS\n",
               ai, achieved, theoretical);
    }
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return points;
}
