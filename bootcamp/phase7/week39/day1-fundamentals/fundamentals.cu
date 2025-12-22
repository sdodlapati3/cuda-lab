/**
 * Week 39, Day 1: Kernel Fusion Fundamentals
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// Unfused: 3 separate kernels
__global__ void addBiasKernel(float* x, const float* bias, int n, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * d) {
        int col = idx % d;
        x[idx] += bias[col];
    }
}

__global__ void geluKernel(float* x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = x[idx];
        x[idx] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
    }
}

__global__ void dropoutKernel(float* x, float* mask, int n, float p) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] = mask[idx] > p ? x[idx] / (1.0f - p) : 0.0f;
    }
}

// Fused: single kernel
__global__ void fusedBiasGeluDropout(
    float* x, const float* bias, const float* rand, 
    int n, int d, float p
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * d) {
        int col = idx % d;
        
        // Bias
        float v = x[idx] + bias[col];
        
        // GELU
        v = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
        
        // Dropout
        x[idx] = rand[idx] > p ? v / (1.0f - p) : 0.0f;
    }
}

int main() {
    printf("Week 39 Day 1: Kernel Fusion Fundamentals\n\n");
    
    printf("Why Fuse Kernels?\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║ Problem: Each kernel launch has overhead                          ║\n");
    printf("║   • Launch latency (~5-10 µs per kernel)                          ║\n");
    printf("║   • Memory read/write between kernels                             ║\n");
    printf("║   • Cache invalidation                                            ║\n");
    printf("║                                                                   ║\n");
    printf("║ Solution: Fuse element-wise operations                            ║\n");
    printf("║   • Single kernel launch                                          ║\n");
    printf("║   • Data stays in registers/L1                                    ║\n");
    printf("║   • Reduced memory traffic                                        ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");
    
    const int n = 4096, d = 1024;
    const float dropout_p = 0.1f;
    
    float *d_x, *d_bias, *d_rand;
    cudaMalloc(&d_x, n * d * sizeof(float));
    cudaMalloc(&d_bias, d * sizeof(float));
    cudaMalloc(&d_rand, n * d * sizeof(float));
    
    dim3 block(256);
    dim3 grid((n * d + 255) / 256);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;
    
    // Benchmark unfused
    cudaEventRecord(start);
    for (int i = 0; i < 1000; i++) {
        addBiasKernel<<<grid, block>>>(d_x, d_bias, n, d);
        geluKernel<<<grid, block>>>(d_x, n * d);
        dropoutKernel<<<grid, block>>>(d_x, d_rand, n * d, dropout_p);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("Unfused (3 kernels): %.2f ms / 1000 iters\n", ms);
    
    // Benchmark fused
    cudaEventRecord(start);
    for (int i = 0; i < 1000; i++) {
        fusedBiasGeluDropout<<<grid, block>>>(d_x, d_bias, d_rand, n, d, dropout_p);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("Fused (1 kernel): %.2f ms / 1000 iters\n\n", ms);
    
    printf("Memory Traffic Analysis:\n");
    printf("┌─────────────────────┬───────────────┬───────────────┐\n");
    printf("│ Operation           │ Unfused       │ Fused         │\n");
    printf("├─────────────────────┼───────────────┼───────────────┤\n");
    printf("│ Read x              │ 3× N          │ 1× N          │\n");
    printf("│ Write x             │ 3× N          │ 1× N          │\n");
    printf("│ Total               │ 6× N          │ 2× N          │\n");
    printf("│ Reduction           │ -             │ 3×            │\n");
    printf("└─────────────────────┴───────────────┴───────────────┘\n");
    
    cudaFree(d_x); cudaFree(d_bias); cudaFree(d_rand);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    
    return 0;
}
