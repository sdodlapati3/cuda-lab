/**
 * fusion_basics.cu - Why and how to fuse kernels
 * 
 * Learning objectives:
 * - Measure kernel launch overhead
 * - Compare fused vs separate kernels
 * - Understand memory traffic reduction
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// ============================================================================
// Separate Kernels (Unfused)
// ============================================================================

__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void mul_kernel(const float* c, float scalar, float* d, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d[idx] = c[idx] * scalar;
    }
}

__global__ void sqrt_kernel(const float* d, float* e, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        e[idx] = sqrtf(d[idx]);
    }
}

// ============================================================================
// Fused Kernel
// ============================================================================

__global__ void fused_kernel(const float* a, const float* b, 
                              float scalar, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float c = a[idx] + b[idx];   // In register
        float d = c * scalar;         // In register
        out[idx] = sqrtf(d);          // Only one write
    }
}

// ============================================================================
// Benchmark: Kernel Launch Overhead
// ============================================================================

__global__ void empty_kernel() {
    // Does nothing - measures pure launch overhead
}

void measure_launch_overhead() {
    printf("1. Kernel Launch Overhead\n");
    printf("─────────────────────────────────────────\n");
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    const int NUM_LAUNCHES = 1000;
    
    // Warmup
    empty_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < NUM_LAUNCHES; i++) {
        empty_kernel<<<1, 1>>>();
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    float overhead_us = ms * 1000.0f / NUM_LAUNCHES;
    
    printf("   %d launches: %.2f ms total\n", NUM_LAUNCHES, ms);
    printf("   Per-launch overhead: %.2f μs\n\n", overhead_us);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    printf("=== Kernel Fusion Basics ===\n\n");
    
    // First, measure launch overhead
    measure_launch_overhead();
    
    const int N = 1 << 24;  // 16M elements
    const size_t bytes = N * sizeof(float);
    const float scalar = 2.0f;
    
    printf("2. Fused vs Unfused Comparison\n");
    printf("─────────────────────────────────────────\n");
    printf("   %d elements (%.0f MB)\n\n", N, bytes / 1e6);
    
    // Allocate
    float *d_a, *d_b, *d_c, *d_d, *d_e, *d_out;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);  // Intermediate for unfused
    cudaMalloc(&d_d, bytes);  // Intermediate for unfused
    cudaMalloc(&d_e, bytes);  // Output for unfused
    cudaMalloc(&d_out, bytes);  // Output for fused
    
    // Initialize
    float* h_data = new float[N];
    for (int i = 0; i < N; i++) h_data[i] = 1.0f;
    cudaMemcpy(d_a, h_data, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_data, bytes, cudaMemcpyHostToDevice);
    delete[] h_data;
    
    int block_size = 256;
    int num_blocks = (N + block_size - 1) / block_size;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // ========================================================================
    // Unfused Version
    // ========================================================================
    
    // Warmup
    add_kernel<<<num_blocks, block_size>>>(d_a, d_b, d_c, N);
    mul_kernel<<<num_blocks, block_size>>>(d_c, scalar, d_d, N);
    sqrt_kernel<<<num_blocks, block_size>>>(d_d, d_e, N);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        add_kernel<<<num_blocks, block_size>>>(d_a, d_b, d_c, N);
        mul_kernel<<<num_blocks, block_size>>>(d_c, scalar, d_d, N);
        sqrt_kernel<<<num_blocks, block_size>>>(d_d, d_e, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float unfused_ms;
    cudaEventElapsedTime(&unfused_ms, start, stop);
    unfused_ms /= 100;
    
    // ========================================================================
    // Fused Version
    // ========================================================================
    
    // Warmup
    fused_kernel<<<num_blocks, block_size>>>(d_a, d_b, scalar, d_out, N);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        fused_kernel<<<num_blocks, block_size>>>(d_a, d_b, scalar, d_out, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float fused_ms;
    cudaEventElapsedTime(&fused_ms, start, stop);
    fused_ms /= 100;
    
    // ========================================================================
    // Analysis
    // ========================================================================
    
    // Verify correctness
    float h_unfused, h_fused;
    cudaMemcpy(&h_unfused, d_e, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_fused, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    
    float expected = sqrtf((1.0f + 1.0f) * scalar);
    bool correct = (fabsf(h_unfused - expected) < 1e-5) && 
                   (fabsf(h_fused - expected) < 1e-5);
    
    // Memory traffic analysis
    // Unfused: read a,b; write c; read c; write d; read d; write e
    //          = 6 * N * sizeof(float)
    // Fused:   read a,b; write out = 3 * N * sizeof(float)
    float unfused_bytes = 6.0f * N * sizeof(float);
    float fused_bytes = 3.0f * N * sizeof(float);
    
    float unfused_bw = unfused_bytes / (unfused_ms / 1000) / 1e9;
    float fused_bw = fused_bytes / (fused_ms / 1000) / 1e9;
    float effective_bw = unfused_bytes / (fused_ms / 1000) / 1e9;
    
    printf("   ┌────────────────┬────────────┬────────────┐\n");
    printf("   │ Metric         │ Unfused    │ Fused      │\n");
    printf("   ├────────────────┼────────────┼────────────┤\n");
    printf("   │ Kernel launches│ 3          │ 1          │\n");
    printf("   │ Memory traffic │ %.0f MB    │ %.0f MB    │\n", 
           unfused_bytes/1e6, fused_bytes/1e6);
    printf("   │ Time           │ %.3f ms   │ %.3f ms   │\n",
           unfused_ms, fused_ms);
    printf("   │ Bandwidth      │ %.0f GB/s  │ %.0f GB/s  │\n",
           unfused_bw, fused_bw);
    printf("   │ Speedup        │ 1.00x      │ %.2fx      │\n",
           unfused_ms / fused_ms);
    printf("   └────────────────┴────────────┴────────────┘\n\n");
    
    printf("   Correctness: %s (expected %.4f)\n\n", 
           correct ? "PASSED" : "FAILED", expected);
    
    printf("3. Memory Traffic Analysis\n");
    printf("─────────────────────────────────────────\n");
    printf("   Unfused operations:\n");
    printf("     add:  read a,b → write c (3 transactions)\n");
    printf("     mul:  read c   → write d (2 transactions)\n");
    printf("     sqrt: read d   → write e (2 transactions)\n");
    printf("     Total: 7 transactions\n\n");
    printf("   Fused operation:\n");
    printf("     fused: read a,b → write out (3 transactions)\n");
    printf("     Total: 3 transactions\n\n");
    printf("   Traffic reduction: %.1fx\n\n", unfused_bytes / fused_bytes);
    
    printf("=== Key Points ===\n\n");
    printf("1. Kernel launch overhead: ~5-10 μs each\n");
    printf("2. Fusing eliminates intermediate memory traffic\n");
    printf("3. Speedup depends on compute vs memory ratio\n");
    printf("4. Memory-bound kernels benefit most from fusion\n");
    
    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_d);
    cudaFree(d_e);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
