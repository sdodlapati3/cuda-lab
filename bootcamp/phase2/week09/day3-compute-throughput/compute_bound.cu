/**
 * compute_bound.cu - Example compute-bound kernel
 * 
 * Learning objectives:
 * - Identify compute-bound behavior
 * - See how adding compute doesn't slow memory-bound kernels
 * - See how memory-bound kernels become compute-bound
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// Parametric kernel: vary compute per memory access
template<int OPS_PER_ELEMENT>
__global__ void variable_compute(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = in[idx];
        
        #pragma unroll
        for (int i = 0; i < OPS_PER_ELEMENT; i++) {
            val = val * 1.0001f + 0.0001f;  // 2 FLOPs (1 FMA)
        }
        
        out[idx] = val;
    }
}

// Transcendental-heavy (expensive compute)
__global__ void transcendental_kernel(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = in[idx] + 0.1f;  // Avoid edge cases
        // sin + cos + exp = very expensive
        out[idx] = sinf(val) + cosf(val) + expf(-val * 0.1f);
    }
}

int main() {
    printf("=== Compute vs Memory Bound Transition ===\n\n");
    
    const int N = 1 << 24;  // 16M
    const int TRIALS = 20;
    
    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    
    // Initialize
    float* h_in = new float[N];
    for (int i = 0; i < N; i++) h_in[i] = 0.5f;
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
    
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;
    
    // Memory traffic is constant: 8 bytes per element (read + write)
    float memory_bytes = N * 8.0f;
    
    printf("%-15s %10s %10s %10s %12s %s\n", 
           "Ops/Element", "Time(ms)", "GB/s", "GFLOPS", "AI", "Bound");
    printf("------------------------------------------------------------------------\n");
    
    // OPS = 1
    cudaEventRecord(start);
    for (int i = 0; i < TRIALS; i++) {
        variable_compute<1><<<blocks, threads>>>(d_in, d_out, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float time_per = ms / TRIALS;
    float bw = memory_bytes / (time_per / 1000) / 1e9;
    float flops = N * 2.0f;
    float gflops = flops / (time_per / 1000) / 1e9;
    float ai = flops / memory_bytes;
    printf("%-15d %10.3f %10.1f %10.1f %12.3f %s\n", 
           1, time_per, bw, gflops, ai, "memory");
    
    // OPS = 10
    cudaEventRecord(start);
    for (int i = 0; i < TRIALS; i++) {
        variable_compute<10><<<blocks, threads>>>(d_in, d_out, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    time_per = ms / TRIALS;
    bw = memory_bytes / (time_per / 1000) / 1e9;
    flops = N * 20.0f;
    gflops = flops / (time_per / 1000) / 1e9;
    ai = flops / memory_bytes;
    printf("%-15d %10.3f %10.1f %10.1f %12.3f %s\n", 
           10, time_per, bw, gflops, ai, "memory");
    
    // OPS = 50
    cudaEventRecord(start);
    for (int i = 0; i < TRIALS; i++) {
        variable_compute<50><<<blocks, threads>>>(d_in, d_out, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    time_per = ms / TRIALS;
    bw = memory_bytes / (time_per / 1000) / 1e9;
    flops = N * 100.0f;
    gflops = flops / (time_per / 1000) / 1e9;
    ai = flops / memory_bytes;
    printf("%-15d %10.3f %10.1f %10.1f %12.3f %s\n", 
           50, time_per, bw, gflops, ai, "transition");
    
    // OPS = 100
    cudaEventRecord(start);
    for (int i = 0; i < TRIALS; i++) {
        variable_compute<100><<<blocks, threads>>>(d_in, d_out, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    time_per = ms / TRIALS;
    bw = memory_bytes / (time_per / 1000) / 1e9;
    flops = N * 200.0f;
    gflops = flops / (time_per / 1000) / 1e9;
    ai = flops / memory_bytes;
    printf("%-15d %10.3f %10.1f %10.1f %12.3f %s\n", 
           100, time_per, bw, gflops, ai, "compute");
    
    // OPS = 500
    cudaEventRecord(start);
    for (int i = 0; i < TRIALS; i++) {
        variable_compute<500><<<blocks, threads>>>(d_in, d_out, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    time_per = ms / TRIALS;
    bw = memory_bytes / (time_per / 1000) / 1e9;
    flops = N * 1000.0f;
    gflops = flops / (time_per / 1000) / 1e9;
    ai = flops / memory_bytes;
    printf("%-15d %10.3f %10.1f %10.1f %12.3f %s\n", 
           500, time_per, bw, gflops, ai, "compute");
    
    // Transcendentals
    cudaEventRecord(start);
    for (int i = 0; i < TRIALS; i++) {
        transcendental_kernel<<<blocks, threads>>>(d_in, d_out, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    time_per = ms / TRIALS;
    bw = memory_bytes / (time_per / 1000) / 1e9;
    // sin/cos/exp are ~10-20 "equivalent FLOPs" each
    flops = N * 50.0f;  // Approximate
    gflops = flops / (time_per / 1000) / 1e9;
    ai = flops / memory_bytes;
    printf("%-15s %10.3f %10.1f %10.1f %12.3f %s\n", 
           "transcendental", time_per, bw, gflops, ai, "compute");
    
    printf("\n=== Key Observations ===\n");
    printf("1. Low ops/element: Time constant, GFLOPS scales (memory-bound)\n");
    printf("2. High ops/element: Time scales with ops (compute-bound)\n");
    printf("3. Transition happens when AI ≈ ridge point (~10 FLOPS/byte)\n");
    printf("4. Bandwidth drops as kernel becomes compute-bound\n");
    printf("5. This is the roofline model in action!\n");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;
    
    return 0;
}
