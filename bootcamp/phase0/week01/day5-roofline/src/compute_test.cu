/**
 * Compute Test - Measure peak GFLOPS
 * 
 * Strategy: Pure compute kernel with data reuse (no memory bottleneck).
 * This gives us the "compute roof" of the roofline.
 */

#include <cuda_runtime.h>
#include <cstdio>

// FP32 compute kernel - repeated FMA operations
__global__ void compute_fp32_kernel(float* result, int iterations) {
    float val = 1.0f + 0.001f * threadIdx.x;  // Avoid same value optimization
    
    // Unrolled FMA loop - each iteration is 2 FLOPs (multiply + add)
    #pragma unroll
    for (int i = 0; i < 1024; i++) {
        val = val * 1.0001f + 0.0001f;
        val = val * 1.0001f + 0.0001f;
        val = val * 1.0001f + 0.0001f;
        val = val * 1.0001f + 0.0001f;
    }
    
    // Prevent optimization - write result (only thread 0)
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = val;
    }
}

// FP16 compute kernel (if supported)
#if __CUDA_ARCH__ >= 700
#include <cuda_fp16.h>

__global__ void compute_fp16_kernel(half* result, int iterations) {
    half val = __float2half(1.0f + 0.001f * threadIdx.x);
    half one = __float2half(1.0001f);
    half tiny = __float2half(0.0001f);
    
    #pragma unroll
    for (int i = 0; i < 1024; i++) {
        val = __hfma(val, one, tiny);
        val = __hfma(val, one, tiny);
        val = __hfma(val, one, tiny);
        val = __hfma(val, one, tiny);
    }
    
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = val;
    }
}
#endif

struct ComputeResults {
    float fp32_gflops;
    float fp16_gflops;
    float theoretical_fp32_gflops;
};

ComputeResults measure_compute() {
    const int THREADS = 256;
    const int BLOCKS = 1024;  // Use many blocks to saturate SMs
    const int WARMUP = 10;
    const int ITERATIONS = 100;
    
    // FLOPs per kernel call
    // 1024 iterations × 4 FMAs × 2 FLOPs/FMA = 8192 FLOPs per thread
    const size_t FLOPS_PER_THREAD = 1024 * 4 * 2;
    const size_t FLOPS_PER_KERNEL = (size_t)THREADS * BLOCKS * FLOPS_PER_THREAD;
    
    printf("\n=== Compute Measurement ===\n");
    printf("Threads: %d, Blocks: %d\n", THREADS, BLOCKS);
    printf("FLOPs per kernel: %.2f GFLOP\n", FLOPS_PER_KERNEL / 1e9);
    
    float *d_result;
    cudaMalloc(&d_result, sizeof(float));
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    for (int i = 0; i < WARMUP; i++) {
        compute_fp32_kernel<<<BLOCKS, THREADS>>>(d_result, 1);
    }
    cudaDeviceSynchronize();
    
    // Measure FP32
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        compute_fp32_kernel<<<BLOCKS, THREADS>>>(d_result, 1);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float fp32_ms;
    cudaEventElapsedTime(&fp32_ms, start, stop);
    float fp32_gflops = (FLOPS_PER_KERNEL * ITERATIONS / 1e9) / (fp32_ms / 1e3);
    
    // Get theoretical peak
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    // FP32 cores × 2 (FMA = 2 ops) × clock rate
    // Note: This is approximate - actual formula varies by architecture
    float theoretical_fp32 = props.multiProcessorCount * 64 * 2 * props.clockRate / 1e6;
    
    printf("FP32 GFLOPS:          %.1f\n", fp32_gflops);
    printf("Theoretical FP32:     %.1f GFLOPS (approx)\n", theoretical_fp32);
    printf("Achieved:             %.1f%%\n", 100.0f * fp32_gflops / theoretical_fp32);
    
    // FP16 measurement (similar pattern)
    float fp16_gflops = 0.0f;
    #if __CUDA_ARCH__ >= 700
    // Would measure FP16 here
    #endif
    
    cudaFree(d_result);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return {fp32_gflops, fp16_gflops, theoretical_fp32};
}
