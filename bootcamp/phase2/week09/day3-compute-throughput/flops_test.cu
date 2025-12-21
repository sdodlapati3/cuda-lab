/**
 * flops_test.cu - Measure peak compute throughput
 * 
 * Learning objectives:
 * - Measure theoretical vs achieved FLOPS
 * - Understand FMA contribution
 * - Establish compute ceiling for roofline
 */

#include <cuda_runtime.h>
#include <cstdio>

// FMA-heavy kernel to measure peak FLOPS
// Each thread does ITERATIONS * UNROLL FMAs
template<int ITERATIONS, int UNROLL>
__global__ void fma_kernel(float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Use registers to avoid memory bottleneck
    float a = 1.0001f;
    float b = 0.9999f;
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
    float c4 = 0.0f, c5 = 0.0f, c6 = 0.0f, c7 = 0.0f;
    
    #pragma unroll 1
    for (int i = 0; i < ITERATIONS; i++) {
        // 8-way unroll for ILP
        #pragma unroll
        for (int j = 0; j < UNROLL / 8; j++) {
            c0 = a * c0 + b;
            c1 = a * c1 + b;
            c2 = a * c2 + b;
            c3 = a * c3 + b;
            c4 = a * c4 + b;
            c5 = a * c5 + b;
            c6 = a * c6 + b;
            c7 = a * c7 + b;
        }
    }
    
    // Prevent optimization
    if (idx == 0) {
        out[0] = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7;
    }
}

void print_device_compute_info() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("=== Device: %s ===\n", prop.name);
    
    float gpu_clock_ghz = prop.clockRate / 1e6;
    int cuda_cores_per_sm;
    
    // CUDA cores per SM by compute capability
    int major = prop.major, minor = prop.minor;
    if (major == 8 && minor == 0) cuda_cores_per_sm = 64;       // A100
    else if (major == 8 && minor == 6) cuda_cores_per_sm = 128; // RTX 30xx
    else if (major == 8 && minor == 9) cuda_cores_per_sm = 128; // RTX 40xx
    else if (major == 7 && minor == 5) cuda_cores_per_sm = 64;  // Turing
    else if (major == 7 && minor == 0) cuda_cores_per_sm = 64;  // V100
    else cuda_cores_per_sm = 64;
    
    int total_cores = prop.multiProcessorCount * cuda_cores_per_sm;
    float peak_tflops = 2.0 * gpu_clock_ghz * total_cores / 1000.0;
    
    printf("SM Count: %d\n", prop.multiProcessorCount);
    printf("Estimated CUDA Cores: %d\n", total_cores);
    printf("GPU Clock: %.2f GHz\n", gpu_clock_ghz);
    printf("Theoretical Peak FP32: %.1f TFLOPS\n", peak_tflops);
    printf("\n");
}

int main() {
    print_device_compute_info();
    
    printf("=== Measuring Peak FLOPS ===\n\n");
    
    float* d_out;
    cudaMalloc(&d_out, sizeof(float));
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Configuration
    const int THREADS = 256;
    const int BLOCKS = 2048;  // Saturate GPU
    const int TRIALS = 10;
    
    // Vary FMA count per thread
    const int ITERATIONS = 1000;
    const int UNROLL = 64;  // 64 FMAs per iteration
    
    // Total FLOPs = threads × blocks × iterations × unroll × 2 (FMA = 2 FLOPs)
    long long total_flops = (long long)THREADS * BLOCKS * ITERATIONS * UNROLL * 2;
    
    printf("Threads: %d, Blocks: %d\n", THREADS, BLOCKS);
    printf("FMAs per thread: %d\n", ITERATIONS * UNROLL);
    printf("Total FLOPs per kernel: %.2f GFLOPS\n", total_flops / 1e9);
    printf("\n");
    
    // Warmup
    for (int i = 0; i < 3; i++) {
        fma_kernel<ITERATIONS, UNROLL><<<BLOCKS, THREADS>>>(d_out, 0);
    }
    cudaDeviceSynchronize();
    
    // Measure
    cudaEventRecord(start);
    for (int t = 0; t < TRIALS; t++) {
        fma_kernel<ITERATIONS, UNROLL><<<BLOCKS, THREADS>>>(d_out, 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    float achieved_tflops = (total_flops * TRIALS) / (ms / 1000) / 1e12;
    
    printf("Time: %.2f ms for %d iterations\n", ms, TRIALS);
    printf("Achieved: %.2f TFLOPS\n", achieved_tflops);
    
    // Get theoretical for comparison
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    float gpu_clock_ghz = prop.clockRate / 1e6;
    int cuda_cores_per_sm = (prop.major == 8) ? ((prop.minor == 0) ? 64 : 128) : 64;
    float peak_tflops = 2.0 * gpu_clock_ghz * prop.multiProcessorCount * cuda_cores_per_sm / 1000.0;
    
    printf("Efficiency: %.1f%% of theoretical peak\n", 100 * achieved_tflops / peak_tflops);
    
    printf("\n=== This is Your Compute Ceiling ===\n");
    printf("For the roofline model, use: %.1f TFLOPS\n", achieved_tflops);
    printf("(Practical peak, not theoretical)\n");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_out);
    
    return 0;
}
