/**
 * Day 4: Roofline Model Demo
 * 
 * Measure and visualize kernel performance on roofline.
 */

#include <cstdio>
#include <cuda_runtime.h>
#include <cmath>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// Device info
double get_peak_bandwidth_gb() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    // Bandwidth = 2 * memoryClockRate * busWidth / 8
    return 2.0 * prop.memoryClockRate * 1e3 * prop.memoryBusWidth / 8.0 / 1e9;
}

double get_peak_flops_gflops() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    // FP32: 2 * clockRate * cores
    int cores_per_sm;
    if (prop.major == 8 && prop.minor == 0) cores_per_sm = 64;  // A100
    else if (prop.major == 8) cores_per_sm = 128;  // Ampere consumer
    else if (prop.major == 7) cores_per_sm = 64;   // Volta/Turing
    else cores_per_sm = 128;  // Default
    
    return 2.0 * prop.clockRate * 1e3 * prop.multiProcessorCount * cores_per_sm / 1e9;
}

// ============================================================================
// SAXPY - Low arithmetic intensity
// ============================================================================
__global__ void saxpy_kernel(int n, float a, float* x, float* y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = a * x[idx] + y[idx];  // 2 FLOPs, 12 bytes
    }
}

// ============================================================================
// High compute kernel - High arithmetic intensity
// ============================================================================
__global__ void compute_heavy_kernel(int n, float* x, float* y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx];
        // 200+ FLOPs per element
        for (int i = 0; i < 50; i++) {
            val = sinf(val) * cosf(val) + val * 0.99f;
        }
        y[idx] = val;
    }
}

struct KernelResult {
    const char* name;
    double flops;
    double bytes;
    double time_ms;
    double ai;
    double achieved_gflops;
    double achieved_bandwidth;
};

KernelResult benchmark_saxpy(int n, float* d_x, float* d_y) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Warmup
    saxpy_kernel<<<numBlocks, blockSize>>>(n, 2.0f, d_x, d_y);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    int iters = 100;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) {
        saxpy_kernel<<<numBlocks, blockSize>>>(n, 2.0f, d_x, d_y);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= iters;
    
    double flops = 2.0 * n;  // 2 FLOPs per element
    double bytes = 3.0 * n * sizeof(float);  // 2 reads + 1 write
    double ai = flops / bytes;
    double achieved_gflops = flops / (ms * 1e6);
    double achieved_bw = bytes / (ms * 1e6);
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return {"SAXPY", flops, bytes, ms, ai, achieved_gflops, achieved_bw};
}

KernelResult benchmark_compute(int n, float* d_x, float* d_y) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Warmup
    compute_heavy_kernel<<<numBlocks, blockSize>>>(n, d_x, d_y);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    int iters = 10;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) {
        compute_heavy_kernel<<<numBlocks, blockSize>>>(n, d_x, d_y);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= iters;
    
    // Approximate FLOPs (50 iterations * ~4 FLOPs per trig)
    double flops = 200.0 * n;
    double bytes = 2.0 * n * sizeof(float);  // 1 read + 1 write
    double ai = flops / bytes;
    double achieved_gflops = flops / (ms * 1e6);
    double achieved_bw = bytes / (ms * 1e6);
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return {"Compute Heavy", flops, bytes, ms, ai, achieved_gflops, achieved_bw};
}

void print_roofline(KernelResult r, double peak_gflops, double peak_bw) {
    double ridge_point = peak_gflops / peak_bw;
    double roofline_limit = (r.ai < ridge_point) 
        ? r.ai * peak_bw  // Memory bound
        : peak_gflops;    // Compute bound
    
    printf("\n%s:\n", r.name);
    printf("  FLOPs: %.2e, Bytes: %.2e\n", r.flops, r.bytes);
    printf("  Arithmetic Intensity: %.2f FLOP/Byte\n", r.ai);
    printf("  Time: %.3f ms\n", r.time_ms);
    printf("  Achieved: %.1f GFLOP/s (%.1f%% of roofline limit)\n", 
           r.achieved_gflops, 100.0 * r.achieved_gflops / roofline_limit);
    printf("  Bandwidth: %.1f GB/s (%.1f%% of peak)\n",
           r.achieved_bandwidth, 100.0 * r.achieved_bandwidth / peak_bw);
    printf("  Bound: %s\n", r.ai < ridge_point ? "MEMORY" : "COMPUTE");
}

int main() {
    printf("Roofline Model Analysis\n");
    printf("=======================\n\n");
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s\n", prop.name);
    
    double peak_bw = get_peak_bandwidth_gb();
    double peak_gflops = get_peak_flops_gflops();
    double ridge = peak_gflops / peak_bw;
    
    printf("Peak Bandwidth: %.1f GB/s\n", peak_bw);
    printf("Peak FP32: %.1f GFLOP/s\n", peak_gflops);
    printf("Ridge Point: %.1f FLOP/Byte\n", ridge);
    
    const int N = 1 << 24;
    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, N * sizeof(float)));
    
    float* h_x = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) h_x[i] = 1.0f;
    CUDA_CHECK(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_x, N * sizeof(float), cudaMemcpyHostToDevice));
    
    // Benchmark kernels
    KernelResult saxpy = benchmark_saxpy(N, d_x, d_y);
    KernelResult compute = benchmark_compute(N, d_x, d_y);
    
    print_roofline(saxpy, peak_gflops, peak_bw);
    print_roofline(compute, peak_gflops, peak_bw);
    
    printf("\n============================================\n");
    printf("Visual Roofline (AI vs GFLOP/s):\n\n");
    printf("  Peak: %.0f GFLOP/s\n", peak_gflops);
    printf("       |\n");
    printf("       |           _____ compute ceiling\n");
    printf("       |         /\n");
    printf("       |       / * Compute Heavy (AI=%.1f)\n", compute.ai);
    printf("       |     /\n");
    printf("       |   /\n");
    printf("       | * SAXPY (AI=%.2f)\n", saxpy.ai);
    printf("       |/\n");
    printf("  0    +------------------------> AI\n");
    printf("       0    ridge=%.1f\n", ridge);
    
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    free(h_x);
    
    return 0;
}
