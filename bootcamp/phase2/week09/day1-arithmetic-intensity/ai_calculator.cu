/**
 * ai_calculator.cu - Arithmetic Intensity Calculator
 * 
 * Learning objectives:
 * - Understand what arithmetic intensity means
 * - Calculate AI for different kernels
 * - Predict memory vs compute bound
 */

#include <cuda_runtime.h>
#include <cstdio>

// Get device peak performance
void print_device_roofline() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("=== Device: %s ===\n\n", prop.name);
    
    // Memory bandwidth (theoretical peak)
    // memoryClockRate is in kHz, memoryBusWidth in bits
    float mem_clock_ghz = prop.memoryClockRate / 1e6;  // kHz to GHz
    float bus_width_bytes = prop.memoryBusWidth / 8.0;
    float peak_bw_gb = 2.0 * mem_clock_ghz * bus_width_bytes;  // 2x for DDR
    
    printf("Memory:\n");
    printf("  Clock: %.2f GHz\n", mem_clock_ghz);
    printf("  Bus Width: %d bits\n", prop.memoryBusWidth);
    printf("  Peak Bandwidth: %.0f GB/s\n", peak_bw_gb);
    
    // Compute (theoretical peak for FP32)
    // Each SM has N CUDA cores, clock rate in kHz
    float gpu_clock_ghz = prop.clockRate / 1e6;
    int cuda_cores_per_sm;
    
    // Approximate CUDA cores per SM based on compute capability
    int major = prop.major;
    int minor = prop.minor;
    if (major == 8 && minor == 0) cuda_cores_per_sm = 64;       // A100
    else if (major == 8 && minor == 6) cuda_cores_per_sm = 128; // RTX 30xx
    else if (major == 8 && minor == 9) cuda_cores_per_sm = 128; // RTX 40xx
    else if (major == 7 && minor == 5) cuda_cores_per_sm = 64;  // Turing
    else if (major == 7 && minor == 0) cuda_cores_per_sm = 64;  // V100
    else cuda_cores_per_sm = 64;  // Default estimate
    
    int total_cores = prop.multiProcessorCount * cuda_cores_per_sm;
    float peak_tflops = 2.0 * gpu_clock_ghz * total_cores / 1000.0;  // 2x for FMA
    
    printf("\nCompute:\n");
    printf("  SM Count: %d\n", prop.multiProcessorCount);
    printf("  CUDA Cores per SM: ~%d\n", cuda_cores_per_sm);
    printf("  Total CUDA Cores: ~%d\n", total_cores);
    printf("  GPU Clock: %.2f GHz\n", gpu_clock_ghz);
    printf("  Peak FP32: ~%.1f TFLOPS\n", peak_tflops);
    
    // Ridge point
    float ridge_point = (peak_tflops * 1000) / peak_bw_gb;
    printf("\nRoofline Ridge Point: %.2f FLOPS/Byte\n", ridge_point);
    printf("  Kernels with AI < %.2f are MEMORY-BOUND\n", ridge_point);
    printf("  Kernels with AI > %.2f are COMPUTE-BOUND\n", ridge_point);
}

// Calculate AI for common patterns
void calculate_ai_examples() {
    printf("\n=== Arithmetic Intensity Examples ===\n\n");
    
    struct KernelAI {
        const char* name;
        const char* formula;
        int flops;
        int bytes;
        float ai;
        const char* bound;
    };
    
    // Assuming ridge point ~10 FLOPS/byte for modern GPUs
    float ridge = 10.0f;
    
    KernelAI examples[] = {
        {"Vector Add", "C[i] = A[i] + B[i]", 1, 12, 1.0f/12, "memory"},
        {"SAXPY", "Y[i] = a*X[i] + Y[i]", 2, 12, 2.0f/12, "memory"},
        {"Reduction", "sum += A[i]", 1, 4, 0.25f, "memory"},
        {"Dot Product", "sum += A[i]*B[i]", 2, 8, 0.25f, "memory"},
        {"Stencil 3pt", "B[i] = (A[i-1]+A[i]+A[i+1])/3", 3, 4, 0.75f, "memory"},
        {"Stencil 5pt", "5-point stencil", 5, 4, 1.25f, "memory"},
        {"MatMul tile", "N iterations, tile reuse", 2, 8.0f/32, 8.0f, "memory"},
        {"Dense GEMM", "Large matrix, full reuse", 2, 0.5f, 4.0f, "approaching compute"},
    };
    
    printf("%-20s %-30s %8s %8s %8s %s\n", 
           "Kernel", "Formula", "FLOPs", "Bytes", "AI", "Bound");
    printf("--------------------------------------------------------------------------------\n");
    
    for (auto& ex : examples) {
        printf("%-20s %-30s %8d %8d %8.3f %s\n",
               ex.name, ex.formula, ex.flops, ex.bytes, ex.ai, ex.bound);
    }
    
    printf("\n");
    printf("Key insight: Most kernels have AI << ridge point (%.1f)\n", ridge);
    printf("            => Most kernels are MEMORY-BOUND\n");
    printf("            => Optimize memory access patterns first!\n");
}

// Demonstrate how to measure AI empirically
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];  // 1 FLOP, 12 bytes
    }
}

__global__ void saxpy(float alpha, const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = alpha * x[idx] + y[idx];  // 2 FLOPs, 12 bytes
    }
}

void measure_kernel_ai() {
    printf("\n=== Measuring Kernel Performance vs AI ===\n\n");
    
    const int N = 1 << 24;
    const int TRIALS = 100;
    
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));
    
    // Initialize
    cudaMemset(d_a, 1, N * sizeof(float));
    cudaMemset(d_b, 1, N * sizeof(float));
    
    int blocks = (N + 255) / 256;
    int threads = 256;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;
    
    // Vector add: AI = 1/12 = 0.083
    cudaEventRecord(start);
    for (int i = 0; i < TRIALS; i++) {
        vector_add<<<blocks, threads>>>(d_a, d_b, d_c, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    
    float bytes = N * 12.0f;  // 2 reads + 1 write
    float flops = N * 1.0f;
    float bw = (bytes * TRIALS) / (ms / 1000) / 1e9;
    float gflops = (flops * TRIALS) / (ms / 1000) / 1e9;
    
    printf("Vector Add:\n");
    printf("  AI (theoretical): %.3f FLOPS/Byte\n", 1.0f/12);
    printf("  Bandwidth: %.1f GB/s\n", bw);
    printf("  GFLOPS: %.1f\n", gflops);
    printf("  This is memory-bound (AI << ridge point)\n\n");
    
    // SAXPY: AI = 2/12 = 0.167
    cudaEventRecord(start);
    for (int i = 0; i < TRIALS; i++) {
        saxpy<<<blocks, threads>>>(2.0f, d_a, d_b, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    
    flops = N * 2.0f;
    gflops = (flops * TRIALS) / (ms / 1000) / 1e9;
    bw = (bytes * TRIALS) / (ms / 1000) / 1e9;  // Same bytes
    
    printf("SAXPY:\n");
    printf("  AI (theoretical): %.3f FLOPS/Byte\n", 2.0f/12);
    printf("  Bandwidth: %.1f GB/s\n", bw);
    printf("  GFLOPS: %.1f\n", gflops);
    printf("  2x FLOPs but same bandwidth => still memory-bound\n");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

int main() {
    print_device_roofline();
    calculate_ai_examples();
    measure_kernel_ai();
    
    printf("\n=== Key Takeaways ===\n");
    printf("1. AI = FLOPs / Bytes tells you if kernel is memory or compute bound\n");
    printf("2. Ridge point = Peak FLOPS / Peak BW (typically 5-15 for modern GPUs)\n");
    printf("3. Most kernels (AI < 1) are deeply memory-bound\n");
    printf("4. For memory-bound kernels, optimize memory access first\n");
    printf("5. Adding more compute to memory-bound kernel doesn't help!\n");
    
    return 0;
}
