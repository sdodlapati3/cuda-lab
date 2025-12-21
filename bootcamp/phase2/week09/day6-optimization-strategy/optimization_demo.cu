/**
 * optimization_demo.cu - Optimization strategies based on roofline
 * 
 * Learning objectives:
 * - See optimization impact on roofline position
 * - Compare naive vs optimized kernels
 * - Understand which optimizations help which bound types
 */

#include <cuda_runtime.h>
#include <cstdio>

// ========== Memory-Bound Optimization Example: Strided → Coalesced ==========

// Bad: Strided access (column-major on row-major data)
__global__ void strided_access(const float* in, float* out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        // Column-major access pattern - bad for coalescing
        int idx_in = x * height + y;  // Strided
        int idx_out = y * width + x;  // Row-major output
        out[idx_out] = in[idx_in] * 2.0f;
    }
}

// Good: Coalesced access
__global__ void coalesced_access(const float* in, float* out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        // Row-major access - coalesced
        int idx = y * width + x;
        out[idx] = in[idx] * 2.0f;
    }
}

// ========== AI Improvement: Kernel Fusion ==========

// Bad: Separate kernels (3 kernel launches, 3x memory traffic)
__global__ void scale_kernel(const float* in, float* out, float s, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx] * s;
}

__global__ void add_kernel(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] + b[idx];
}

// Good: Fused kernel (1 launch, 1x memory traffic for same FLOPs)
__global__ void fused_scale_add(const float* a, const float* b, float* out, 
                                 float s1, float s2, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * s1 + b[idx] * s2;  // 3 FLOPs instead of 2
    }
}

// ========== Compute-Bound: ILP Improvement ==========

// Low ILP: Sequential dependencies
__global__ void low_ilp(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = in[idx];
        // Serial chain - each depends on previous
        for (int i = 0; i < 100; i++) {
            val = val * 1.0001f + 0.0001f;
        }
        out[idx] = val;
    }
}

// High ILP: Independent operations
__global__ void high_ilp(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = in[idx];
        // 4 independent chains
        float v0 = val, v1 = val, v2 = val, v3 = val;
        
        for (int i = 0; i < 25; i++) {
            v0 = v0 * 1.0001f + 0.0001f;
            v1 = v1 * 1.0002f + 0.0002f;
            v2 = v2 * 1.0003f + 0.0003f;
            v3 = v3 * 1.0004f + 0.0004f;
        }
        out[idx] = v0 + v1 + v2 + v3;
    }
}

void benchmark(const char* name, void (*func)(), int trials) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    for (int i = 0; i < 3; i++) func();
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < trials; i++) func();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("%-25s: %8.3f ms\n", name, ms / trials);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Global variables for benchmark functions
float *g_d_in, *g_d_out, *g_d_a, *g_d_b, *g_d_temp;
int g_width, g_height, g_n;
dim3 g_blocks2d, g_threads2d, g_blocks1d, g_threads1d;

void run_strided() {
    strided_access<<<g_blocks2d, g_threads2d>>>(g_d_in, g_d_out, g_width, g_height);
}

void run_coalesced() {
    coalesced_access<<<g_blocks2d, g_threads2d>>>(g_d_in, g_d_out, g_width, g_height);
}

void run_separate_kernels() {
    scale_kernel<<<g_blocks1d, g_threads1d>>>(g_d_a, g_d_temp, 2.0f, g_n);
    scale_kernel<<<g_blocks1d, g_threads1d>>>(g_d_b, g_d_out, 3.0f, g_n);
    add_kernel<<<g_blocks1d, g_threads1d>>>(g_d_temp, g_d_out, g_d_out, g_n);
}

void run_fused_kernel() {
    fused_scale_add<<<g_blocks1d, g_threads1d>>>(g_d_a, g_d_b, g_d_out, 2.0f, 3.0f, g_n);
}

void run_low_ilp() {
    low_ilp<<<g_blocks1d, g_threads1d>>>(g_d_in, g_d_out, g_n);
}

void run_high_ilp() {
    high_ilp<<<g_blocks1d, g_threads1d>>>(g_d_in, g_d_out, g_n);
}

int main() {
    printf("=== Optimization Strategy Demos ===\n\n");
    
    // Setup for 2D kernels
    g_width = 4096;
    g_height = 4096;
    g_n = g_width * g_height;
    
    cudaMalloc(&g_d_in, g_n * sizeof(float));
    cudaMalloc(&g_d_out, g_n * sizeof(float));
    cudaMalloc(&g_d_a, g_n * sizeof(float));
    cudaMalloc(&g_d_b, g_n * sizeof(float));
    cudaMalloc(&g_d_temp, g_n * sizeof(float));
    
    g_threads2d = dim3(16, 16);
    g_blocks2d = dim3((g_width + 15) / 16, (g_height + 15) / 16);
    g_threads1d = dim3(256);
    g_blocks1d = dim3((g_n + 255) / 256);
    
    const int TRIALS = 50;
    
    printf("1. Memory Access Pattern (strided vs coalesced)\n");
    printf("   Roofline impact: Move UP toward memory ceiling\n");
    printf("   ------------------------------------------------\n");
    benchmark("strided_access", run_strided, TRIALS);
    benchmark("coalesced_access", run_coalesced, TRIALS);
    printf("\n");
    
    printf("2. Kernel Fusion (separate vs fused)\n");
    printf("   Roofline impact: Move RIGHT (higher AI) and UP\n");
    printf("   ------------------------------------------------\n");
    benchmark("separate_kernels (3x)", run_separate_kernels, TRIALS);
    benchmark("fused_kernel (1x)", run_fused_kernel, TRIALS);
    printf("\n");
    
    printf("3. Instruction-Level Parallelism (low vs high ILP)\n");
    printf("   Roofline impact: Move UP toward compute ceiling\n");
    printf("   ------------------------------------------------\n");
    benchmark("low_ilp", run_low_ilp, TRIALS);
    benchmark("high_ilp", run_high_ilp, TRIALS);
    printf("\n");
    
    printf("=== Optimization Decision Matrix ===\n\n");
    printf("Symptom                     | Likely Cause          | Solution\n");
    printf("----------------------------+-----------------------+----------------------\n");
    printf("Low BW, low AI              | Poor access patterns  | Coalesce, reorder\n");
    printf("Good BW, below ceiling      | Optimal for this AI   | Increase AI (fusion)\n");
    printf("Low FLOPS, high AI          | Low ILP / occupancy   | Unroll, more threads\n");
    printf("Both metrics low            | Launch overhead       | Persistent kernels\n");
    printf("Both near ceiling           | Hardware limit        | Change algorithm\n");
    printf("\n");
    
    printf("=== Optimization Workflow ===\n\n");
    printf("1. Profile with NCU: ncu --set roofline ./kernel\n");
    printf("2. Identify position on roofline\n");
    printf("3. Determine bottleneck (memory vs compute vs latency)\n");
    printf("4. Apply appropriate optimization\n");
    printf("5. Re-profile and measure improvement\n");
    printf("6. Repeat until at ceiling or acceptable performance\n");
    
    cudaFree(g_d_in);
    cudaFree(g_d_out);
    cudaFree(g_d_a);
    cudaFree(g_d_b);
    cudaFree(g_d_temp);
    
    return 0;
}
