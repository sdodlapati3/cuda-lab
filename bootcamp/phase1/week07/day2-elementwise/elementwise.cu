/**
 * elementwise.cu - Generalized element-wise operations
 * 
 * Learning objectives:
 * - Implement various map patterns
 * - Compare performance characteristics
 * - Understand arithmetic intensity
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// Template for element-wise operations
template<typename Op>
__global__ void elementwise_unary(float* out, const float* in, int n, Op op) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = op(in[idx]);
    }
}

template<typename Op>
__global__ void elementwise_binary(float* out, const float* a, const float* b, 
                                   int n, Op op) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = op(a[idx], b[idx]);
    }
}

// Individual operations for cleaner profiling
__global__ void copy_kernel(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx];
    }
}

__global__ void scale_kernel(float* out, const float* in, float s, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = s * in[idx];
    }
}

__global__ void add_kernel(float* out, const float* a, const float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

__global__ void mul_kernel(float* out, const float* a, const float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * b[idx];
    }
}

__global__ void sqrt_kernel(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = sqrtf(in[idx]);
    }
}

__global__ void exp_kernel(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = expf(in[idx]);
    }
}

__global__ void tanh_kernel(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = tanhf(in[idx]);
    }
}

// Sigmoid: commonly used in neural networks
__global__ void sigmoid_kernel(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = 1.0f / (1.0f + expf(-in[idx]));
    }
}

// ReLU: very simple but important
__global__ void relu_kernel(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fmaxf(0.0f, in[idx]);
    }
}

struct BenchmarkResult {
    const char* name;
    int reads_per_elem;
    int writes_per_elem;
    int flops_per_elem;
    float time_ms;
    float bandwidth;
    float gflops;
};

int main() {
    printf("=== Element-wise Operations Comparison ===\n\n");
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    
    float peak_bandwidth = prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8) * 2 / 1e9;
    printf("Peak bandwidth: %.0f GB/s\n\n", peak_bandwidth);
    
    const int N = 1 << 24;  // 16M elements
    const int TRIALS = 100;
    size_t bytes = N * sizeof(float);
    
    printf("Array size: %d elements (%.1f MB)\n\n", N, bytes / 1e6);
    
    // Allocate
    float *h_a = new float[N];
    float *h_b = new float[N];
    
    for (int i = 0; i < N; i++) {
        h_a[i] = (rand() / (float)RAND_MAX) * 2 - 1;  // [-1, 1]
        h_b[i] = (rand() / (float)RAND_MAX) * 2 - 1;
    }
    
    float *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_out, bytes);
    
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    
    int block_size = 256;
    int num_blocks = (N + block_size - 1) / block_size;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    BenchmarkResult results[] = {
        {"Copy",    1, 1, 0, 0, 0, 0},
        {"Scale",   1, 1, 1, 0, 0, 0},
        {"Add",     2, 1, 1, 0, 0, 0},
        {"Mul",     2, 1, 1, 0, 0, 0},
        {"Sqrt",    1, 1, 1, 0, 0, 0},  // sqrt is ~20 cycles
        {"Exp",     1, 1, 20, 0, 0, 0}, // exp is ~20 FLOPs equivalent
        {"Tanh",    1, 1, 30, 0, 0, 0}, // tanh is expensive
        {"Sigmoid", 1, 1, 25, 0, 0, 0},
        {"ReLU",    1, 1, 1, 0, 0, 0},
    };
    
    #define BENCHMARK(name, kernel_call, idx) \
        kernel_call; \
        cudaDeviceSynchronize(); \
        cudaEventRecord(start); \
        for (int t = 0; t < TRIALS; t++) { kernel_call; } \
        cudaEventRecord(stop); \
        cudaEventSynchronize(stop); \
        cudaEventElapsedTime(&results[idx].time_ms, start, stop); \
        results[idx].time_ms /= TRIALS;
    
    BENCHMARK("Copy",    copy_kernel<<<num_blocks, block_size>>>(d_out, d_a, N), 0);
    BENCHMARK("Scale",   scale_kernel<<<num_blocks, block_size>>>(d_out, d_a, 2.0f, N), 1);
    BENCHMARK("Add",     add_kernel<<<num_blocks, block_size>>>(d_out, d_a, d_b, N), 2);
    BENCHMARK("Mul",     mul_kernel<<<num_blocks, block_size>>>(d_out, d_a, d_b, N), 3);
    BENCHMARK("Sqrt",    sqrt_kernel<<<num_blocks, block_size>>>(d_out, d_a, N), 4);
    BENCHMARK("Exp",     exp_kernel<<<num_blocks, block_size>>>(d_out, d_a, N), 5);
    BENCHMARK("Tanh",    tanh_kernel<<<num_blocks, block_size>>>(d_out, d_a, N), 6);
    BENCHMARK("Sigmoid", sigmoid_kernel<<<num_blocks, block_size>>>(d_out, d_a, N), 7);
    BENCHMARK("ReLU",    relu_kernel<<<num_blocks, block_size>>>(d_out, d_a, N), 8);
    
    // Calculate metrics
    for (auto& r : results) {
        int total_bytes = (r.reads_per_elem + r.writes_per_elem) * sizeof(float) * N;
        int total_flops = r.flops_per_elem * N;
        r.bandwidth = total_bytes / r.time_ms / 1e6;  // GB/s
        r.gflops = total_flops / r.time_ms / 1e6;     // GFLOPS
    }
    
    printf("%-10s %-10s %-12s %-12s %-15s %-10s\n",
           "Op", "Time(ms)", "BW(GB/s)", "BW Eff%", "GFLOPS", "Intensity");
    printf("----------------------------------------------------------------------------\n");
    
    for (const auto& r : results) {
        float intensity = (float)r.flops_per_elem / 
                         ((r.reads_per_elem + r.writes_per_elem) * sizeof(float));
        printf("%-10s %-10.3f %-12.1f %-12.1f%% %-15.1f %.2f\n",
               r.name, r.time_ms, r.bandwidth, 
               100.0f * r.bandwidth / peak_bandwidth,
               r.gflops, intensity);
    }
    
    printf("\n=== Analysis ===\n");
    printf("1. Copy/Scale/Add/Mul/ReLU: Pure memory-bound (same BW)\n");
    printf("2. Sqrt/Exp/Tanh/Sigmoid: Mix of compute and memory\n");
    printf("3. ReLU is as fast as copy despite comparison (branch-free)\n");
    printf("4. Transcendental functions (exp, tanh) have higher latency\n");
    printf("5. All simple ops achieve similar bandwidth → memory limited\n");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    delete[] h_a;
    delete[] h_b;
    
    return 0;
}
