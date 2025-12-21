/**
 * Benchmark template
 */

#include "cuda_utils.cuh"
#include <vector>
#include <algorithm>
#include <cmath>

// Your kernels here
__global__ void kernel_v1(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] *= 2.0f;
}

__global__ void kernel_v2(float4* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float4 val = data[idx];
        data[idx] = make_float4(val.x * 2, val.y * 2, val.z * 2, val.w * 2);
    }
}

struct BenchResult {
    const char* name;
    float mean_ms;
    float stddev_ms;
    float bandwidth_gb_s;
};

template <typename Func>
BenchResult benchmark(const char* name, Func fn, size_t bytes, 
                      int warmup = 10, int iterations = 100) {
    // Warmup
    for (int i = 0; i < warmup; i++) fn();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Collect times
    std::vector<float> times(iterations);
    CudaTimer timer;
    
    for (int i = 0; i < iterations; i++) {
        timer.start();
        fn();
        timer.stop();
        times[i] = timer.elapsed_ms();
    }
    
    // Stats
    float sum = 0;
    for (float t : times) sum += t;
    float mean = sum / iterations;
    
    float sq_sum = 0;
    for (float t : times) sq_sum += (t - mean) * (t - mean);
    float stddev = std::sqrt(sq_sum / iterations);
    
    float bw = (bytes / 1e9) / (mean / 1e3);
    
    return {name, mean, stddev, bw};
}

void print_results(const std::vector<BenchResult>& results) {
    printf("\n╔══════════════════════════════════════════════════════════╗\n");
    printf("║ Kernel          │ Time (μs) │ σ (μs) │ GB/s   │ Speedup ║\n");
    printf("╠══════════════════════════════════════════════════════════╣\n");
    
    float baseline = results[0].mean_ms;
    for (const auto& r : results) {
        printf("║ %-15s │ %9.1f │ %6.1f │ %6.1f │ %6.2f× ║\n",
               r.name, r.mean_ms * 1000, r.stddev_ms * 1000,
               r.bandwidth_gb_s, baseline / r.mean_ms);
    }
    printf("╚══════════════════════════════════════════════════════════╝\n\n");
}

int main() {
    get_gpu_info().print();
    
    const int N = 1 << 24;  // 16M
    const size_t bytes = N * sizeof(float) * 2;  // read + write
    
    DeviceBuffer<float> data(N);
    
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    int blocks4 = (N/4 + threads - 1) / threads;
    
    std::vector<BenchResult> results;
    
    results.push_back(benchmark("v1_scalar", [&]() {
        kernel_v1<<<blocks, threads>>>(data.data(), N);
    }, bytes));
    
    results.push_back(benchmark("v2_vector4", [&]() {
        kernel_v2<<<blocks4, threads>>>((float4*)data.data(), N/4);
    }, bytes));
    
    print_results(results);
    
    return 0;
}
