#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include "cuda_timer.cuh"

/**
 * BenchmarkConfig - Settings for benchmark runs
 */
struct BenchmarkConfig {
    int warmup_iterations = 10;
    int timed_iterations = 100;
    bool print_header = true;
    bool print_speedup = true;
};

/**
 * BenchmarkResult - Stats from a single benchmark
 */
struct BenchmarkResult {
    const char* name;
    float mean_ms;
    float min_ms;
    float max_ms;
    float stddev_ms;
    size_t bytes_accessed;   // For bandwidth calculation
    size_t flops;            // For GFLOPS calculation
    
    float bandwidth_gb_s() const {
        return (bytes_accessed / 1e9) / (mean_ms / 1e3);
    }
    
    float gflops() const {
        return (flops / 1e9) / (mean_ms / 1e3);
    }
};

/**
 * Global result storage for comparison
 */
inline std::vector<BenchmarkResult>& get_results() {
    static std::vector<BenchmarkResult> results;
    return results;
}

/**
 * Detailed timing with statistics
 */
template <typename Func>
BenchmarkResult benchmark_detailed(const char* name, 
                                   Func kernel_fn,
                                   size_t bytes = 0,
                                   size_t flops = 0,
                                   const BenchmarkConfig& config = BenchmarkConfig()) {
    std::vector<float> times;
    times.reserve(config.timed_iterations);
    
    // Warmup
    for (int i = 0; i < config.warmup_iterations; i++) {
        kernel_fn();
    }
    cudaDeviceSynchronize();
    
    // Timed runs - measure each iteration separately for stddev
    CudaTimer timer;
    for (int i = 0; i < config.timed_iterations; i++) {
        timer.start();
        kernel_fn();
        timer.stop();
        times.push_back(timer.elapsed_ms());
    }
    
    // Calculate statistics
    float sum = 0, min_val = times[0], max_val = times[0];
    for (float t : times) {
        sum += t;
        min_val = std::min(min_val, t);
        max_val = std::max(max_val, t);
    }
    float mean = sum / times.size();
    
    float sq_sum = 0;
    for (float t : times) {
        sq_sum += (t - mean) * (t - mean);
    }
    float stddev = std::sqrt(sq_sum / times.size());
    
    BenchmarkResult result = {
        name,
        mean,
        min_val,
        max_val,
        stddev,
        bytes,
        flops
    };
    
    get_results().push_back(result);
    return result;
}

/**
 * Print results table
 */
inline void print_results(const char* title = "Benchmark Results") {
    auto& results = get_results();
    if (results.empty()) return;
    
    // Get GPU info for peak bandwidth
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    float peak_bw = 2.0f * props.memoryClockRate * (props.memoryBusWidth / 8) / 1e6;
    
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════════╗\n");
    printf("║  %-66s  ║\n", title);
    printf("╠══════════════════════════════════════════════════════════════════════╣\n");
    printf("║ Kernel              │ Time (μs) │ GB/s   │ %%Peak │ Speedup │ σ (μs) ║\n");
    printf("╠══════════════════════════════════════════════════════════════════════╣\n");
    
    float baseline = results[0].mean_ms;
    for (const auto& r : results) {
        float speedup = baseline / r.mean_ms;
        float pct_peak = 100.0f * r.bandwidth_gb_s() / peak_bw;
        printf("║ %-19s │ %9.1f │ %6.1f │ %5.1f%% │ %6.2f× │ %6.1f ║\n",
               r.name,
               r.mean_ms * 1000,  // Convert to μs
               r.bandwidth_gb_s(),
               pct_peak,
               speedup,
               r.stddev_ms * 1000);
    }
    
    printf("╚══════════════════════════════════════════════════════════════════════╝\n");
    printf("  GPU: %s | Peak Bandwidth: %.0f GB/s\n\n", props.name, peak_bw);
}

/**
 * Clear results for next benchmark suite
 */
inline void clear_results() {
    get_results().clear();
}
