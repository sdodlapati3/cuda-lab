#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <functional>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>

// ============================================================================
// CUDA Error Checking
// ============================================================================
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// Timer (CUDA Events)
// ============================================================================
class CudaTimer {
public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    
    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    
    void start(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(start_, stream));
    }
    
    void stop(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(stop_, stream));
    }
    
    float elapsed_ms() {
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_, stop_;
};

// ============================================================================
// Benchmark Configuration
// ============================================================================
struct BenchmarkConfig {
    size_t warmup_iterations = 10;
    size_t timed_iterations = 100;
    size_t data_size = 1 << 24;  // Elements
    size_t element_bytes = sizeof(float);
    size_t flops_per_element = 1;
    bool verify_correctness = true;
};

// ============================================================================
// Benchmark Result
// ============================================================================
struct BenchmarkResult {
    std::string name;
    double mean_ms;
    double stddev_ms;
    double min_ms;
    double max_ms;
    double p50_ms;
    double p99_ms;
    double bandwidth_gb_s;
    double tflops;
    double percent_peak_bandwidth;
    double percent_peak_compute;
    bool correct;
};

// ============================================================================
// Device Info
// ============================================================================
struct DeviceInfo {
    std::string name;
    double peak_bandwidth_gb_s;
    double peak_tflops;
    int sm_count;
    int compute_capability;
    
    static DeviceInfo query(int device = 0) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        
        DeviceInfo info;
        info.name = prop.name;
        info.peak_bandwidth_gb_s = 2.0 * prop.memoryClockRate * 
                                   (prop.memoryBusWidth / 8) / 1.0e6;
        info.sm_count = prop.multiProcessorCount;
        info.compute_capability = prop.major * 10 + prop.minor;
        
        // Estimate peak TFLOPS (FP32, varies by architecture)
        int cores_per_sm = 128;  // Ampere default
        if (prop.major == 7) cores_per_sm = 64;  // Volta
        if (prop.major == 8 && prop.minor == 6) cores_per_sm = 128;  // Ampere
        if (prop.major == 8 && prop.minor == 9) cores_per_sm = 128;  // Ada
        if (prop.major == 9) cores_per_sm = 128;  // Hopper
        
        info.peak_tflops = 2.0 * prop.clockRate * 1e-6 * 
                           prop.multiProcessorCount * cores_per_sm / 1e3;
        
        return info;
    }
};

// ============================================================================
// Statistics
// ============================================================================
inline std::vector<double> compute_stats(std::vector<double>& times) {
    std::sort(times.begin(), times.end());
    
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double mean = sum / times.size();
    
    double sq_sum = 0;
    for (auto t : times) sq_sum += (t - mean) * (t - mean);
    double stddev = std::sqrt(sq_sum / times.size());
    
    double min_val = times.front();
    double max_val = times.back();
    double p50 = times[times.size() / 2];
    double p99 = times[static_cast<size_t>(times.size() * 0.99)];
    
    return {mean, stddev, min_val, max_val, p50, p99};
}

// ============================================================================
// Benchmark Runner
// ============================================================================
template<typename KernelFunc>
BenchmarkResult run_benchmark(
    const std::string& name,
    KernelFunc kernel,
    const BenchmarkConfig& config,
    const DeviceInfo& device
) {
    CudaTimer timer;
    std::vector<double> times;
    times.reserve(config.timed_iterations);
    
    // Warmup
    for (size_t i = 0; i < config.warmup_iterations; ++i) {
        kernel();
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Timed runs
    for (size_t i = 0; i < config.timed_iterations; ++i) {
        timer.start();
        kernel();
        timer.stop();
        times.push_back(timer.elapsed_ms());
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Calculate statistics
    auto stats = compute_stats(times);
    
    BenchmarkResult result;
    result.name = name;
    result.mean_ms = stats[0];
    result.stddev_ms = stats[1];
    result.min_ms = stats[2];
    result.max_ms = stats[3];
    result.p50_ms = stats[4];
    result.p99_ms = stats[5];
    
    // Calculate bandwidth (assuming read + write)
    double bytes_transferred = 2.0 * config.data_size * config.element_bytes;
    result.bandwidth_gb_s = bytes_transferred / (result.mean_ms * 1e6);
    result.percent_peak_bandwidth = 100.0 * result.bandwidth_gb_s / 
                                    device.peak_bandwidth_gb_s;
    
    // Calculate TFLOPS
    double total_flops = config.data_size * config.flops_per_element;
    result.tflops = total_flops / (result.mean_ms * 1e9);
    result.percent_peak_compute = 100.0 * result.tflops / device.peak_tflops;
    
    result.correct = true;  // Override with verification if needed
    
    return result;
}

// ============================================================================
// Pretty Printing
// ============================================================================
inline void print_results(
    const std::vector<BenchmarkResult>& results,
    const DeviceInfo& device
) {
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  Device: " << std::left << std::setw(65) << device.name << "║\n";
    std::cout << "║  Peak BW: " << std::fixed << std::setprecision(1) 
              << std::setw(7) << device.peak_bandwidth_gb_s << " GB/s"
              << "   Peak Compute: " << std::setw(7) << device.peak_tflops << " TFLOPS"
              << std::setw(16) << "" << "║\n";
    std::cout << "╠═══════════════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║ Kernel              │ Time (μs) │ GB/s   │ % Peak │ TFLOPS │ Correct      ║\n";
    std::cout << "╠═══════════════════════════════════════════════════════════════════════════╣\n";
    
    for (const auto& r : results) {
        std::cout << "║ " << std::left << std::setw(19) << r.name.substr(0, 19)
                  << " │ " << std::right << std::setw(9) << std::fixed 
                  << std::setprecision(1) << (r.mean_ms * 1000)
                  << " │ " << std::setw(6) << std::setprecision(1) << r.bandwidth_gb_s
                  << " │ " << std::setw(6) << std::setprecision(1) << r.percent_peak_bandwidth << "%"
                  << " │ " << std::setw(6) << std::setprecision(3) << r.tflops
                  << " │ " << std::setw(12) << (r.correct ? "✓" : "✗")
                  << " ║\n";
    }
    
    std::cout << "╚═══════════════════════════════════════════════════════════════════════════╝\n";
}

// ============================================================================
// Roofline Data Export (for plotting)
// ============================================================================
inline void export_roofline_csv(
    const std::vector<BenchmarkResult>& results,
    const DeviceInfo& device,
    const std::string& filename
) {
    std::ofstream f(filename);
    f << "name,arithmetic_intensity,achieved_performance,peak_bandwidth,peak_compute\n";
    
    for (const auto& r : results) {
        // AI = FLOPS / Bytes
        double ai = r.tflops * 1e12 / (r.bandwidth_gb_s * 1e9);
        f << r.name << "," << ai << "," << r.tflops << ","
          << device.peak_bandwidth_gb_s << "," << device.peak_tflops << "\n";
    }
}
