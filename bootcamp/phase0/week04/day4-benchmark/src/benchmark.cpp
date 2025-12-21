/**
 * benchmark.cpp - Benchmark framework implementation
 */

#include "benchmark/benchmark.h"
#include "benchmark/stats.h"
#include <cstdio>
#include <fstream>

namespace bench {

Benchmark::Benchmark(const std::string& name, int warmup, int runs)
    : name_(name), warmup_(warmup), runs_(runs) {}

Result Benchmark::run(const std::string& impl_name,
                      std::function<float()> kernel_fn) {
    printf("Benchmarking: %s\n", impl_name.c_str());
    
    // Warmup
    for (int i = 0; i < warmup_; i++) {
        kernel_fn();
    }
    
    // Timed runs
    std::vector<double> times;
    times.reserve(runs_);
    
    for (int i = 0; i < runs_; i++) {
        float ms = kernel_fn();
        times.push_back(ms);
    }
    
    Stats s = Stats::compute(times);
    
    Result r;
    r.name = impl_name;
    r.mean_ms = s.mean;
    r.std_ms = s.std_dev;
    r.min_ms = s.min;
    r.max_ms = s.max;
    r.runs = runs_;
    
    printf("  Mean: %.4f ms (std: %.4f)\n", s.mean, s.std_dev);
    
    return r;
}

void Benchmark::add_result(const Result& r) {
    results_.push_back(r);
}

void Benchmark::print_results() const {
    printf("=== Benchmark Results: %s ===\n", name_.c_str());
    printf("%-25s %10s %10s %10s %10s %12s\n",
           "Implementation", "Mean(ms)", "Std(ms)", "Min(ms)", "Max(ms)", "GB/s");
    printf("%s\n", std::string(82, '-').c_str());
    
    for (const auto& r : results_) {
        printf("%-25s %10.4f %10.4f %10.4f %10.4f %12.2f\n",
               r.name.c_str(), r.mean_ms, r.std_ms, r.min_ms, r.max_ms,
               r.throughput_gbps);
    }
}

void Benchmark::export_csv(const std::string& filename) const {
    std::ofstream f(filename);
    f << "implementation,mean_ms,std_ms,min_ms,max_ms,throughput_gbps,runs\n";
    for (const auto& r : results_) {
        f << r.name << ","
          << r.mean_ms << ","
          << r.std_ms << ","
          << r.min_ms << ","
          << r.max_ms << ","
          << r.throughput_gbps << ","
          << r.runs << "\n";
    }
}

void Benchmark::export_json(const std::string& filename) const {
    std::ofstream f(filename);
    f << "{\n";
    f << "  \"benchmark\": \"" << name_ << "\",\n";
    f << "  \"results\": [\n";
    for (size_t i = 0; i < results_.size(); i++) {
        const auto& r = results_[i];
        f << "    {\n";
        f << "      \"name\": \"" << r.name << "\",\n";
        f << "      \"mean_ms\": " << r.mean_ms << ",\n";
        f << "      \"std_ms\": " << r.std_ms << ",\n";
        f << "      \"min_ms\": " << r.min_ms << ",\n";
        f << "      \"max_ms\": " << r.max_ms << ",\n";
        f << "      \"throughput_gbps\": " << r.throughput_gbps << ",\n";
        f << "      \"runs\": " << r.runs << "\n";
        f << "    }" << (i < results_.size() - 1 ? "," : "") << "\n";
    }
    f << "  ]\n";
    f << "}\n";
}

}  // namespace bench
