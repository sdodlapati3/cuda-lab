#pragma once
/**
 * benchmark.h - Benchmark framework
 */

#include <string>
#include <vector>
#include <functional>

namespace bench {

struct Result {
    std::string name;
    double mean_ms;
    double std_ms;
    double min_ms;
    double max_ms;
    int runs;
    
    double throughput_gbps = 0.0;  // Optional
};

class Benchmark {
public:
    Benchmark(const std::string& name, int warmup = 3, int runs = 10);
    
    // Run a benchmark
    Result run(const std::string& impl_name,
               std::function<float()> kernel_fn);  // Returns time in ms
    
    // Store result
    void add_result(const Result& r);
    
    // Print results table
    void print_results() const;
    
    // Export to CSV
    void export_csv(const std::string& filename) const;
    
    // Export to JSON
    void export_json(const std::string& filename) const;
    
private:
    std::string name_;
    int warmup_;
    int runs_;
    std::vector<Result> results_;
};

}  // namespace bench
