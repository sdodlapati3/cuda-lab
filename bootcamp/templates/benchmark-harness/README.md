# Benchmark Harness Template

A clean, reproducible benchmarking framework for CUDA kernels.

## Usage

```cpp
#include "benchmark.cuh"

int main() {
    BenchmarkConfig config;
    config.warmup_iterations = 10;
    config.timed_iterations = 100;
    config.data_size = 1 << 24;  // 16M elements
    
    // Register your kernels
    benchmark("naive_reduction", naive_reduction, config);
    benchmark("warp_reduction", warp_reduction, config);
    benchmark("cub_reduction", cub_reduction, config);
    
    // Print comparison table
    print_results();
    return 0;
}
```

## Output

```
╔══════════════════════════════════════════════════════════════════╗
║                    REDUCTION BENCHMARK (16M float)               ║
╠══════════════════════════════════════════════════════════════════╣
║ Kernel              │ Time (μs) │ GB/s   │ % Peak │ Speedup     ║
╠══════════════════════════════════════════════════════════════════╣
║ naive_reduction     │   524.3   │  122.1 │  12.7% │ 1.00×       ║
║ warp_reduction      │   203.7   │  314.2 │  32.7% │ 2.57×       ║
║ cub_reduction       │    82.1   │  779.8 │  81.2% │ 6.39×       ║
╚══════════════════════════════════════════════════════════════════╝
```

## Files

```
benchmark_harness/
├── CMakeLists.txt
├── include/
│   ├── benchmark.cuh       # Main benchmark framework
│   ├── timer.cuh           # CUDA event-based timing
│   ├── roofline.cuh        # Roofline plotting utilities
│   └── statistics.cuh      # Mean, stddev, percentiles
├── src/
│   └── benchmark.cu
└── examples/
    ├── reduction_bench.cu
    └── transpose_bench.cu
```
