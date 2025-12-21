# Day 4: Benchmark Harness

## Learning Objectives

- Build a reusable benchmarking framework
- Use CUDA Events for accurate GPU timing
- Calculate and report GB/s and GFLOPS
- Understand warmup and statistical validity

## Why Not Just Use `time`?

Wall-clock timing (`time ./mykernel`) includes:
- CUDA initialization (can be 100ms+)
- Memory allocation
- Host-device transfers
- Kernel launch overhead
- Everything else

**CUDA Events** measure only GPU execution time, with microsecond precision.

## The Timing Pattern

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
my_kernel<<<blocks, threads>>>(...);
cudaEventRecord(stop);

cudaEventSynchronize(stop);  // Wait for kernel

float ms;
cudaEventElapsedTime(&ms, start, stop);
```

## Key Files

```
day4-benchmark-harness/
├── CMakeLists.txt
├── include/
│   ├── cuda_timer.cuh      # Event-based timing class
│   └── benchmark.cuh       # Benchmark config + runner
├── src/
│   ├── benchmark.cu        # Main benchmark driver
│   └── kernels.cu          # Example kernels to benchmark
└── README.md
```

## Quick Start

```bash
mkdir build && cd build
cmake -G Ninja ..
ninja
./benchmark
```

## What Good Benchmarking Requires

### 1. Warmup Runs
```cpp
// First few runs are slow (GPU frequency scaling, cache warmup)
for (int i = 0; i < WARMUP; i++) {
    my_kernel<<<...>>>(...);
}
cudaDeviceSynchronize();
// NOW start timing
```

### 2. Multiple Iterations
```cpp
const int ITERATIONS = 100;
// Take average, report min/max/stddev
```

### 3. Prevent Compiler Optimization
```cpp
// Make sure results are "used" so compiler doesn't optimize away
volatile float result = output[0];
```

### 4. Measure the Right Thing
- **Bandwidth-bound kernels:** Report GB/s
- **Compute-bound kernels:** Report GFLOPS
- **Both:** Report arithmetic intensity + roofline position

## Output Format

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

## Exercises

1. Add stddev and percentiles to timing output
2. Create a CSV export for plotting
3. Add automatic regression testing (fail if slower than baseline)
4. Compare your harness output with `ncu` profiler output
