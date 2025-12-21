# Day 4: CUDA Benchmark Template

## Overview

A comprehensive benchmark template for comparing multiple CUDA implementations.

## When to Use

- Comparing algorithm implementations
- Performance regression testing
- Generating performance reports
- Optimization experiments

## Structure

```
benchmark-project/
├── CMakeLists.txt
├── build.sh
├── README.md
├── include/
│   └── benchmark/
│       ├── benchmark.h
│       ├── stats.h
│       └── implementations.cuh
├── src/
│   ├── main.cpp
│   ├── benchmark.cpp
│   └── implementations.cu
└── results/
    └── .gitkeep
```

## Features

- Multiple implementation comparison
- Statistical analysis (mean, std, min, max)
- CSV/JSON output
- Warmup runs
- Configurable iterations
