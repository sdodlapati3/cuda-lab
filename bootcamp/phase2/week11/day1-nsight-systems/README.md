# Day 1: Nsight Systems Overview

## Learning Objectives

- Use nsys for timeline profiling
- Identify kernel overlap and gaps
- Analyze memory transfer patterns

## Key Concepts

### Timeline Analysis

Nsight Systems shows:
- Kernel execution times
- Memory transfer (H2D, D2H, D2D)
- CUDA API calls
- CPU activity and GPU utilization

### Basic Commands

```bash
# Generate profile
nsys profile -o report ./app

# View in GUI
nsight-sys report.nsys-rep

# Quick stats
nsys stats report.nsys-rep
```

### What to Look For

| Pattern | Issue | Solution |
|---------|-------|----------|
| Gaps between kernels | Launch overhead | Kernel fusion, streams |
| Long H2D/D2H | Transfer bound | Async transfers, pinned memory |
| Short kernels | Low GPU util | Kernel fusion |
| Serialized ops | Missing overlap | Streams |

## Build & Run

```bash
./build.sh
./build/timeline_demo
nsys profile -o report ./build/timeline_demo
```
