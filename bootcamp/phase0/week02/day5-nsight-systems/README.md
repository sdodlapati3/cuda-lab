# Day 5: Nsight Systems Profiling

## What You'll Learn

- Profile GPU applications with Nsight Systems
- Read timeline views and understand execution flow
- Identify performance bottlenecks
- Optimize host-device interactions

## Why Nsight Systems?

- **System-wide** profiling (CPU + GPU)
- **Timeline** visualization of execution
- **Dependency** analysis
- **Bottleneck** identification

## Quick Start

```bash
./build.sh

# Profile application
nsys profile -o report ./build/timeline_demo

# View in GUI (requires X11)
nsys-ui report.nsys-rep

# Or generate text report
nsys stats report.nsys-rep
```

## Key Nsight Systems Commands

### Profile Options
```bash
# Basic profiling
nsys profile ./my_app

# Save to specific file
nsys profile -o my_report ./my_app

# Include CUDA API trace
nsys profile --trace=cuda ./my_app

# Full tracing
nsys profile --trace=cuda,nvtx,osrt ./my_app

# Capture GPU metrics
nsys profile --cuda-memory-usage=true ./my_app
```

### Report Generation
```bash
# Summary statistics
nsys stats report.nsys-rep

# Export to SQLite
nsys export --type=sqlite report.nsys-rep

# Export to text
nsys stats -r cuda_gpu_trace report.nsys-rep
```

## Timeline View

```
CPU Thread:
████████░░░░░░░████████░░░░░░░████████   (cudaMemcpy H2D, kernel launch, cudaMemcpy D2H)
         ↓              ↓              ↓
GPU:     ░░░░░███░░░░░░░░░░░███░░░░░░░░░   (kernel execution)
              ↑                   ↑
         kernel1             kernel2
```

## What to Look For

### 1. Serial Bottlenecks
```
CPU: ████████████████████████████████████
GPU: ░░░░░░░░███░░░░░░░░░░░███░░░░░░░░░░░
     ^^^^^^^         ^^^^^^^
     Waiting!        Waiting!
```
**Fix**: Overlap CPU work with GPU execution

### 2. Memory Transfer Dominance
```
H2D: ████████████████████
GPU:                     ███
D2H:                        ██████████████
```
**Fix**: Pin memory, use async transfers, overlap

### 3. Kernel Gaps
```
GPU: ███░░░░░░███░░░░░░███
        ^^^^     ^^^^
        Idle     Idle
```
**Fix**: Use streams, merge kernels, reduce sync points

### 4. Small Kernels
```
GPU: █░█░█░█░█░█░█░█░█░
```
**Fix**: Merge kernels, use persistent kernels

## NVTX Markers

Add custom ranges to your code:

```cpp
#include <nvtx3/nvToolsExt.h>

void my_function() {
    nvtxRangePush("My Function");
    
    // Your code here
    
    nvtxRangePop();
}
```

Or use scoped markers:
```cpp
#include <nvtx3/nvtx3.hpp>

void my_function() {
    nvtx3::scoped_range range{"My Function"};
    // Automatically popped at end of scope
}
```

## Common Bottleneck Patterns

| Pattern | Symptom | Solution |
|---------|---------|----------|
| Sync-heavy | Many gaps in GPU | Reduce cudaDeviceSynchronize |
| Transfer-bound | Long memcpy bars | Pin memory, overlap, reduce transfers |
| Launch overhead | Thin kernel bars, many gaps | Merge kernels, graphs |
| Serial CPU | Long CPU bars before GPU | Overlap CPU/GPU work |

## Exercises

1. Profile the demo and identify the bottleneck
2. Add NVTX markers to your own code
3. Compare optimized vs unoptimized versions
4. Practice reading timeline views
