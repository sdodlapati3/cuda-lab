# Week 7: Memory Optimization Deep Dive

## Overview
This week focuses on advanced memory optimization techniques that are critical for achieving peak GPU performance. We cover occupancy, register pressure, cache utilization, and unified memory.

## Learning Goals
By the end of this week, you will:
- Understand and calculate GPU occupancy
- Analyze and reduce register pressure
- Optimize L1/L2 cache usage
- Implement unified memory effectively
- Use memory access patterns for peak bandwidth

## Prerequisites
- Weeks 1-6 completed
- Understanding of shared memory
- Profiler basics (will be expanded in Week 8)

## Daily Schedule

### Day 1: Occupancy Analysis
- What is occupancy and why it matters
- Occupancy limiters: threads, registers, shared memory
- Calculating theoretical occupancy
- CUDA Occupancy Calculator

### Day 2: Register Optimization
- Register usage and spilling
- Launch bounds optimization
- Local memory implications
- Register tiling techniques

### Day 3: Cache Optimization
- L1 and L2 cache architecture
- Cache configuration options
- Memory access coalescing deep dive
- Texture and constant memory caches

### Day 4: Unified Memory
- Unified memory programming model
- Page migration mechanics
- Memory hints and prefetching
- When to use unified vs explicit memory

### Day 5: Practice & Quiz
- Week 7 checkpoint quiz
- Optimization case studies
- Profile-driven optimization

## Key Concepts

### Occupancy
```
Occupancy = Active Warps per SM / Maximum Warps per SM

Limiters:
- Threads per block (max 1024)
- Registers per thread (64K total per SM)
- Shared memory per block (48-164KB per SM)
```

### Register Usage
```
Registers needed = num_threads × regs_per_thread
Too many registers → spilling to local memory → slow
Too few threads → low occupancy → underutilization
```

### Cache Hierarchy
```
L1 Cache (per SM): ~128KB, configurable with shared memory
L2 Cache (shared): 1.5-40MB depending on GPU
Texture Cache: 48KB per SM, read-only, 2D spatial locality
Constant Cache: 8KB per SM, broadcast to warp
```

## Performance Targets
- Achieve >50% theoretical occupancy
- Eliminate register spilling
- Maintain >80% memory bandwidth utilization
- Reduce memory traffic through caching

## Resources
- CUDA Occupancy Calculator (Excel/spreadsheet)
- NVIDIA Nsight Compute occupancy section
- CUDA Best Practices Guide - Memory chapter
