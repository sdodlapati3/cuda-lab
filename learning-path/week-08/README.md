# Week 8: Profiling & Analysis

## Overview
This week focuses on using NVIDIA's profiling tools to understand GPU performance, identify bottlenecks, and guide optimization efforts. Profiling is essential for writing efficient CUDA code.

## Learning Goals
By the end of this week, you will:
- Use Nsight Compute for kernel analysis
- Use Nsight Systems for application profiling
- Understand the roofline model
- Identify compute vs memory bottlenecks
- Apply profile-guided optimization

## Prerequisites
- Weeks 1-7 completed
- CUDA toolkit installed with Nsight tools
- Basic understanding of GPU architecture

## Daily Schedule

### Day 1: Nsight Compute Basics
- Introduction to kernel profiling
- Key performance metrics
- Using ncu command line
- Reading section reports

### Day 2: Roofline Analysis
- Understanding the roofline model
- Arithmetic intensity calculation
- Identifying performance limiters
- Compute-bound vs memory-bound

### Day 3: Nsight Systems
- Application-level profiling
- CPU-GPU timeline analysis
- Identifying synchronization issues
- Optimizing data transfers

### Day 4: Bottleneck Analysis
- Systematic bottleneck identification
- Memory bottlenecks (bandwidth, latency)
- Compute bottlenecks (throughput, latency)
- Optimization strategies

### Day 5: Practice & Quiz
- Week 8 checkpoint quiz
- Profile-guided optimization exercise
- Real-world profiling scenarios

## Key Concepts

### Nsight Compute Metrics
```
Compute Metrics:
- SM Throughput (%)
- Compute (SM) [%] utilization
- Issue Slot Utilization

Memory Metrics:
- Memory Throughput (%)
- L1/TEX Cache Hit Rate
- L2 Cache Hit Rate
- DRAM Throughput
```

### Roofline Model
```
Performance (GFLOPS) = min(Peak Compute, Peak Bandwidth × Arithmetic Intensity)

Arithmetic Intensity = FLOPs / Bytes
  - Low AI (<10): Memory-bound
  - High AI (>10): Compute-bound
```

### Common Bottlenecks
```
Memory-bound:
- Low arithmetic intensity
- High memory traffic
- Poor coalescing
- Cache thrashing

Compute-bound:
- Low occupancy
- Instruction latency
- Control divergence
- Register spilling
```

## Performance Targets
- Achieve >50% peak memory bandwidth
- Achieve >50% peak compute throughput
- Minimize idle time in timeline

## Resources
- Nsight Compute Documentation
- Nsight Systems User Guide
- CUDA Best Practices Guide - Profiling chapter
