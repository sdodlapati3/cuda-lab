# Week 12: Latency Hiding

## Theme: Mastering GPU Latency and Throughput

The final week focuses on understanding and exploiting latency hiding -
the fundamental mechanism that allows GPUs to achieve high throughput.

## Daily Breakdown

| Day | Topic | Key Concept |
|-----|-------|-------------|
| 1 | Understanding Latency | Memory and instruction latencies |
| 2 | Warp Scheduling | How SMs hide latency |
| 3 | ILP and TLP | Instruction vs Thread parallelism |
| 4 | Memory Latency Hiding | Prefetching, async copies |
| 5 | Compute Latency Hiding | Pipelining, unrolling |
| 6 | Putting It All Together | Complete mental model |

## Key Mental Models

### Latency vs Throughput
```
Latency: Time for ONE operation (e.g., 400 cycles for memory)
Throughput: Operations per second (e.g., GB/s)

GPU hides latency with MASSIVE parallelism:
  While some warps wait, others execute
```

### Little's Law
```
Parallelism = Latency × Throughput

To achieve peak throughput with high latency:
  Need enough parallel work to stay busy
```

### Hiding Mechanisms
```
1. Thread-Level Parallelism (TLP): Many warps per SM
2. Instruction-Level Parallelism (ILP): Independent ops per thread
3. Memory-Level Parallelism (MLP): Multiple outstanding loads
```

## Prerequisites
- Weeks 9-11: Performance models and profiling

## Building
Each day has executable examples:
```bash
cd dayX-topic
./build.sh
./build/executable_name
```
