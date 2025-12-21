# Day 3: Compute Throughput

## Learning Objectives

- Measure peak compute (FLOPS) throughput
- Understand FMA and instruction throughput
- Establish compute ceiling for roofline

## Key Concepts

### Peak FLOPS Calculation

```
Peak FP32 FLOPS = CUDA Cores × Clock × 2 (FMA)

A100:
  6912 FP32 cores × 1.41 GHz × 2 = 19.5 TFLOPS
```

### Measuring Achieved FLOPS

Use compute-intensive kernels with known FLOP counts:
- FMA-heavy loops
- Matrix multiply (compute portion)
- Synthetic benchmarks

### FMA (Fused Multiply-Add)

```cpp
c = a * b + c;  // 1 FMA = 2 FLOPs
```

FMA counts as 2 FLOPs but executes as 1 instruction.

## Build & Run

```bash
./build.sh
./build/flops_test
./build/compute_bound
```
