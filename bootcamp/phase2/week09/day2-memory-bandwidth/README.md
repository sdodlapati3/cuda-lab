# Day 2: Memory Bandwidth Measurement

## Learning Objectives

- Measure peak memory bandwidth empirically
- Understand theoretical vs achieved bandwidth
- Learn bandwidth measurement techniques

## Key Concepts

### Theoretical Peak Bandwidth

```
Peak BW = Memory Clock × Bus Width × 2 (DDR) × ECC factor

A100 80GB:
  Memory Clock: 1.215 GHz
  Bus Width: 5120 bits
  Peak = 1.215 × (5120/8) × 2 = 2039 GB/s (with ECC ~1935 GB/s)
```

### Achieved Bandwidth

Real kernels achieve 70-85% of peak due to:
- ECC overhead
- Refresh cycles
- Protocol overhead
- Access patterns

### How to Measure

**Method 1:** Copy kernel (cudaMemcpy)
```cpp
// Measures D2D copy bandwidth
cudaMemcpy(d_dst, d_src, size, cudaMemcpyDeviceToDevice);
```

**Method 2:** Simple read/write kernel
```cpp
// Measures actual kernel bandwidth
out[idx] = in[idx];  // 8 bytes per thread
```

**Method 3:** Read-only kernel (most accurate for reads)
```cpp
sum += in[idx];  // 4 bytes per thread, minimal compute
```

## Exercises

1. Measure your GPU's peak bandwidth
2. Compare cudaMemcpy vs custom kernel
3. Find practical bandwidth ceiling for your hardware

## Build & Run

```bash
./build.sh
./build/bandwidth_test
```
