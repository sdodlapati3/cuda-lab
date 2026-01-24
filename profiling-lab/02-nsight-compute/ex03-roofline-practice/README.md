# Exercise 03: Roofline Practice

> **Plot kernels on a roofline diagram and identify optimization opportunities**

## 🎯 Objectives

1. Calculate arithmetic intensity for different kernels
2. Generate roofline plots with Nsight Compute
3. Identify memory-bound vs compute-bound kernels
4. Determine optimization strategy based on roofline position

---

## 📋 Prerequisites

- Completed ex01 (memory metrics) and ex02 (compute metrics)
- Understanding of bandwidth and FLOPS concepts
- Access to Nsight Compute with GUI (for visualization)

---

## 🔑 Background: The Roofline Model

### What is Arithmetic Intensity?

**Arithmetic Intensity (AI)** = FLOPs performed / Bytes moved

```
Low AI (< 1):     Memory-bound (bandwidth limited)
Medium AI (1-10): Transition zone
High AI (> 10):   Compute-bound (FLOPS limited)
```

### The Roofline Diagram

```
Performance (GFLOPS)
    ^
1000|                    ______________________ Compute Roof (peak FLOPS)
    |                   /
    |                  /
 100|                 /
    |                /
    |               /  <- Ridge Point
  10|              /
    |             /
    |            / ______________________________ Memory Roof (peak BW × AI)
   1|           /
    |          /
    +----+----+----+----+----+----+----+----+---> Arithmetic Intensity
         0.1  0.5   1    2    5   10   20  50    (FLOP/Byte)
```

### Hardware Limits (Example: A100)

| Limit | Value |
|-------|-------|
| Peak FP32 | 19.5 TFLOPS |
| Peak FP16 | 312 TFLOPS |
| Peak Memory BW | 2039 GB/s |
| Ridge Point (FP32) | 19.5 / 2.039 ≈ 9.6 FLOP/Byte |

---

## 📝 Exercise Steps

### Part 1: Analyze Provided Kernels

We provide three kernels with different arithmetic intensities:

```bash
make all
```

**Kernel A: Vector Add** (very memory-bound)
```cpp
c[i] = a[i] + b[i];
// 1 FLOP, 12 bytes → AI = 0.083
```

**Kernel B: SAXPY with Extra Ops** (moderately memory-bound)
```cpp
c[i] = alpha * a[i] + beta * b[i] + gamma;
// 4 FLOPs, 12 bytes → AI = 0.33
```

**Kernel C: Tiled Matrix Multiply** (can be compute-bound)
```cpp
// With good tiling: ~2*N FLOPs per N/tile_size bytes
// AI scales with tile size
```

### Part 2: Calculate Theoretical AI

For each kernel, calculate:

| Kernel | FLOPs/element | Bytes/element | Theoretical AI |
|--------|---------------|---------------|----------------|
| Vector Add | ? | ? | ? |
| SAXPY Extended | ? | ? | ? |
| MatMul (tile=16) | ? | ? | ? |
| MatMul (tile=32) | ? | ? | ? |

### Part 3: Collect Roofline Data

```bash
# Profile with roofline set
ncu --set roofline -o vectoradd_roofline ./vectoradd
ncu --set roofline -o saxpy_roofline ./saxpy
ncu --set roofline -o matmul_roofline ./matmul

# Open in GUI to see roofline plot
ncu-ui vectoradd_roofline.ncu-rep
```

### Part 4: Extract Metrics

From Nsight Compute, record:

| Kernel | Achieved GFLOPS | Achieved BW (GB/s) | Measured AI | % of Roof |
|--------|-----------------|--------------------| ------------|-----------|
| Vector Add | ? | ? | ? | ? |
| SAXPY Extended | ? | ? | ? | ? |
| MatMul | ? | ? | ? | ? |

**Key metrics to find:**
- `sm__sass_thread_inst_executed_op_fadd_pred_on.sum` - FP32 adds
- `sm__sass_thread_inst_executed_op_fmul_pred_on.sum` - FP32 muls
- `dram__bytes.sum` - DRAM bytes
- `gpu__time_duration.avg` - Kernel duration

### Part 5: Optimization Decisions

Based on roofline position, determine optimization strategy:

| Kernel | Position | Strategy |
|--------|----------|----------|
| Memory-bound | Below memory roof | Improve memory access patterns, reduce bytes moved |
| Compute-bound | Below compute roof | Increase ILP, use faster instructions |
| At roof | On the ceiling | Already optimal for this AI, need algorithmic change |

**Your analysis:**

For each kernel, answer:
1. Is it memory-bound or compute-bound?
2. What % of the relevant roof are we achieving?
3. What optimization would help most?

---

## 💻 Code Files

### vectoradd.cu
```cuda
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
```

### saxpy_extended.cu
```cuda
__global__ void saxpyExtended(float *a, float *b, float *c, 
                               float alpha, float beta, float gamma, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = alpha * a[i] + beta * b[i] + gamma;
    }
}
```

### matmul_tiled.cu
```cuda
// See provided file for full tiled implementation
// Experiment with TILE_SIZE = 16 vs 32
```

---

## 📊 Expected Results

### Vector Add (Memory-Bound)
- **Theoretical AI:** 0.083 FLOP/Byte
- **Expected position:** Far left of roofline, below memory ceiling
- **Target:** >80% of memory bandwidth

### SAXPY Extended (Memory-Bound)
- **Theoretical AI:** 0.33 FLOP/Byte
- **Expected position:** Left side, approaching ridge point
- **Target:** >80% of memory bandwidth

### Matrix Multiply (Varies)
- **Naive AI:** ~0.25 FLOP/Byte (memory-bound)
- **Tiled AI:** ~16+ FLOP/Byte with tile size 16 (compute-bound)
- **Target:** Move from memory-bound to compute-bound region

---

## ✅ Success Criteria

- [ ] Calculated theoretical AI for all kernels
- [ ] Generated roofline plots in Nsight Compute
- [ ] Measured achieved performance vs roof
- [ ] Correctly identified memory-bound vs compute-bound
- [ ] Proposed appropriate optimization strategy for each

---

## 🔑 Key Takeaways

1. **Arithmetic intensity determines the ceiling** - Low AI means memory is your limit
2. **Roofline shows optimization direction** - Below memory roof? Fix memory. Below compute roof? Fix compute.
3. **Tiling increases AI** - More reuse = higher AI = potential to be compute-bound
4. **Can't exceed the roof** - Hardware limits are real; algorithmic changes may be needed

---

## 📚 Reference

### Manual AI Calculation

```python
def calculate_ai(flops_per_iteration, bytes_per_iteration):
    return flops_per_iteration / bytes_per_iteration

# Vector add
ai_vectoradd = calculate_ai(1, 12)  # 1 add, 3 floats × 4 bytes
print(f"Vector Add AI: {ai_vectoradd:.3f}")  # 0.083

# Matrix multiply (naive)
# For C[i,j] += A[i,k] * B[k,j], we do 2*N FLOPs but load/store ~3*N floats
# Actually for naive: 2 FLOPs per element, reading 2N elements for N outputs
ai_matmul_naive = calculate_ai(2, 8)  # 1 mul + 1 add, 2 floats read
print(f"MatMul Naive AI: {ai_matmul_naive:.3f}")  # 0.25
```

### Using ncu for AI Calculation

```bash
# Get FLOPS and bytes
ncu --metrics \
    sm__sass_thread_inst_executed_op_fadd_pred_on.sum,\
    sm__sass_thread_inst_executed_op_fmul_pred_on.sum,\
    sm__sass_thread_inst_executed_op_ffma_pred_on.sum,\
    dram__bytes.sum \
    ./kernel
```
