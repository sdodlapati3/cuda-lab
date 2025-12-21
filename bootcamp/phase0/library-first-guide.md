# Library-First Development Guide

> **Core Principle:** The best kernel is one you don't have to write.

## The Decision Framework

Before writing any custom CUDA kernel, ask yourself:

```
┌─────────────────────────────────────────────────────────────────┐
│                    DO I NEED A CUSTOM KERNEL?                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │ Does cuBLAS/cuDNN solve it?   │
              └───────────────────────────────┘
                     │              │
                    YES             NO
                     │              │
                     ▼              ▼
              ┌──────────┐   ┌──────────────────────┐
              │ USE IT   │   │ Does CUTLASS/CUB     │
              │          │   │ solve it?            │
              └──────────┘   └──────────────────────┘
                                   │          │
                                  YES         NO
                                   │          │
                                   ▼          ▼
                            ┌──────────┐ ┌────────────────────┐
                            │ USE IT   │ │ Can Triton do it   │
                            │          │ │ 90% as fast?       │
                            └──────────┘ └────────────────────┘
                                              │          │
                                             YES         NO
                                              │          │
                                              ▼          ▼
                                       ┌──────────┐ ┌──────────────┐
                                       │ USE      │ │ WRITE CUSTOM │
                                       │ TRITON   │ │ CUDA KERNEL  │
                                       └──────────┘ └──────────────┘
```

## When Libraries Win

### cuBLAS (Matrix Operations)
```cpp
// DON'T: Write your own GEMM
// DO: Use cuBLAS
#include <cublas_v2.h>

cublasHandle_t handle;
cublasCreate(&handle);
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K, &alpha, A, M, B, K, &beta, C, M);
```

**When cuBLAS is right:**
- Any GEMM (matrix multiply)
- Batched matrix operations
- BLAS Level 1/2/3 operations
- You need FP64, FP32, FP16, INT8, etc.

**When to consider custom:**
- You need fusion (GEMM + bias + activation)
- Very small matrices (launch overhead dominates)
- Unusual memory layouts

### cuDNN (Deep Learning Primitives)
```cpp
// DON'T: Write your own convolution
// DO: Use cuDNN
cudnnConvolutionForward(handle, &alpha, xDesc, x,
                        wDesc, w, convDesc, algo,
                        workspace, workspaceSize,
                        &beta, yDesc, y);
```

**cuDNN covers:**
- Convolutions (forward, backward)
- Pooling operations
- Normalization (BatchNorm, LayerNorm, etc.)
- Activations
- RNNs and attention (newer versions)

### CUB (Parallel Primitives)
```cpp
// DON'T: Write your own reduction
// DO: Use CUB
#include <cub/cub.cuh>

cub::DeviceReduce::Sum(d_temp, temp_bytes, d_input, d_output, n);
```

**CUB covers:**
- Reductions (sum, min, max, argmax)
- Scans (prefix sums)
- Sorting
- Histogram
- Select/unique/partition

### CUTLASS (Advanced GEMM)
```cpp
// For fused epilogues that cuBLAS can't do
using Gemm = cutlass::gemm::device::Gemm<
    float, LayoutA, float, LayoutB, float, LayoutC,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    ThreadblockShape, WarpShape, InstructionShape,
    EpilogueFused  // Your custom epilogue!
>;
```

**CUTLASS when:**
- Need epilogue fusion (bias + activation + residual)
- Specific tile sizes for your problem
- Mixed precision control
- Need to understand GEMM internals

### Triton (Productivity Layer)
```python
# 90% of cuBLAS speed with 10% of the code
@triton.jit
def matmul_kernel(A, B, C, M, N, K, ...):
    # Triton handles tiling, shared memory, etc.
    ...
```

**Triton when:**
- Rapid prototyping
- Fused operations
- Auto-tuning needed
- Team doesn't have CUDA expertise

## The Library Hierarchy

| Priority | Library | Best For |
|----------|---------|----------|
| 1 | cuBLAS | All matrix ops |
| 2 | cuDNN | All DL primitives |
| 3 | CUB | Parallel primitives |
| 4 | Thrust | STL-like GPU ops |
| 5 | CUTLASS | Custom GEMM/fusion |
| 6 | Triton | Rapid kernel dev |
| 7 | Custom CUDA | Everything else |

## Benchmark Before You Customize

Always benchmark the library solution first:

```python
import torch
import time

# Library baseline
x = torch.randn(4096, 4096, device='cuda')
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    y = torch.softmax(x, dim=-1)
torch.cuda.synchronize()
library_time = (time.time() - start) / 100
print(f"Library: {library_time*1000:.3f} ms")

# Your custom kernel must beat this!
```

## When Custom Kernels Win

Write custom CUDA when:

1. **Fusion opportunities** - Multiple ops can share memory traffic
2. **Memory layout** - Libraries assume standard layouts
3. **Problem size** - Very small or unusual dimensions
4. **Novel algorithms** - FlashAttention wasn't in cuDNN at first
5. **Research** - You're exploring new ideas

## Checklist Before Writing Custom CUDA

- [ ] Benchmarked library solutions
- [ ] Documented why library isn't sufficient
- [ ] Calculated theoretical speedup potential
- [ ] Considered Triton as middle ground
- [ ] Justified engineering time investment

## Quick Reference: Library APIs

### cuBLAS
```bash
# Documentation
https://docs.nvidia.com/cuda/cublas/

# Link
-lcublas
```

### cuDNN
```bash
# Documentation
https://docs.nvidia.com/deeplearning/cudnn/

# Link
-lcudnn
```

### CUB
```bash
# Documentation
https://nvlabs.github.io/cub/

# Header-only (part of CUDA Toolkit)
#include <cub/cub.cuh>
```

### CUTLASS
```bash
# Documentation
https://github.com/NVIDIA/cutlass

# Header-only
#include <cutlass/cutlass.h>
```

## Real Example: Softmax

### Step 1: PyTorch (baseline)
```python
y = torch.softmax(x, dim=-1)  # ~0.5ms for 4096x4096
```

### Step 2: Check if cuDNN has it
```cpp
// cuDNN softmax exists but may not be faster than PyTorch
cudnnSoftmaxForward(handle, CUDNN_SOFTMAX_ACCURATE, 
                    CUDNN_SOFTMAX_MODE_INSTANCE, ...)
```

### Step 3: Consider fused patterns
```
// If you need: softmax + dropout + residual
// Then: Custom kernel makes sense (no library does this)
```

### Step 4: Try Triton first
```python
@triton.jit
def fused_softmax_dropout_residual(...):
    ...
```

### Step 5: Custom CUDA only if Triton isn't fast enough
```cuda
__global__ void fused_softmax_dropout_residual_kernel(...) {
    // Only if you've proven library + Triton can't do it
}
```

---

> **Remember:** Library authors have years of optimization experience.
> Your custom kernel needs to deliver significant value to justify the investment.
