# SASS Analysis Notes

This file is for your annotated SASS analysis.

## simple_add Kernel

### Source Code
```cpp
__global__ void simple_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
```

### SASS (sm_80)

```sass
// YOUR ANNOTATED SASS GOES HERE
// Run: ./scripts/extract_sass.sh 80
// Then paste and annotate the output

// Example annotation format:
//
// S2R R0, SR_CTAID.X          // R0 = blockIdx.x
// S2R R3, SR_TID.X            // R3 = threadIdx.x  
// IMAD R0, R0, c[0x0][0x0], R3  // R0 = blockIdx.x * blockDim.x + threadIdx.x
//                              // Note: blockDim.x loaded from constant memory
```

### Key Observations

1. **Register allocation:**
   - TODO: How many registers does this kernel use?
   - TODO: What's the max occupancy given this register count?

2. **Memory access pattern:**
   - TODO: Are loads coalesced?
   - TODO: What memory instruction variants are used? (LDG.E.SYS vs LDG.E.128)

3. **Control flow:**
   - TODO: Is the bounds check a branch or predicated?
   - TODO: Does this cause divergence?

## fma_kernel

### Source Code
```cpp
d[idx] = a[idx] * b[idx] + c[idx] * alpha;
```

### Expected SASS
This should compile to FFMA (Fused Floating-point Multiply-Add) instructions.

TODO: Verify this by extracting SASS

## reduce_sum Kernel

### Source Code
Uses shared memory and __syncthreads()

### Key Things to Find
1. LDS/STS instructions (shared memory)
2. BAR.SYNC instruction (__syncthreads)
3. The reduction loop structure

TODO: Annotate the reduction loop SASS
