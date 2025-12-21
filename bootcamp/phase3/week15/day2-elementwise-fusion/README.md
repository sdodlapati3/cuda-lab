# Day 2: Element-wise Fusion

## Learning Objectives

- Fuse multiple element-wise operations
- Implement activation function chains
- Measure memory bandwidth savings

## Key Concepts

### Element-wise Operations

Operations that process each element independently:
- Add, subtract, multiply
- Activation functions (ReLU, sigmoid, tanh)
- Normalization
- Elementwise comparisons

### Fusion Pattern

```cpp
// Fuse: y = sigmoid(relu(x * scale + bias))
__global__ void fused_activations(const float* x, float scale, 
                                   const float* bias, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx] * scale + bias[idx];
        val = fmaxf(0.0f, val);  // ReLU
        y[idx] = 1.0f / (1.0f + expf(-val));  // Sigmoid
    }
}
```

### When to Fuse Element-wise Ops

✅ **Always fuse** consecutive element-wise ops:
- Same input/output dimensions
- No inter-element dependencies
- Reduces memory traffic proportionally

## Build & Run

```bash
./build.sh
./build/elementwise_fusion
```
