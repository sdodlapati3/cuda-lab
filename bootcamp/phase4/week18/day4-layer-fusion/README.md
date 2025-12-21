# Day 4: Layer Fusion

## Learning Objectives
- Fuse consecutive operations for inference
- Reduce kernel launch overhead
- Optimize memory traffic

## Key Concepts

### Common Fusion Patterns
```
Linear + Bias + ReLU → FusedLinear
Conv + BatchNorm + ReLU → FusedConv
MatMul + Scale + Softmax → FusedAttentionScore
```

### Benefits
1. Reduced kernel launches
2. Intermediate data stays in registers/cache
3. Less global memory traffic
