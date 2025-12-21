# Day 6: Edge Detection

## Learning Objectives
- Implement Sobel edge detection
- Compute gradient magnitude and direction
- Apply non-maximum suppression

## Key Concepts

### Sobel Operator
Computes image gradients in X and Y directions:
```
Gx = [-1  0  1]    Gy = [-1 -2 -1]
     [-2  0  2]         [ 0  0  0]
     [-1  0  1]         [ 1  2  1]

Magnitude = sqrt(Gx² + Gy²)
Direction = atan2(Gy, Gx)
```

### Separable Sobel
Sobel is separable:
```
Sobel_x = [1 2 1]ᵀ * [-1 0 1]
Sobel_y = [-1 0 1]ᵀ * [1 2 1]
```

### Implementation Options
1. Direct 2D convolution (simple)
2. Separable passes (faster)
3. Compute both gradients in one pass (fused)
