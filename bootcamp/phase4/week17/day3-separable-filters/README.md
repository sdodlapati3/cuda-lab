# Day 3: Separable Filters

## Learning Objectives
- Decompose 2D filters into 1D passes
- Implement horizontal + vertical convolution
- Understand when separability applies

## Key Concepts

### Separable Filters
A 2D filter is separable if it can be written as:
```
Filter2D[y][x] = FilterV[y] * FilterH[x]
```

Examples: Gaussian blur, box blur, Sobel (partially)

### Two-Pass Algorithm
```
Pass 1: Horizontal 1D convolution (N operations per pixel)
Pass 2: Vertical 1D convolution (N operations per pixel)
Total: 2N operations vs N² for direct 2D
```

### Performance Gain
- 3x3: 9 → 6 operations (1.5x)
- 5x5: 25 → 10 operations (2.5x)
- 7x7: 49 → 14 operations (3.5x)
- NxN: N² → 2N operations (N/2 improvement)

## Memory Access Pattern
- Horizontal pass: Coalesced reads
- Vertical pass: Strided reads (less efficient)
- Transpose trick: Transpose → horizontal → transpose
