# Day 1: Convolution Basics

## Learning Objectives
- Implement naive 2D convolution on GPU
- Handle image borders correctly
- Understand memory access patterns

## Key Concepts

### 2D Convolution
Slides a filter (kernel) across the image, computing weighted sums:
```
output[y][x] = Σ Σ image[y+fy][x+fx] * filter[fy][fx]
```

### Border Handling
- **Zero padding**: Treat out-of-bounds as 0
- **Clamp**: Repeat edge pixels
- **Mirror**: Reflect at boundaries

### Naive Implementation Issues
- Each thread reads `filter_size²` pixels
- Neighboring threads read overlapping pixels
- Memory traffic = output_pixels × filter_size² × sizeof(pixel)

## Exercises

1. **Basic Convolution**: Implement 3x3 box blur
2. **Border Handling**: Add clamp and mirror modes
3. **Benchmark**: Compare different filter sizes (3x3, 5x5, 7x7)

## Performance Analysis

For 1920x1080 image with 3x3 filter:
- Naive: Each pixel read up to 9 times
- Expected bandwidth: ~10-20% of peak (memory-bound with inefficient access)

Tomorrow we'll add shared memory to dramatically improve this.
