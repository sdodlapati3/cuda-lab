# Week 17: Image Processing

## Overview

GPU-accelerated image processing is one of the most practical applications of CUDA. This week covers fundamental algorithms from convolution to edge detection.

## Daily Schedule

| Day | Topic | Key Learning |
|-----|-------|--------------|
| 1 | Convolution Basics | 2D convolution, border handling |
| 2 | Tiled Convolution | Shared memory for halos |
| 3 | Separable Filters | Decompose 2D filters |
| 4 | Histogram & Equalization | Atomic histograms, CDF |
| 5 | Image Resizing | Interpolation methods |
| 6 | Edge Detection | Sobel operator |

## Key Concepts

### Image as 2D Array
```
Image[height][width] in row-major order
Pixel access: image[y * width + x]
For RGB: image[(y * width + x) * 3 + channel]
```

### Convolution Operation
```
For each output pixel (x, y):
    sum = 0
    For each filter position (fx, fy):
        sum += image[y+fy][x+fx] * filter[fy][fx]
    output[y][x] = sum
```

### Memory Access Pattern
```
Naive: Each thread reads filter_size² pixels
       Massive redundancy (neighbors overlap)

Tiled: Load tile + halo into shared memory
       Each pixel loaded once per tile
```

## Performance Targets

| Kernel | Target | Metric |
|--------|--------|--------|
| 3x3 Convolution | >100 GB/s | Memory bandwidth |
| Histogram | >50 GB/s | With atomics |
| Bilinear Resize | >80 GB/s | Texture vs manual |

## NPP Reference

NVIDIA Performance Primitives (NPP) provides optimized image functions:
- `nppiFilter` - Convolution
- `nppiHistogram` - Histogram computation
- `nppiResize` - Image resizing

Always benchmark your implementation against NPP.

## Prerequisites

```bash
# Image I/O (optional, for testing with real images)
# Our examples use synthetic test images
```

## Building

Each day's exercise:
```bash
cd day1-convolution-basics
./build.sh
./build/convolution_basic
```
