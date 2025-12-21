# Day 5: Image Resizing

## Learning Objectives
- Implement bilinear and bicubic interpolation
- Understand texture memory for interpolation
- Handle non-integer coordinate sampling

## Key Concepts

### Bilinear Interpolation
```
Given floating point (x, y):
1. Find 4 neighboring integer pixels
2. Compute weights from fractional parts
3. Blend: top-left*w1 + top-right*w2 + bottom-left*w3 + bottom-right*w4
```

### Coordinate Mapping
```
For downscaling by factor S:
    src_x = dst_x * S
    src_y = dst_y * S

Fractional coordinates require interpolation
```

### Texture Memory
- Hardware-accelerated interpolation
- Automatic boundary clamping
- Cache optimized for 2D locality
