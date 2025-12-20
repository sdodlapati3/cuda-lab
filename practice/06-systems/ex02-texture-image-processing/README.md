# Exercise 2: Texture Image Processing

## Objective

Use CUDA texture objects to implement image processing operations with hardware-accelerated filtering.

## Task

Implement a 2D image processor that:
1. Creates a texture object from input image data
2. Performs bilinear upsampling (2x scale)
3. Applies a Gaussian blur using texture sampling
4. Compares performance with global memory implementation

## Requirements

1. Use `cudaCreateTextureObject` (not deprecated texture references)
2. Use CUDA arrays for 2D spatial locality
3. Implement bilinear filtering mode
4. Handle border pixels with clamp addressing

## File to Complete

- `texture_processor.cu`

## Expected Output

```
=== Texture Image Processing ===

Input: 256x256 image
Output: 512x512 (2x upscale)

Bilinear Upscale:
  Texture: 0.42 ms
  Global:  1.83 ms
  Speedup: 4.36x

Gaussian Blur (5x5):
  Texture: 1.21 ms
  Global:  3.45 ms
  Speedup: 2.85x
```

## Hints

- Use `cudaFilterModeLinear` for bilinear interpolation
- Use normalized coordinates (`normalizedCoords = true`)
- For upscaling: `u = (x + 0.5) / outputWidth`
- CUDA arrays are allocated with `cudaMallocArray`
