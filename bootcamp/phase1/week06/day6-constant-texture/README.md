# Day 6: Constant & Texture Memory

## Learning Objectives

- Use constant memory for broadcast data
- Understand texture memory benefits
- Choose the right memory type for your access pattern

## Constant Memory

### Characteristics

- 64 KB total (device-wide)
- Cached per SM
- **Optimized for broadcast**: All threads read same address
- Read-only from kernel

### Declaration & Usage

```cuda
__constant__ float coefficients[256];

__host__ void setup() {
    float h_coeffs[256] = {...};
    cudaMemcpyToSymbol(coefficients, h_coeffs, sizeof(h_coeffs));
}

__global__ void kernel() {
    float c = coefficients[0];  // Broadcast: same value to all threads
}
```

### Best Use Cases

- Filter coefficients (same for all pixels)
- Lookup tables (small, read-only)
- Configuration parameters
- Math constants

### Anti-pattern

```cuda
// BAD: Each thread reads different index
float c = coefficients[threadIdx.x];  // Serialized access!

// GOOD: All threads read same index
float c = coefficients[iteration % 10];  // Broadcast
```

## Texture Memory

### Characteristics

- Read-only cache optimized for 2D spatial locality
- Hardware interpolation (free!)
- Automatic boundary handling
- Useful for image processing

### Modern Approach: Surface Objects

```cuda
cudaTextureObject_t tex;
cudaResourceDesc resDesc = {};
resDesc.resType = cudaResourceTypeLinear;
resDesc.res.linear.devPtr = d_data;
resDesc.res.linear.sizeInBytes = size;

cudaTextureDesc texDesc = {};
texDesc.readMode = cudaReadModeElementType;

cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);

// In kernel
float val = tex1Dfetch<float>(tex, idx);
```

### Use Cases

- Image processing (2D spatial locality)
- Volume rendering (3D interpolation)
- Lookup tables with interpolation

## Exercises

1. **Constant memory benchmark**: Compare broadcast vs scattered access
2. **Filter kernel**: Use constant memory for coefficients
3. **Texture sampling**: Implement bilinear interpolation

## Build & Run

```bash
./build.sh
./build/constant_memory
./build/texture_demo
```
