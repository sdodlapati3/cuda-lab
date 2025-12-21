# Day 2: Pinned Memory

## Learning Objectives

- Understand page-locked (pinned) memory
- Enable async host-device transfers
- Measure bandwidth improvements

## Key Concepts

### Pageable vs Pinned Memory

```cpp
// Pageable (default) - OS can swap to disk
float* h_pageable = new float[N];

// Pinned - page-locked, never swapped
float* h_pinned;
cudaMallocHost(&h_pinned, N * sizeof(float));
```

### Why Pinned Memory is Faster

1. **No staging buffer**: DMA directly from pinned memory
2. **Async transfer**: CPU can continue while transfer happens
3. **Higher bandwidth**: Direct path to GPU

```
Pageable:  CPU Mem → Staging Buffer → GPU (2 copies)
Pinned:    CPU Mem → GPU (1 copy, DMA)
```

### Async Transfer Pattern

```cpp
cudaMallocHost(&h_pinned, size);

// Overlap transfer with compute
cudaMemcpyAsync(d_ptr, h_pinned, size, 
                cudaMemcpyHostToDevice, stream);
kernel<<<..., stream>>>(d_ptr);  // Runs while copying
```

### Caveats

- Pinned memory is a limited resource
- Excessive pinning can hurt system performance
- Use only for hot transfer paths

## Build & Run

```bash
./build.sh
./build/pinned_memory
```
