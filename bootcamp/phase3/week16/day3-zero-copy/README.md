# Day 3: Zero-Copy and Mapped Memory

## Learning Objectives

- Use mapped (zero-copy) memory
- Understand when zero-copy is beneficial
- Compare transfer vs zero-copy patterns

## Key Concepts

### What is Zero-Copy?

Host memory accessible directly from GPU without explicit transfer.

```cpp
// Allocate mapped memory
float* h_mapped;
cudaHostAlloc(&h_mapped, size, cudaHostAllocMapped);

// Get device pointer
float* d_mapped;
cudaHostGetDevicePointer(&d_mapped, h_mapped, 0);

// Kernel accesses host memory directly
kernel<<<...>>>(d_mapped);
```

### When to Use Zero-Copy

✅ **Good for:**
- Data read/written once
- Small, irregular accesses
- Integrated GPU (shared memory system)
- Memory larger than GPU capacity

❌ **Bad for:**
- Repeated access to same data
- Large sequential transfers
- Discrete GPU with PCIe bottleneck

### Unified Virtual Addressing (UVA)

```cpp
// With UVA, pointer works on both host and device
float* h_mapped;
cudaHostAlloc(&h_mapped, size, cudaHostAllocMapped);

// Use same pointer in kernel (UVA enabled)
kernel<<<...>>>(h_mapped);  // Works directly!
```

## Build & Run

```bash
./build.sh
./build/zero_copy
```
