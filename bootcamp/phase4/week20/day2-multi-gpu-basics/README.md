# Day 2: Multi-GPU Basics

## Learning Objectives
- Query and select CUDA devices
- Transfer data between GPUs
- Implement simple data parallelism
- Understand peer-to-peer memory access

## Device Management

### Querying Devices
```cpp
int deviceCount;
cudaGetDeviceCount(&deviceCount);

for (int i = 0; i < deviceCount; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device %d: %s\n", i, prop.name);
}
```

### Selecting Device
```cpp
cudaSetDevice(0);  // Use GPU 0
// All subsequent CUDA calls use GPU 0

cudaSetDevice(1);  // Switch to GPU 1
// Now using GPU 1
```

## Data Parallelism Pattern
```
Input Data: [A, B, C, D]
           /    \
        GPU 0   GPU 1
        [A,B]   [C,D]
          |       |
       Process  Process
          |       |
        [A',B'] [C',D']
           \    /
Output Data: [A', B', C', D']
```

## Peer-to-Peer Access
Direct GPU-to-GPU memory transfer without CPU:
```cpp
int canAccess;
cudaDeviceCanAccessPeer(&canAccess, gpu0, gpu1);

if (canAccess) {
    cudaSetDevice(gpu0);
    cudaDeviceEnablePeerAccess(gpu1, 0);
    
    // Now gpu0 can access gpu1's memory
    cudaMemcpyPeer(d_dst, gpu0, d_src, gpu1, size);
}
```

## Synchronization
```cpp
// Per-device synchronization
cudaSetDevice(0);
cudaDeviceSynchronize();

cudaSetDevice(1);
cudaDeviceSynchronize();

// Or use streams per device
```

## Exercises
1. Query all available GPUs
2. Implement vector addition across 2 GPUs
3. Measure data transfer overhead
4. Compare P2P vs host-staged transfer
