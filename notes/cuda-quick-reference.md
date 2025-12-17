# CUDA Quick Reference Cheatsheet

> 🚀 Quick lookup for common CUDA patterns and syntax.  
> 📖 Full docs: [CUDA Programming Guide](../cuda-programming-guide/index.md)

---

## 🔧 Kernel Basics

### Function Specifiers
```cpp
__global__ void kernel()  // GPU, callable from CPU (kernel)
__device__ void func()    // GPU, callable from GPU only
__host__ void func()      // CPU only (default)
__host__ __device__       // Both CPU and GPU
```

### Kernel Launch
```cpp
kernel<<<gridDim, blockDim, sharedMem, stream>>>(args);

// Examples:
kernel<<<1, 256>>>(data);                    // 1 block, 256 threads
kernel<<<numBlocks, 256>>>(data);            // N blocks, 256 threads each
kernel<<<dim3(2,2), dim3(16,16)>>>(data);    // 2D grid and blocks
kernel<<<blocks, threads, 1024, stream>>>(data);  // With shared mem & stream
```

### Built-in Variables
```cpp
threadIdx.x, .y, .z   // Thread index within block (0 to blockDim-1)
blockIdx.x, .y, .z    // Block index within grid (0 to gridDim-1)
blockDim.x, .y, .z    // Threads per block
gridDim.x, .y, .z     // Blocks in grid
warpSize              // Threads per warp (32)
```

### Global Thread Index (1D)
```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;

// Grid-stride loop pattern:
for (int i = idx; i < n; i += stride) {
    output[i] = input[i] * 2;
}
```

### Global Thread Index (2D)
```cpp
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int idx = y * width + x;
```

---

## 💾 Memory Types

| Type | Declaration | Scope | Lifetime | Speed |
|------|-------------|-------|----------|-------|
| Register | `int x;` | Thread | Thread | Fastest |
| Local | `int arr[N];` (large) | Thread | Thread | Slow (DRAM) |
| Shared | `__shared__ int s[N];` | Block | Block | Fast |
| Global | `__device__ int g;` | Grid | App | Slow (DRAM) |
| Constant | `__constant__ int c;` | Grid | App | Fast (cached) |

### Memory Allocation
```cpp
// Device memory
cudaMalloc(&d_ptr, size);
cudaFree(d_ptr);

// Host-Device transfers
cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice);
cudaMemcpy(h_ptr, d_ptr, size, cudaMemcpyDeviceToHost);

// Unified Memory (auto-migrates)
cudaMallocManaged(&ptr, size);

// Pinned host memory (faster transfers)
cudaMallocHost(&h_ptr, size);
cudaFreeHost(h_ptr);
```

### Shared Memory
```cpp
// Static allocation
__shared__ float tile[16][16];

// Dynamic allocation (size set at launch)
extern __shared__ float dynamicSmem[];
kernel<<<blocks, threads, sharedBytes>>>();
```

---

## 🔄 Synchronization

```cpp
__syncthreads();          // Sync all threads in block
__syncwarp(mask);         // Sync threads in warp
cudaDeviceSynchronize();  // Sync host with device (all streams)
cudaStreamSynchronize(s); // Sync host with specific stream
```

---

## 🌊 Streams & Async

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);

// Async operations
cudaMemcpyAsync(dst, src, size, kind, stream);
kernel<<<grid, block, 0, stream>>>(args);

cudaStreamSynchronize(stream);
cudaStreamDestroy(stream);
```

### Events (timing/sync)
```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start, stream);
kernel<<<...>>>();
cudaEventRecord(stop, stream);
cudaEventSynchronize(stop);

float ms;
cudaEventElapsedTime(&ms, start, stop);
```

---

## ⚛️ Atomics

```cpp
atomicAdd(&addr, val);     // addr += val
atomicSub(&addr, val);     // addr -= val
atomicMin(&addr, val);     // addr = min(addr, val)
atomicMax(&addr, val);     // addr = max(addr, val)
atomicAnd(&addr, val);     // addr &= val
atomicOr(&addr, val);      // addr |= val
atomicXor(&addr, val);     // addr ^= val
atomicExch(&addr, val);    // swap
atomicCAS(&addr, cmp, val); // compare-and-swap
```

---

## 🧵 Warp Primitives

```cpp
// Vote functions
__all_sync(mask, pred);     // All threads have pred true?
__any_sync(mask, pred);     // Any thread has pred true?
__ballot_sync(mask, pred);  // Bitmap of pred results

// Shuffle (exchange within warp)
__shfl_sync(mask, val, srcLane);      // Get val from srcLane
__shfl_up_sync(mask, val, delta);     // Get from lane - delta
__shfl_down_sync(mask, val, delta);   // Get from lane + delta
__shfl_xor_sync(mask, val, laneMask); // Get from lane ^ laneMask

// Reduce (sm_80+)
__reduce_add_sync(mask, val);
__reduce_min_sync(mask, val);
__reduce_max_sync(mask, val);
```

---

## 🚨 Error Handling

```cpp
cudaError_t err = cudaMalloc(&ptr, size);
if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
}

// Check after kernel launch
kernel<<<...>>>();
cudaError_t err = cudaGetLastError();
```

### Error Check Macro
```cpp
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Usage:
CUDA_CHECK(cudaMalloc(&d_ptr, size));
```

---

## 📊 Device Properties

```cpp
int deviceCount;
cudaGetDeviceCount(&deviceCount);

cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);

printf("Device: %s\n", prop.name);
printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
printf("Max threads/block: %d\n", prop.maxThreadsPerBlock);
printf("Shared mem/block: %zu\n", prop.sharedMemPerBlock);
printf("Warp size: %d\n", prop.warpSize);
```

---

## 🧮 Common Patterns

### Vector Addition
```cpp
__global__ void vecAdd(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
```

### Matrix Multiplication (Tiled)
```cpp
__global__ void matMul(float *A, float *B, float *C, int N) {
    __shared__ float As[TILE][TILE], Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0;
    
    for (int t = 0; t < N/TILE; t++) {
        As[threadIdx.y][threadIdx.x] = A[row*N + t*TILE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t*TILE + threadIdx.y)*N + col];
        __syncthreads();
        
        for (int k = 0; k < TILE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    C[row*N + col] = sum;
}
```

### Reduction (Warp-optimized)
```cpp
__global__ void reduce(float *input, float *output, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? input[i] : 0;
    __syncthreads();
    
    // Reduce in shared memory
    for (int s = blockDim.x/2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    
    // Warp-level reduction (no sync needed)
    if (tid < 32) {
        float val = sdata[tid] + sdata[tid + 32];
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        if (tid == 0) output[blockIdx.x] = val;
    }
}
```

---

## 📐 Launch Configuration

### Calculate Grid Size
```cpp
int threadsPerBlock = 256;
int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
kernel<<<numBlocks, threadsPerBlock>>>(data, n);
```

### Occupancy-based Launch
```cpp
int minGridSize, blockSize;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, 0, 0);
int gridSize = (n + blockSize - 1) / blockSize;
kernel<<<gridSize, blockSize>>>(data, n);
```

---

## 📝 Compilation

```bash
# Basic compile
nvcc program.cu -o program

# Specify architecture
nvcc -arch=sm_80 program.cu -o program

# Debug build
nvcc -g -G program.cu -o program

# Optimization
nvcc -O3 program.cu -o program

# Separate compilation
nvcc -dc file1.cu file2.cu
nvcc file1.o file2.o -o program
```

---

## 🔗 Related Resources

- [CUDA Programming Guide](../cuda-programming-guide/index.md) - Full documentation
- [Practice Examples](../practice/) - Hands-on code
