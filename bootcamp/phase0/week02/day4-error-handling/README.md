# Day 4: Async Error Handling

## What You'll Learn

- Understand CUDA's asynchronous execution model
- Proper error checking patterns
- Catch errors from async operations
- Build robust error handling infrastructure

## The Problem: CUDA is Asynchronous

Most CUDA operations are **asynchronous**:
- Kernel launches return immediately
- Memory copies can be async
- Errors may appear later

```cpp
my_kernel<<<...>>>();      // Returns immediately
// Error from kernel hasn't happened yet!
cudaError_t err = cudaGetLastError();  // Might miss the error!

cudaDeviceSynchronize();   // Wait for kernel
err = cudaGetLastError();  // NOW we can check
```

## Quick Start

```bash
./build.sh
./build/async_errors
```

## Error Checking Patterns

### Pattern 1: Check After Every API Call
```cpp
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

CUDA_CHECK(cudaMalloc(&ptr, size));
CUDA_CHECK(cudaMemcpy(...));
```

### Pattern 2: Check After Kernel Launch
```cpp
my_kernel<<<blocks, threads>>>();
CUDA_CHECK(cudaGetLastError());  // Check launch errors
CUDA_CHECK(cudaDeviceSynchronize());  // Wait and check execution errors
```

### Pattern 3: Sticky Errors
```cpp
// CUDA errors are "sticky" - they persist until cleared
cudaError_t err = cudaGetLastError();  // Clears error state
if (err != cudaSuccess) {
    // Handle error
}
```

### Pattern 4: Callback-Based (Advanced)
```cpp
// Set error callback (CUDA 12+)
cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1024 * 1024);
```

## Common Async Errors

| Error | Cause | When Detected |
|-------|-------|---------------|
| Invalid configuration | Bad launch params | At launch |
| Out of memory | Too much shared mem | At launch |
| Illegal address | OOB access | After sync |
| Misaligned access | Bad pointer | After sync |

## Error Propagation Strategy

```cpp
cudaError_t run_pipeline() {
    cudaError_t err;
    
    // Multiple async operations
    kernel1<<<...>>>();
    kernel2<<<...>>>();
    kernel3<<<...>>>();
    
    // Single sync point at the end
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        // One of the kernels failed - which one?
        // Use CUDA event-based profiling to isolate
        return err;
    }
    
    return cudaSuccess;
}
```

## Debug vs Release

```cpp
#ifdef DEBUG
    #define CUDA_CHECK_KERNEL() do { \
        CUDA_CHECK(cudaGetLastError()); \
        CUDA_CHECK(cudaDeviceSynchronize()); \
    } while(0)
#else
    #define CUDA_CHECK_KERNEL() ((void)0)
#endif
```

## Exercises

1. Create a kernel that triggers an async error
2. Practice different error checking patterns
3. Build an error handling wrapper class
4. Test error propagation across multiple kernels
