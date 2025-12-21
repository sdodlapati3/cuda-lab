/**
 * Day 1: Hello GPU - Your first CMake-built CUDA program
 * 
 * Learning objectives:
 * 1. Understand kernel launch syntax <<<blocks, threads>>>
 * 2. See how to query device properties
 * 3. Verify build system is working correctly
 */

#include <cstdio>

// Device function: runs on GPU
__global__ void hello_kernel(int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        printf("Hello from thread %d (block %d, local %d)\n", 
               idx, blockIdx.x, threadIdx.x);
    }
}

// Query and print device info
void print_device_info() {
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    
    printf("\n=== GPU Device Info ===\n");
    printf("Device: %s\n", props.name);
    printf("Compute Capability: %d.%d\n", props.major, props.minor);
    printf("SM Count: %d\n", props.multiProcessorCount);
    printf("Max Threads/Block: %d\n", props.maxThreadsPerBlock);
    printf("Max Threads/SM: %d\n", props.maxThreadsPerMultiProcessor);
    printf("Warp Size: %d\n", props.warpSize);
    printf("Global Memory: %.1f GB\n", props.totalGlobalMem / 1e9);
    printf("Shared Memory/Block: %zu KB\n", props.sharedMemPerBlock / 1024);
    printf("Memory Bus Width: %d bits\n", props.memoryBusWidth);
    printf("Peak Memory Bandwidth: %.0f GB/s\n", 
           2.0 * props.memoryClockRate * (props.memoryBusWidth / 8) / 1e6);
    printf("========================\n\n");
}

int main() {
    print_device_info();
    
    // Launch kernel: 2 blocks of 4 threads each = 8 threads total
    const int n = 8;
    const int threads_per_block = 4;
    const int blocks = (n + threads_per_block - 1) / threads_per_block;
    
    printf("Launching kernel: %d blocks x %d threads\n\n", blocks, threads_per_block);
    
    hello_kernel<<<blocks, threads_per_block>>>(n);
    
    // Wait for kernel to complete
    cudaDeviceSynchronize();
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    printf("\nKernel completed successfully!\n");
    return 0;
}
