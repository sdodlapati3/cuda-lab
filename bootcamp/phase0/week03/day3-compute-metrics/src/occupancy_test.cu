/**
 * Day 3: Occupancy Analysis
 * 
 * Compare different configurations and their occupancy.
 */

#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ============================================================================
// Kernel with few registers (high occupancy)
// ============================================================================
__global__ void low_register_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] + 1.0f;
    }
}

// ============================================================================
// Kernel with many registers (lower occupancy)
// ============================================================================
__global__ void high_register_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Force many registers
        float r0 = data[idx];
        float r1 = r0 * 1.1f, r2 = r0 * 1.2f, r3 = r0 * 1.3f, r4 = r0 * 1.4f;
        float r5 = r1 * 1.1f, r6 = r2 * 1.2f, r7 = r3 * 1.3f, r8 = r4 * 1.4f;
        float r9 = r5 + r6, r10 = r7 + r8, r11 = r1 + r2, r12 = r3 + r4;
        float r13 = r9 + r10, r14 = r11 + r12;
        float r15 = r5 * r6, r16 = r7 * r8, r17 = r1 * r2, r18 = r3 * r4;
        float r19 = r15 + r16, r20 = r17 + r18;
        data[idx] = r13 + r14 + r19 + r20;
    }
}

// ============================================================================
// Kernel with shared memory (occupancy limited by smem)
// ============================================================================
__global__ void shared_memory_kernel(float* data, int n) {
    __shared__ float smem[4096];  // 16KB shared memory
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    if (idx < n) {
        smem[tid] = data[idx];
        __syncthreads();
        data[idx] = smem[tid] * 2.0f;
    }
}

// ============================================================================
// Test different block sizes
// ============================================================================
template<int BLOCK_SIZE>
__global__ void block_size_kernel(float* data, int n) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] + 1.0f;
    }
}

void print_occupancy_info(const char* name, void* kernel, int blockSize, 
                          int sharedMem = 0) {
    int maxActiveBlocks;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocks, kernel, blockSize, sharedMem));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    int maxWarpsPerSM = prop.maxThreadsPerMultiProcessor / 32;
    int activeWarps = maxActiveBlocks * (blockSize / 32);
    float occupancy = 100.0f * activeWarps / maxWarpsPerSM;
    
    printf("  %s:\n", name);
    printf("    Block size: %d, Max blocks/SM: %d\n", blockSize, maxActiveBlocks);
    printf("    Active warps: %d/%d, Occupancy: %.1f%%\n", 
           activeWarps, maxWarpsPerSM, occupancy);
}

int main() {
    printf("Occupancy Analysis\n");
    printf("==================\n\n");
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s\n", prop.name);
    printf("Max threads/SM: %d (%d warps)\n", 
           prop.maxThreadsPerMultiProcessor,
           prop.maxThreadsPerMultiProcessor / 32);
    printf("Max threads/block: %d\n", prop.maxThreadsPerBlock);
    printf("Registers/SM: %d\n", prop.regsPerMultiprocessor);
    printf("Shared mem/SM: %zu KB\n\n", prop.sharedMemPerMultiprocessor / 1024);
    
    const int N = 1 << 20;
    float* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
    
    // Query occupancy for different kernels
    printf("Occupancy by kernel type:\n");
    print_occupancy_info("Low register", (void*)low_register_kernel, 256);
    print_occupancy_info("High register", (void*)high_register_kernel, 256);
    print_occupancy_info("Shared memory (16KB)", (void*)shared_memory_kernel, 256, 16384);
    
    printf("\nOccupancy by block size:\n");
    print_occupancy_info("Block=64", (void*)block_size_kernel<64>, 64);
    print_occupancy_info("Block=128", (void*)block_size_kernel<128>, 128);
    print_occupancy_info("Block=256", (void*)block_size_kernel<256>, 256);
    print_occupancy_info("Block=512", (void*)block_size_kernel<512>, 512);
    print_occupancy_info("Block=1024", (void*)block_size_kernel<1024>, 1024);
    
    // Run kernels for profiling
    printf("\nRunning kernels for ncu profiling...\n");
    
    int numBlocks = (N + 255) / 256;
    
    printf("1. Low register kernel\n");
    low_register_kernel<<<numBlocks, 256>>>(d_data, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("2. High register kernel\n");
    high_register_kernel<<<numBlocks, 256>>>(d_data, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("3. Shared memory kernel\n");
    shared_memory_kernel<<<numBlocks, 256>>>(d_data, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaFree(d_data));
    
    printf("\nProfile with:\n");
    printf("  ncu --set compute ./build/occupancy_test\n");
    
    return 0;
}
