/**
 * Multi-GPU Programming Basics
 * 
 * Demonstrates:
 * - Device enumeration and selection
 * - Data parallel workload distribution
 * - Peer-to-peer memory access
 * - Multi-GPU synchronization
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#define BLOCK_SIZE 256

// Simple kernel for demonstration
__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void vectorScale(float* data, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= scale;
    }
}

void printDeviceInfo() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    printf("=== CUDA Device Information ===\n");
    printf("Number of CUDA devices: %d\n\n", deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total memory: %.2f GB\n", prop.totalGlobalMem / 1e9);
        printf("  SMs: %d\n", prop.multiProcessorCount);
        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("\n");
    }
}

void checkPeerAccess(int deviceCount) {
    printf("=== Peer-to-Peer Access Matrix ===\n");
    printf("     ");
    for (int j = 0; j < deviceCount; j++) printf("GPU%d ", j);
    printf("\n");
    
    for (int i = 0; i < deviceCount; i++) {
        printf("GPU%d ", i);
        for (int j = 0; j < deviceCount; j++) {
            if (i == j) {
                printf("  -  ");
            } else {
                int canAccess;
                cudaDeviceCanAccessPeer(&canAccess, i, j);
                printf("  %s  ", canAccess ? "Y" : "N");
            }
        }
        printf("\n");
    }
    printf("\n");
}

void singleGPUBenchmark(int n, float* h_a, float* h_b, float* h_c) {
    printf("=== Single GPU Benchmark ===\n");
    
    float *d_a, *d_b, *d_c;
    size_t size = n * sizeof(float);
    
    cudaSetDevice(0);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Allocate and transfer
    cudaEventRecord(start);
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    // Compute
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vectorAdd<<<numBlocks, BLOCK_SIZE>>>(d_a, d_b, d_c, n);
    
    // Transfer back
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    printf("  Elements: %d\n", n);
    printf("  Time: %.2f ms\n", ms);
    printf("  Throughput: %.2f GB/s\n", 3.0 * size / ms / 1e6);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("\n");
}

void multiGPUBenchmark(int n, float* h_a, float* h_b, float* h_c, int numGPUs) {
    printf("=== Multi-GPU Benchmark (%d GPUs) ===\n", numGPUs);
    
    if (numGPUs < 2) {
        printf("  Skipped: Need at least 2 GPUs\n\n");
        return;
    }
    
    int elementsPerGPU = n / numGPUs;
    size_t sizePerGPU = elementsPerGPU * sizeof(float);
    
    // Allocate per-GPU arrays
    std::vector<float*> d_a(numGPUs), d_b(numGPUs), d_c(numGPUs);
    std::vector<cudaStream_t> streams(numGPUs);
    
    cudaEvent_t start, stop;
    cudaSetDevice(0);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    // Allocate and transfer to each GPU
    for (int gpu = 0; gpu < numGPUs; gpu++) {
        cudaSetDevice(gpu);
        cudaStreamCreate(&streams[gpu]);
        
        cudaMalloc(&d_a[gpu], sizePerGPU);
        cudaMalloc(&d_b[gpu], sizePerGPU);
        cudaMalloc(&d_c[gpu], sizePerGPU);
        
        int offset = gpu * elementsPerGPU;
        cudaMemcpyAsync(d_a[gpu], h_a + offset, sizePerGPU, 
                        cudaMemcpyHostToDevice, streams[gpu]);
        cudaMemcpyAsync(d_b[gpu], h_b + offset, sizePerGPU,
                        cudaMemcpyHostToDevice, streams[gpu]);
    }
    
    // Launch kernels on each GPU
    for (int gpu = 0; gpu < numGPUs; gpu++) {
        cudaSetDevice(gpu);
        int numBlocks = (elementsPerGPU + BLOCK_SIZE - 1) / BLOCK_SIZE;
        vectorAdd<<<numBlocks, BLOCK_SIZE, 0, streams[gpu]>>>(
            d_a[gpu], d_b[gpu], d_c[gpu], elementsPerGPU);
    }
    
    // Copy results back
    for (int gpu = 0; gpu < numGPUs; gpu++) {
        cudaSetDevice(gpu);
        int offset = gpu * elementsPerGPU;
        cudaMemcpyAsync(h_c + offset, d_c[gpu], sizePerGPU,
                        cudaMemcpyDeviceToHost, streams[gpu]);
    }
    
    // Synchronize all
    for (int gpu = 0; gpu < numGPUs; gpu++) {
        cudaSetDevice(gpu);
        cudaStreamSynchronize(streams[gpu]);
    }
    
    cudaSetDevice(0);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    printf("  Elements per GPU: %d\n", elementsPerGPU);
    printf("  Time: %.2f ms\n", ms);
    printf("  Throughput: %.2f GB/s\n", 3.0 * n * sizeof(float) / ms / 1e6);
    
    // Cleanup
    for (int gpu = 0; gpu < numGPUs; gpu++) {
        cudaSetDevice(gpu);
        cudaFree(d_a[gpu]);
        cudaFree(d_b[gpu]);
        cudaFree(d_c[gpu]);
        cudaStreamDestroy(streams[gpu]);
    }
    cudaSetDevice(0);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("\n");
}

void peerAccessDemo(int numGPUs) {
    printf("=== Peer-to-Peer Access Demo ===\n");
    
    if (numGPUs < 2) {
        printf("  Skipped: Need at least 2 GPUs\n\n");
        return;
    }
    
    // Check and enable P2P
    int canAccess;
    cudaDeviceCanAccessPeer(&canAccess, 0, 1);
    
    if (!canAccess) {
        printf("  P2P not supported between GPU 0 and GPU 1\n\n");
        return;
    }
    
    cudaSetDevice(0);
    cudaDeviceEnablePeerAccess(1, 0);
    cudaSetDevice(1);
    cudaDeviceEnablePeerAccess(0, 0);
    
    printf("  P2P enabled between GPU 0 and GPU 1\n");
    
    // Allocate on each GPU
    int n = 10000000;
    float *d_src, *d_dst;
    
    cudaSetDevice(0);
    cudaMalloc(&d_src, n * sizeof(float));
    
    cudaSetDevice(1);
    cudaMalloc(&d_dst, n * sizeof(float));
    
    // Benchmark P2P transfer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    cudaMemcpyPeer(d_dst, 1, d_src, 0, n * sizeof(float));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    printf("  P2P transfer: %d elements in %.2f ms\n", n, ms);
    printf("  P2P bandwidth: %.2f GB/s\n", n * sizeof(float) / ms / 1e6);
    
    // Cleanup
    cudaSetDevice(0);
    cudaFree(d_src);
    cudaSetDevice(1);
    cudaFree(d_dst);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("\n");
}

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount == 0) {
        printf("No CUDA devices found!\n");
        return 1;
    }
    
    printDeviceInfo();
    checkPeerAccess(deviceCount);
    
    // Prepare test data
    int n = 50000000;  // 50M elements
    printf("Allocating %.2f MB test data...\n\n", 3.0 * n * sizeof(float) / 1e6);
    
    float* h_a = (float*)malloc(n * sizeof(float));
    float* h_b = (float*)malloc(n * sizeof(float));
    float* h_c = (float*)malloc(n * sizeof(float));
    
    for (int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }
    
    // Benchmarks
    singleGPUBenchmark(n, h_a, h_b, h_c);
    multiGPUBenchmark(n, h_a, h_b, h_c, deviceCount);
    peerAccessDemo(deviceCount);
    
    // Verify result
    printf("=== Verification ===\n");
    bool pass = true;
    for (int i = 0; i < n && pass; i++) {
        if (h_c[i] != 3.0f) pass = false;
    }
    printf("  Result: %s\n", pass ? "PASS" : "FAIL");
    
    // Cleanup
    free(h_a);
    free(h_b);
    free(h_c);
    
    return 0;
}
