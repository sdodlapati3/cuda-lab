/**
 * shared_basics.cu - Introduction to shared memory
 * 
 * Learning objectives:
 * - Declare and use shared memory
 * - Understand synchronization requirements
 * - See performance benefits
 */

#include <cuda_runtime.h>
#include <cstdio>

// Without shared memory: many global memory accesses
__global__ void reverse_global(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int block_start = blockIdx.x * blockDim.x;
    int block_end = block_start + blockDim.x - 1;
    
    if (idx < n && block_start + (blockDim.x - 1 - threadIdx.x) < n) {
        // Read from opposite end of block
        int partner = block_start + (blockDim.x - 1 - threadIdx.x);
        if (idx < partner) {
            float temp = data[idx];
            data[idx] = data[partner];
            data[partner] = temp;
        }
    }
}

// With shared memory: fewer global accesses
__global__ void reverse_shared(float* data, int n) {
    __shared__ float tile[256];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // 1. Load to shared memory (coalesced global read)
    if (idx < n) {
        tile[tid] = data[idx];
    }
    
    // 2. Synchronize - all threads must finish loading
    __syncthreads();
    
    // 3. Write reversed (from shared memory)
    if (idx < n) {
        data[idx] = tile[blockDim.x - 1 - tid];
    }
}

// Dynamic shared memory example
__global__ void reverse_dynamic(float* data, int n) {
    extern __shared__ float dynamic_tile[];  // Size set at launch
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    if (idx < n) {
        dynamic_tile[tid] = data[idx];
    }
    __syncthreads();
    
    if (idx < n) {
        data[idx] = dynamic_tile[blockDim.x - 1 - tid];
    }
}

// Data sharing between threads
__global__ void neighbor_average(const float* input, float* output, int n) {
    __shared__ float tile[258];  // 256 + 2 for halos
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Load main data
    if (idx < n) {
        tile[tid + 1] = input[idx];
    }
    
    // Load halo elements
    if (tid == 0 && idx > 0) {
        tile[0] = input[idx - 1];
    }
    if (tid == blockDim.x - 1 && idx < n - 1) {
        tile[tid + 2] = input[idx + 1];
    }
    
    __syncthreads();
    
    // Compute average with neighbors
    if (idx > 0 && idx < n - 1) {
        output[idx] = (tile[tid] + tile[tid + 1] + tile[tid + 2]) / 3.0f;
    }
}

int main() {
    printf("=== Shared Memory Basics ===\n\n");
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    printf("Shared memory per block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("Shared memory per SM: %zu KB\n\n", prop.sharedMemPerMultiprocessor / 1024);
    
    const int N = 256;
    float* d_data;
    float* h_data = new float[N];
    
    for (int i = 0; i < N; i++) h_data[i] = (float)i;
    
    cudaMalloc(&d_data, N * sizeof(float));
    
    // Demo 1: Reverse with shared memory
    printf("=== Demo 1: Block Reverse ===\n");
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    
    printf("Before: [%.0f, %.0f, %.0f, ..., %.0f]\n", 
           h_data[0], h_data[1], h_data[2], h_data[N-1]);
    
    reverse_shared<<<1, N>>>(d_data, N);
    cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("After:  [%.0f, %.0f, %.0f, ..., %.0f]\n\n",
           h_data[0], h_data[1], h_data[2], h_data[N-1]);
    
    // Demo 2: Dynamic shared memory
    printf("=== Demo 2: Dynamic Shared Memory ===\n");
    for (int i = 0; i < N; i++) h_data[i] = (float)i;
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    
    size_t shared_size = N * sizeof(float);
    printf("Allocating %zu bytes of dynamic shared memory\n", shared_size);
    reverse_dynamic<<<1, N, shared_size>>>(d_data, N);
    cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("Result: [%.0f, %.0f, %.0f, ..., %.0f]\n\n",
           h_data[0], h_data[1], h_data[2], h_data[N-1]);
    
    // Demo 3: Neighbor communication
    printf("=== Demo 3: Neighbor Average ===\n");
    float* d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    
    for (int i = 0; i < N; i++) h_data[i] = (float)i;
    cudaMemcpy(d_input, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    
    neighbor_average<<<1, N>>>(d_input, d_output, N);
    cudaMemcpy(h_data, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("Input:  [0, 1, 2, 3, 4, ...]\n");
    printf("Output: [%.1f, %.1f, %.1f, %.1f, ...] (3-point average)\n\n",
           h_data[0], h_data[1], h_data[2], h_data[3]);
    
    cudaFree(d_data);
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_data;
    
    printf("=== Key Points ===\n");
    printf("1. __shared__ declares shared memory\n");
    printf("2. __syncthreads() is REQUIRED after writes\n");
    printf("3. extern __shared__ for dynamic sizing\n");
    printf("4. Shared memory enables neighbor communication\n");
    
    return 0;
}
