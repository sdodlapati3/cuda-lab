/**
 * memory_metrics.cu - Kernels demonstrating memory patterns
 * 
 * Profile with: ncu --set memory ./build/memory_metrics
 */

#include <cuda_runtime.h>
#include <cstdio>

// Perfect coalescing - 100% efficiency
__global__ void coalesced_access(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] * 2.0f;
    }
}

// Strided access - poor coalescing
__global__ void strided_access(const float* in, float* out, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int real_idx = idx * stride;
    if (real_idx < n) {
        out[real_idx] = in[real_idx] * 2.0f;
    }
}

// Random access - worst case
__global__ void random_access(const float* in, float* out, const int* indices, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int rand_idx = indices[idx];
        out[idx] = in[rand_idx] * 2.0f;
    }
}

// Sequential but misaligned
__global__ void misaligned_access(const float* in, float* out, int n, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx + offset < n) {
        out[idx] = in[idx + offset] * 2.0f;
    }
}

// Good cache reuse - stencil
__global__ void stencil_reuse(const float* in, float* out, int n) {
    __shared__ float smem[258];  // 256 + 2 halo
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Load with halos
    if (idx < n) smem[tid + 1] = in[idx];
    if (tid == 0 && idx > 0) smem[0] = in[idx - 1];
    if (tid == 255 && idx < n - 1) smem[257] = in[idx + 1];
    __syncthreads();
    
    if (idx > 0 && idx < n - 1) {
        out[idx] = 0.25f * smem[tid] + 0.5f * smem[tid + 1] + 0.25f * smem[tid + 2];
    }
}

int main() {
    printf("Memory Metrics Demo\n\n");
    printf("Profile with: ncu --set memory ./build/memory_metrics\n\n");
    
    const int N = 1 << 22;  // 4M
    float *d_in, *d_out;
    int *d_indices;
    
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMalloc(&d_indices, N * sizeof(int));
    
    // Initialize
    float* h_data = new float[N];
    int* h_indices = new int[N];
    for (int i = 0; i < N; i++) {
        h_data[i] = 1.0f;
        h_indices[i] = (i * 31337) % N;  // Pseudo-random
    }
    cudaMemcpy(d_in, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, h_indices, N * sizeof(int), cudaMemcpyHostToDevice);
    
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    printf("Launching kernels with different memory patterns...\n\n");
    
    printf("1. coalesced_access - expect ~100%% memory efficiency\n");
    coalesced_access<<<blocks, threads>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    
    printf("2. strided_access (stride=2) - ~50%% efficiency\n");
    strided_access<<<blocks/2, threads>>>(d_in, d_out, N, 2);
    cudaDeviceSynchronize();
    
    printf("3. strided_access (stride=32) - very poor efficiency\n");
    strided_access<<<blocks/32, threads>>>(d_in, d_out, N, 32);
    cudaDeviceSynchronize();
    
    printf("4. random_access - worst case, many cache misses\n");
    random_access<<<blocks, threads>>>(d_in, d_out, d_indices, N);
    cudaDeviceSynchronize();
    
    printf("5. misaligned_access (offset=1) - slight inefficiency\n");
    misaligned_access<<<blocks, threads>>>(d_in, d_out, N, 1);
    cudaDeviceSynchronize();
    
    printf("6. stencil_reuse - good L1/smem reuse\n");
    stencil_reuse<<<blocks, threads>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    
    printf("\n=== Memory Metrics Guide ===\n\n");
    printf("Metric                      | Good       | Bad\n");
    printf("------------------------------------------------------\n");
    printf("Global Load Efficiency      | 100%%       | < 25%%\n");
    printf("L2 Hit Rate                 | High       | Low (cache thrashing)\n");
    printf("DRAM Throughput             | Near peak  | Far from peak\n");
    printf("Memory Sectors/Request      | 1          | > 1 (uncoalesced)\n");
    printf("L1 Global Load Hit Rate     | Task-dep   | -\n");
    printf("\nWatch for 'Memory Workload Analysis' section in ncu!\n");
    
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_indices);
    delete[] h_data;
    delete[] h_indices;
    
    return 0;
}
