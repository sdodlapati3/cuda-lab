/**
 * coalescing_demo.cu - Demonstrate memory coalescing impact
 * 
 * Learning objectives:
 * - See performance impact of coalesced vs strided access
 * - Understand memory transaction efficiency
 */

#include <cuda_runtime.h>
#include <cstdio>

// Coalesced access: threads read consecutive addresses
__global__ void coalesced_read(const float* data, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    float sum = 0.0f;
    for (int i = idx; i < n; i += stride) {
        sum += data[i];  // Thread i reads data[i] - coalesced!
    }
    output[idx % 1024] = sum;
}

// Strided access: threads read with large stride
__global__ void strided_read(const float* data, float* output, int n, int access_stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    float sum = 0.0f;
    for (int i = idx; i < n / access_stride; i += stride) {
        // Thread i reads data[i * access_stride] - NOT coalesced!
        sum += data[i * access_stride];
    }
    output[idx % 1024] = sum;
}

// Random access pattern (worst case)
__global__ void random_read(const float* data, const int* indices, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    float sum = 0.0f;
    for (int i = idx; i < n; i += stride) {
        sum += data[indices[i]];  // Random access!
    }
    output[idx % 1024] = sum;
}

int main() {
    printf("=== Memory Coalescing Benchmark ===\n\n");
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    printf("Memory bandwidth: %.0f GB/s (theoretical peak)\n\n", 
           prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8) * 2 / 1e9);
    
    const int N = 32 * 1024 * 1024;  // 32M elements = 128 MB
    const size_t bytes = N * sizeof(float);
    
    float* d_data, *d_output;
    int* d_indices;
    
    cudaMalloc(&d_data, bytes);
    cudaMalloc(&d_output, 1024 * sizeof(float));
    cudaMalloc(&d_indices, N * sizeof(int));
    
    // Initialize data
    float* h_data = new float[N];
    int* h_indices = new int[N];
    for (int i = 0; i < N; i++) {
        h_data[i] = 1.0f;
        h_indices[i] = (i * 17 + 5) % N;  // Pseudo-random
    }
    
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, h_indices, N * sizeof(int), cudaMemcpyHostToDevice);
    
    int block_size = 256;
    int num_blocks = 256;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    printf("=== Access Pattern Comparison ===\n");
    printf("%20s %12s %15s %12s\n", "Pattern", "Time (ms)", "Bandwidth (GB/s)", "Efficiency");
    printf("------------------------------------------------------------------\n");
    
    // Benchmark coalesced
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    coalesced_read<<<num_blocks, block_size>>>(d_data, d_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    float bw = bytes / ms / 1e6;
    float peak_bw = prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8) * 2 / 1e9;
    printf("%20s %12.3f %15.1f %11.1f%%\n", "Coalesced (stride 1)", ms, bw, 100.0f * bw / peak_bw / 1000);
    float coalesced_bw = bw;
    
    // Benchmark strided access
    int strides[] = {2, 4, 8, 16, 32};
    for (int s : strides) {
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        strided_read<<<num_blocks, block_size>>>(d_data, d_output, N, s);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        cudaEventElapsedTime(&ms, start, stop);
        bw = (bytes / s) / ms / 1e6;  // Fewer bytes read
        
        char label[32];
        snprintf(label, sizeof(label), "Stride %d", s);
        printf("%20s %12.3f %15.1f %11.1f%%\n", label, ms, bw, 100.0f * bw / coalesced_bw);
    }
    
    // Benchmark random access
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    random_read<<<num_blocks, block_size>>>(d_data, d_indices, d_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&ms, start, stop);
    bw = bytes / ms / 1e6;
    printf("%20s %12.3f %15.1f %11.1f%%\n", "Random", ms, bw, 100.0f * bw / coalesced_bw);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    cudaFree(d_output);
    cudaFree(d_indices);
    delete[] h_data;
    delete[] h_indices;
    
    printf("\n=== Key Insights ===\n");
    printf("1. Coalesced access achieves highest bandwidth\n");
    printf("2. Stride-N reduces effective bandwidth by ~N×\n");
    printf("3. Random access is worst - avoid if possible\n");
    printf("4. Profile with: ncu --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second\n");
    
    return 0;
}
