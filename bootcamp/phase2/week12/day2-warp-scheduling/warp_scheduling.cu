/**
 * warp_scheduling.cu - Demonstrate warp scheduling effects
 * 
 * Learning objectives:
 * - See how warp count affects performance
 * - Understand stall hiding
 */

#include <cuda_runtime.h>
#include <cstdio>

// Memory-bound kernel - needs many warps to hide latency
__global__ void memory_bound_kernel(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] * 2.0f;
    }
}

// Compute-bound kernel - fewer warps might be OK
__global__ void compute_bound_kernel(float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = 1.0f;
        for (int i = 0; i < 100; i++) {
            val = val * 1.0001f + 0.0001f;
        }
        out[idx] = val;
    }
}

// Kernel with forced stalls (syncthreads)
__global__ void sync_heavy_kernel(float* out, int n) {
    __shared__ float smem[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    smem[tid] = (float)tid;
    
    // Many sync points = many stall cycles
    for (int i = 0; i < 20; i++) {
        __syncthreads();
        smem[tid] = smem[255 - tid] + 0.01f;
    }
    
    if (idx < n) out[idx] = smem[tid];
}

template<typename Kernel>
void test_warp_counts(const char* name, Kernel kernel, float* d_in, float* d_out, int n) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    printf("=== %s ===\n", name);
    printf("%-20s | %12s | %12s\n", "Configuration", "Time (ms)", "Rel. Perf");
    printf("-----------------------------------------------\n");
    
    float baseline = 0;
    
    // Test different block sizes (affects warps per SM)
    for (int block_size : {32, 64, 128, 256, 512, 1024}) {
        int blocks = (n + block_size - 1) / block_size;
        int warps_per_block = block_size / 32;
        
        // Warmup
        kernel<<<blocks, block_size>>>(d_out, n);
        cudaDeviceSynchronize();
        
        cudaEventRecord(start);
        for (int i = 0; i < 100; i++) {
            kernel<<<blocks, block_size>>>(d_out, n);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        float time_per = ms / 100;
        
        if (baseline == 0) baseline = time_per;
        float rel_perf = baseline / time_per;
        
        char config[64];
        snprintf(config, sizeof(config), "%d threads (%d warps)", block_size, warps_per_block);
        printf("%-20s | %12.4f | %11.2fx\n", config, time_per, rel_perf);
    }
    printf("\n");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    printf("=== Warp Scheduling Demo ===\n\n");
    
    const int N = 1 << 22;  // 4M
    float *d_in, *d_out;
    
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    
    float* h_data = new float[N];
    for (int i = 0; i < N; i++) h_data[i] = 1.0f;
    cudaMemcpy(d_in, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Memory-bound: more warps = better latency hiding
    auto mem_kernel = [d_in, d_out, N](int blocks, int threads) {
        memory_bound_kernel<<<blocks, threads>>>(d_in, d_out, N);
    };
    
    // Manual tests for different kernels
    printf("Memory-bound kernel (benefits from more warps for latency hiding):\n\n");
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    float baseline = 0;
    printf("%-20s | %12s | %12s\n", "Block Size", "Time (ms)", "Rel. Perf");
    printf("-----------------------------------------------\n");
    
    for (int block_size : {32, 64, 128, 256, 512}) {
        int blocks = (N + block_size - 1) / block_size;
        
        cudaEventRecord(start);
        for (int i = 0; i < 100; i++) {
            memory_bound_kernel<<<blocks, block_size>>>(d_in, d_out, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        if (baseline == 0) baseline = ms;
        
        printf("%-20d | %12.4f | %11.2fx\n", block_size, ms/100, baseline/ms);
    }
    printf("\n");
    
    printf("Compute-bound kernel (less sensitive to warp count):\n\n");
    
    baseline = 0;
    printf("%-20s | %12s | %12s\n", "Block Size", "Time (ms)", "Rel. Perf");
    printf("-----------------------------------------------\n");
    
    for (int block_size : {32, 64, 128, 256, 512}) {
        int blocks = (N + block_size - 1) / block_size;
        
        cudaEventRecord(start);
        for (int i = 0; i < 20; i++) {
            compute_bound_kernel<<<blocks, block_size>>>(d_out, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        if (baseline == 0) baseline = ms;
        
        printf("%-20d | %12.4f | %11.2fx\n", block_size, ms/20, baseline/ms);
    }
    printf("\n");
    
    printf("=== Key Insights ===\n\n");
    printf("1. Memory-bound kernels: Need more warps to hide memory latency\n");
    printf("2. Compute-bound kernels: Less sensitive to warp count\n");
    printf("3. Scheduler picks from ELIGIBLE warps each cycle\n");
    printf("4. More eligible warps = better chance of finding work\n");
    printf("5. Profile with ncu to see stall reasons:\n");
    printf("   ncu --metrics smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct\n");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_data;
    
    return 0;
}
