/**
 * warp_divergence.cu - Demonstrate and measure warp divergence
 * 
 * Learning objectives:
 * - See performance impact of divergence
 * - Learn to write divergence-free code
 * - Profile divergence with Nsight
 */

#include <cuda_runtime.h>
#include <cstdio>

// Highly divergent: every thread takes different path
__global__ void highly_divergent(float* data, int n, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    float val = data[idx];
    
    for (int iter = 0; iter < iterations; iter++) {
        // Each thread lane takes a different branch!
        // This is the WORST case for divergence
        int lane = threadIdx.x % 32;
        switch (lane % 8) {
            case 0: val = val * 1.1f; break;
            case 1: val = val * 1.2f; break;
            case 2: val = val * 1.3f; break;
            case 3: val = val * 1.4f; break;
            case 4: val = val + 0.1f; break;
            case 5: val = val + 0.2f; break;
            case 6: val = val + 0.3f; break;
            case 7: val = val + 0.4f; break;
        }
    }
    data[idx] = val;
}

// Non-divergent: all threads in warp take same path
__global__ void non_divergent(float* data, int n, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    float val = data[idx];
    
    for (int iter = 0; iter < iterations; iter++) {
        // All threads in a warp take the SAME branch
        // because we branch on warp_id, not lane
        int warp_id = threadIdx.x / 32;
        switch (warp_id % 8) {
            case 0: val = val * 1.1f; break;
            case 1: val = val * 1.2f; break;
            case 2: val = val * 1.3f; break;
            case 3: val = val * 1.4f; break;
            case 4: val = val + 0.1f; break;
            case 5: val = val + 0.2f; break;
            case 6: val = val + 0.3f; break;
            case 7: val = val + 0.4f; break;
        }
    }
    data[idx] = val;
}

// Medium divergence: half warp diverges
__global__ void half_divergent(float* data, int n, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    float val = data[idx];
    
    for (int iter = 0; iter < iterations; iter++) {
        int lane = threadIdx.x % 32;
        if (lane < 16) {
            val = val * 1.1f;
            val = val + 0.5f;
        } else {
            val = val / 1.1f;
            val = val - 0.5f;
        }
    }
    data[idx] = val;
}

int main() {
    printf("=== Warp Divergence Benchmark ===\n\n");
    
    const int N = 1 << 20;  // 1M elements
    const int ITERATIONS = 1000;
    
    float* d_data;
    cudaMalloc(&d_data, N * sizeof(float));
    
    int block_size = 256;  // 8 warps per block
    int num_blocks = (N + block_size - 1) / block_size;
    
    printf("Array size: %d elements\n", N);
    printf("Block size: %d threads (%d warps/block)\n", block_size, block_size / 32);
    printf("Iterations: %d\n\n", ITERATIONS);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    float ms;
    
    // Warm up
    highly_divergent<<<num_blocks, block_size>>>(d_data, N, 10);
    cudaDeviceSynchronize();
    
    // Benchmark highly divergent
    cudaMemset(d_data, 0, N * sizeof(float));
    cudaEventRecord(start);
    highly_divergent<<<num_blocks, block_size>>>(d_data, N, ITERATIONS);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("Highly divergent (8-way):    %8.3f ms\n", ms);
    float baseline = ms;
    
    // Benchmark half divergent
    cudaMemset(d_data, 0, N * sizeof(float));
    cudaEventRecord(start);
    half_divergent<<<num_blocks, block_size>>>(d_data, N, ITERATIONS);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("Half divergent (2-way):      %8.3f ms (%.2fx faster)\n", ms, baseline / ms);
    
    // Benchmark non-divergent
    cudaMemset(d_data, 0, N * sizeof(float));
    cudaEventRecord(start);
    non_divergent<<<num_blocks, block_size>>>(d_data, N, ITERATIONS);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("Non-divergent (warp-uniform):%8.3f ms (%.2fx faster)\n", ms, baseline / ms);
    
    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("\n=== Key Insights ===\n");
    printf("1. Divergence serializes execution paths\n");
    printf("2. 8-way divergence ≈ 8× slower than uniform\n");
    printf("3. Branch on warp_id (threadIdx.x / 32), not lane\n");
    printf("4. Use Nsight Compute to see 'Warp State' stalls\n");
    printf("\n");
    printf("Profile command:\n");
    printf("  ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active ./build/warp_divergence\n");
    
    return 0;
}
