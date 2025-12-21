/**
 * warp_reduce.cu - Efficient warp-level reductions
 * 
 * Learning objectives:
 * - Implement warp reduce with shuffles
 * - Compare to shared memory version
 * - Support multiple reduction ops
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cfloat>

#define FULL_MASK 0xffffffff

// ============================================================================
// Warp Reduction Primitives
// ============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(FULL_MASK, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_min(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fminf(val, __shfl_down_sync(FULL_MASK, val, offset));
    }
    return val;
}

// ============================================================================
// Block Reduction using Warp Primitives
// ============================================================================

__global__ void block_reduce_sum_warp(const float* in, float* out, int n) {
    __shared__ float warp_sums[32];  // Max 32 warps per block
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int num_warps = blockDim.x / 32;
    
    // Load value
    float val = (idx < n) ? in[idx] : 0.0f;
    
    // Warp-level reduction
    float warp_sum = warp_reduce_sum(val);
    
    // First lane of each warp writes to shared
    if (lane == 0) {
        warp_sums[warp_id] = warp_sum;
    }
    __syncthreads();
    
    // First warp reduces all warp sums
    if (warp_id == 0) {
        val = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        float block_sum = warp_reduce_sum(val);
        
        if (lane == 0) {
            atomicAdd(out, block_sum);
        }
    }
}

// Traditional shared memory reduction for comparison
__global__ void block_reduce_sum_shared(const float* in, float* out, int n) {
    extern __shared__ float smem[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Load
    smem[tid] = (idx < n) ? in[idx] : 0.0f;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(out, smem[0]);
    }
}

// ============================================================================
// Multi-operation reduction kernel
// ============================================================================

__global__ void warp_reduce_stats(const float* in, float* sum_out, 
                                   float* max_out, float* min_out, int n) {
    __shared__ float warp_sums[32];
    __shared__ float warp_maxs[32];
    __shared__ float warp_mins[32];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int num_warps = blockDim.x / 32;
    
    float val = (idx < n) ? in[idx] : 0.0f;
    float val_for_max = (idx < n) ? in[idx] : -FLT_MAX;
    float val_for_min = (idx < n) ? in[idx] : FLT_MAX;
    
    // Warp reductions
    float wsum = warp_reduce_sum(val);
    float wmax = warp_reduce_max(val_for_max);
    float wmin = warp_reduce_min(val_for_min);
    
    if (lane == 0) {
        warp_sums[warp_id] = wsum;
        warp_maxs[warp_id] = wmax;
        warp_mins[warp_id] = wmin;
    }
    __syncthreads();
    
    // Final reduction by first warp
    if (warp_id == 0) {
        float s = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        float mx = (lane < num_warps) ? warp_maxs[lane] : -FLT_MAX;
        float mn = (lane < num_warps) ? warp_mins[lane] : FLT_MAX;
        
        s = warp_reduce_sum(s);
        mx = warp_reduce_max(mx);
        mn = warp_reduce_min(mn);
        
        if (lane == 0) {
            atomicAdd(sum_out, s);
            // For max/min, use atomicMax/Min (need int reinterpretation)
            // Simplified: just store for single block
        }
    }
}

int main() {
    printf("=== Warp Reductions Demo ===\n\n");
    
    const int N = 1 << 20;  // 1M elements
    const int bytes = N * sizeof(float);
    const int block_size = 256;
    const int blocks = (N + block_size - 1) / block_size;
    
    // Allocate and initialize
    float* h_data = new float[N];
    float sum = 0;
    for (int i = 0; i < N; i++) {
        h_data[i] = 1.0f;  // Sum should be N
        sum += h_data[i];
    }
    
    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, sizeof(float));
    cudaMemcpy(d_in, h_data, bytes, cudaMemcpyHostToDevice);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    printf("Reducing %d elements (expected sum = %.0f)\n\n", N, sum);
    printf("%-30s | %10s | %10s\n", "Method", "Time (ms)", "Result");
    printf("─────────────────────────────────────────────────────\n");
    
    // Warp shuffle reduction
    {
        cudaMemset(d_out, 0, sizeof(float));
        
        cudaEventRecord(start);
        for (int i = 0; i < 100; i++) {
            cudaMemset(d_out, 0, sizeof(float));
            block_reduce_sum_warp<<<blocks, block_size>>>(d_in, d_out, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float result;
        cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost);
        
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        printf("%-30s | %10.4f | %10.0f\n", "Warp Shuffle Reduction", ms/100, result);
    }
    
    // Shared memory reduction
    {
        cudaMemset(d_out, 0, sizeof(float));
        
        cudaEventRecord(start);
        for (int i = 0; i < 100; i++) {
            cudaMemset(d_out, 0, sizeof(float));
            block_reduce_sum_shared<<<blocks, block_size, block_size * sizeof(float)>>>(d_in, d_out, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float result;
        cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost);
        
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        printf("%-30s | %10.4f | %10.0f\n", "Shared Memory Reduction", ms/100, result);
    }
    
    printf("\n=== Why Warp Shuffle is Better ===\n\n");
    printf("1. No __syncthreads() needed within warp\n");
    printf("2. Lower latency (~2 cycles vs ~30 cycles)\n");
    printf("3. No shared memory bank conflicts\n");
    printf("4. Less shared memory usage\n\n");
    
    printf("=== Warp Reduce Code Pattern ===\n\n");
    printf("__device__ float warp_reduce_sum(float val) {\n");
    printf("    for (int offset = 16; offset > 0; offset /= 2) {\n");
    printf("        val += __shfl_down_sync(0xffffffff, val, offset);\n");
    printf("    }\n");
    printf("    return val;  // Result valid in lane 0\n");
    printf("}\n");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_data;
    
    return 0;
}
