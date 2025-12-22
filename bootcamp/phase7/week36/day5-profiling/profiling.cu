/**
 * Week 36, Day 5: Profiling Standard MHA
 * 
 * Key metrics to look for:
 * - Memory throughput (should be close to peak if memory bound)
 * - Compute utilization (should be low if memory bound)
 * - L2 cache hit rate
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#define BLOCK_SIZE 256

__device__ float blockReduceMax(float val) {
    __shared__ float s[32];
    int lane = threadIdx.x % 32, wid = threadIdx.x / 32;
    for (int o = 16; o > 0; o /= 2) val = fmaxf(val, __shfl_down_sync(0xffffffff, val, o));
    if (lane == 0) s[wid] = val;
    __syncthreads();
    val = threadIdx.x < blockDim.x / 32 ? s[threadIdx.x] : -INFINITY;
    if (wid == 0) for (int o = 16; o > 0; o /= 2) val = fmaxf(val, __shfl_down_sync(0xffffffff, val, o));
    return val;
}

__device__ float blockReduceSum(float val) {
    __shared__ float s[32];
    int lane = threadIdx.x % 32, wid = threadIdx.x / 32;
    for (int o = 16; o > 0; o /= 2) val += __shfl_down_sync(0xffffffff, val, o);
    if (lane == 0) s[wid] = val;
    __syncthreads();
    val = threadIdx.x < blockDim.x / 32 ? s[threadIdx.x] : 0;
    if (wid == 0) for (int o = 16; o > 0; o /= 2) val += __shfl_down_sync(0xffffffff, val, o);
    return val;
}

// Instrumented kernels with markers
__global__ void qktKernelProfile(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    float* __restrict__ S,
    int seq, int d, float scale
) {
    int q = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (q < seq && k < seq) {
        float sum = 0.0f;
        for (int i = 0; i < d; i++) {
            sum += Q[q * d + i] * K[k * d + i];
        }
        S[q * seq + k] = sum * scale;
    }
}

__global__ void softmaxProfile(float* S, int seq) {
    __shared__ float s_max, s_sum;
    int row = blockIdx.x;
    float* row_ptr = S + row * seq;
    
    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < seq; i += blockDim.x)
        local_max = fmaxf(local_max, row_ptr[i]);
    local_max = blockReduceMax(local_max);
    if (threadIdx.x == 0) s_max = local_max;
    __syncthreads();
    
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < seq; i += blockDim.x) {
        float e = expf(row_ptr[i] - s_max);
        row_ptr[i] = e;
        local_sum += e;
    }
    local_sum = blockReduceSum(local_sum);
    if (threadIdx.x == 0) s_sum = local_sum;
    __syncthreads();
    
    for (int i = threadIdx.x; i < seq; i += blockDim.x)
        row_ptr[i] /= s_sum;
}

__global__ void pvKernelProfile(
    const float* __restrict__ P,
    const float* __restrict__ V,
    float* __restrict__ O,
    int seq, int d
) {
    int q = blockIdx.y * blockDim.y + threadIdx.y;
    int di = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (q < seq && di < d) {
        float sum = 0.0f;
        for (int k = 0; k < seq; k++) {
            sum += P[q * seq + k] * V[k * d + di];
        }
        O[q * d + di] = sum;
    }
}

int main() {
    printf("Week 36 Day 5: Profiling Standard MHA\n\n");
    
    printf("How to Profile with Nsight:\n");
    printf("  ncu --target-processes all ./profiling\n");
    printf("  ncu --set full ./profiling\n\n");
    
    printf("Key Metrics to Examine:\n");
    printf("  ┌──────────────────────────────────────────────────────────────┐\n");
    printf("  │ Metric                 │ Memory Bound │ Compute Bound        │\n");
    printf("  ├──────────────────────────────────────────────────────────────┤\n");
    printf("  │ DRAM Throughput        │ ~90%% peak    │ <50%% peak           │\n");
    printf("  │ SM Compute Utilization │ <50%%         │ ~80%%+               │\n");
    printf("  │ Achieved Occupancy     │ Can be high  │ High                 │\n");
    printf("  │ Warp Stall (Memory)    │ High         │ Low                  │\n");
    printf("  └──────────────────────────────────────────────────────────────┘\n\n");
    
    const int seq = 1024, d = 64;
    
    float *d_Q, *d_K, *d_V, *d_S, *d_O;
    cudaMalloc(&d_Q, seq * d * sizeof(float));
    cudaMalloc(&d_K, seq * d * sizeof(float));
    cudaMalloc(&d_V, seq * d * sizeof(float));
    cudaMalloc(&d_S, seq * seq * sizeof(float));
    cudaMalloc(&d_O, seq * d * sizeof(float));
    
    float scale = 1.0f / sqrtf((float)d);
    dim3 block2d(16, 16);
    dim3 grid_qkt((seq + 15) / 16, (seq + 15) / 16);
    dim3 grid_pv((d + 15) / 16, (seq + 15) / 16);
    
    printf("Running kernels for profiling...\n\n");
    
    // Warm up
    for (int i = 0; i < 10; i++) {
        qktKernelProfile<<<grid_qkt, block2d>>>(d_Q, d_K, d_S, seq, d, scale);
        softmaxProfile<<<seq, BLOCK_SIZE>>>(d_S, seq);
        pvKernelProfile<<<grid_pv, block2d>>>(d_S, d_V, d_O, seq, d);
    }
    cudaDeviceSynchronize();
    
    // Timed run
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;
    
    // Time each kernel separately
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++)
        qktKernelProfile<<<grid_qkt, block2d>>>(d_Q, d_K, d_S, seq, d, scale);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("QK^T kernel: %.2f us (%.1f GB/s memory)\n", ms * 10,
           (2.0 * seq * d + seq * seq) * 4 * 100 / ms / 1e6);
    
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++)
        softmaxProfile<<<seq, BLOCK_SIZE>>>(d_S, seq);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("Softmax kernel: %.2f us (%.1f GB/s memory)\n", ms * 10,
           2.0 * seq * seq * 4 * 100 / ms / 1e6);
    
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++)
        pvKernelProfile<<<grid_pv, block2d>>>(d_S, d_V, d_O, seq, d);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("PV kernel: %.2f us (%.1f GB/s memory)\n", ms * 10,
           (seq * seq + seq * d + seq * d) * 4 * 100 / ms / 1e6);
    
    printf("\nExpected Observation:\n");
    printf("  All kernels should show high memory throughput\n");
    printf("  and low compute utilization = MEMORY BOUND\n\n");
    
    printf("Why FlashAttention Helps:\n");
    printf("  1. Fuses all 3 kernels → fewer kernel launches\n");
    printf("  2. Never writes S to HBM → massive bandwidth savings\n");
    printf("  3. Higher arithmetic intensity → compute bound\n");
    
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_S); cudaFree(d_O);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    
    return 0;
}
