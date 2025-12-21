/**
 * shuffle_demo.cu - Warp shuffle operations
 * 
 * Learning objectives:
 * - __shfl_sync, __shfl_up_sync, __shfl_down_sync, __shfl_xor_sync
 * - Warp reduce and scan
 * - Register-to-register communication
 */

#include <cuda_runtime.h>
#include <cstdio>

#define FULL_MASK 0xFFFFFFFF
#define WARP_SIZE 32

// Basic shuffle demo
__global__ void shuffle_basic_demo() {
    int lane = threadIdx.x % 32;
    float val = (float)lane;  // Each lane has its ID as value
    
    if (threadIdx.x < 32) {
        // __shfl_sync: Get value from specific lane
        float from_lane_0 = __shfl_sync(FULL_MASK, val, 0);
        
        // __shfl_up_sync: Get value from lane - delta
        float from_left = __shfl_up_sync(FULL_MASK, val, 1);
        
        // __shfl_down_sync: Get value from lane + delta
        float from_right = __shfl_down_sync(FULL_MASK, val, 1);
        
        // __shfl_xor_sync: Get value from lane XOR mask (butterfly)
        float from_xor_1 = __shfl_xor_sync(FULL_MASK, val, 1);  // Swap pairs
        
        if (lane == 5) {
            printf("Lane 5:\n");
            printf("  My value: %.0f\n", val);
            printf("  From lane 0: %.0f\n", from_lane_0);
            printf("  From left (lane 4): %.0f\n", from_left);
            printf("  From right (lane 6): %.0f\n", from_right);
            printf("  From XOR 1 (lane 4): %.0f\n", from_xor_1);
        }
    }
}

// Warp reduce sum using __shfl_down_sync
__device__ float warp_reduce_sum(float val) {
    // Tree reduction: 16 steps down to 1
    val += __shfl_down_sync(FULL_MASK, val, 16);
    val += __shfl_down_sync(FULL_MASK, val, 8);
    val += __shfl_down_sync(FULL_MASK, val, 4);
    val += __shfl_down_sync(FULL_MASK, val, 2);
    val += __shfl_down_sync(FULL_MASK, val, 1);
    return val;  // Result in lane 0
}

// Warp reduce max
__device__ float warp_reduce_max(float val) {
    val = fmaxf(val, __shfl_down_sync(FULL_MASK, val, 16));
    val = fmaxf(val, __shfl_down_sync(FULL_MASK, val, 8));
    val = fmaxf(val, __shfl_down_sync(FULL_MASK, val, 4));
    val = fmaxf(val, __shfl_down_sync(FULL_MASK, val, 2));
    val = fmaxf(val, __shfl_down_sync(FULL_MASK, val, 1));
    return val;
}

// Warp inclusive scan (prefix sum)
__device__ float warp_scan_inclusive(float val) {
    int lane = threadIdx.x % 32;
    
    // Hillis-Steele style scan
    float n;
    n = __shfl_up_sync(FULL_MASK, val, 1);
    if (lane >= 1) val += n;
    
    n = __shfl_up_sync(FULL_MASK, val, 2);
    if (lane >= 2) val += n;
    
    n = __shfl_up_sync(FULL_MASK, val, 4);
    if (lane >= 4) val += n;
    
    n = __shfl_up_sync(FULL_MASK, val, 8);
    if (lane >= 8) val += n;
    
    n = __shfl_up_sync(FULL_MASK, val, 16);
    if (lane >= 16) val += n;
    
    return val;
}

// Broadcast from lane 0
__device__ float warp_broadcast(float val) {
    return __shfl_sync(FULL_MASK, val, 0);
}

// Butterfly exchange (for FFT-like algorithms)
__device__ float butterfly_exchange(float val, int partner_offset) {
    return __shfl_xor_sync(FULL_MASK, val, partner_offset);
}

__global__ void warp_operations_demo() {
    int lane = threadIdx.x % 32;
    float val = (float)(lane + 1);  // 1, 2, 3, ..., 32
    
    if (threadIdx.x < 32) {
        // Reduce sum
        float sum = warp_reduce_sum(val);
        if (lane == 0) printf("Warp sum: %.0f (expected: 528)\n", sum);
        
        // Reduce max
        float max_val = warp_reduce_max(val);
        if (lane == 0) printf("Warp max: %.0f (expected: 32)\n", max_val);
        
        // Scan
        float scan_result = warp_scan_inclusive(val);
        if (lane == 4) printf("Scan at lane 4: %.0f (expected: 15 = 1+2+3+4+5)\n", 
                              scan_result);
        
        // Broadcast
        float my_special = (lane == 7) ? 42.0f : 0.0f;
        float special_from_7 = __shfl_sync(FULL_MASK, my_special, 7);
        if (lane == 0) printf("Broadcast from lane 7: %.0f\n", special_from_7);
        
        // Butterfly (lane 0 <-> lane 1, lane 2 <-> lane 3, etc.)
        float partner_val = butterfly_exchange(val, 1);
        if (lane < 4) printf("Lane %d has %.0f, partner (^1) has %.0f\n", 
                             lane, val, partner_val);
    }
}

// Performance comparison: shuffle vs shared memory
__global__ void reduce_with_shuffle(const float* input, float* output, int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;
    
    float val = (idx < n) ? input[idx] : 0.0f;
    
    // Warp-level reduction (no shared memory!)
    val = warp_reduce_sum(val);
    
    // First lane of each warp stores to shared
    __shared__ float warp_sums[8];  // For 256 threads
    if (lane == 0) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();
    
    // Final reduction by first warp
    if (warp_id == 0) {
        val = (lane < 8) ? warp_sums[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0) {
            output[blockIdx.x] = val;
        }
    }
}

__global__ void reduce_with_shared(const float* input, float* output, int n) {
    __shared__ float sdata[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Traditional shared memory reduction
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

int main() {
    printf("=== Warp Shuffle Operations ===\n\n");
    
    printf("--- Basic Shuffle Demo ---\n");
    shuffle_basic_demo<<<1, 32>>>();
    cudaDeviceSynchronize();
    
    printf("\n--- Warp Operations Demo ---\n");
    warp_operations_demo<<<1, 32>>>();
    cudaDeviceSynchronize();
    
    printf("\n--- Performance Comparison ---\n");
    
    const int N = 1 << 20;
    const int TRIALS = 100;
    
    float* h_input = new float[N];
    for (int i = 0; i < N; i++) h_input[i] = 1.0f;
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, 4096 * sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;
    
    int blocks = (N + 255) / 256;
    
    // Warmup
    reduce_with_shuffle<<<blocks, 256>>>(d_input, d_output, N);
    reduce_with_shared<<<blocks, 256>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    // Shuffle version
    cudaEventRecord(start);
    for (int t = 0; t < TRIALS; t++) {
        reduce_with_shuffle<<<blocks, 256>>>(d_input, d_output, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("Shuffle reduction: %.3f ms\n", ms / TRIALS);
    
    // Shared memory version
    cudaEventRecord(start);
    for (int t = 0; t < TRIALS; t++) {
        reduce_with_shared<<<blocks, 256>>>(d_input, d_output, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("Shared memory reduction: %.3f ms\n", ms / TRIALS);
    
    printf("\n=== Key Insights ===\n");
    printf("1. Shuffle = register-to-register, no shared memory\n");
    printf("2. No __syncthreads() needed within warp\n");
    printf("3. Shuffle is faster for warp-local operations\n");
    printf("4. Use __shfl_down for reduction, __shfl_up for scan\n");
    printf("5. __shfl_xor for butterfly/FFT patterns\n");
    printf("6. Always use _sync suffix and provide mask!\n");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    
    return 0;
}
