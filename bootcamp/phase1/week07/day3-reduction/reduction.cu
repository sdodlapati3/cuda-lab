/**
 * reduction.cu - Parallel sum reduction
 * 
 * Learning objectives:
 * - Evolution of reduction algorithms
 * - Warp shuffle primitives
 * - Achieving high bandwidth
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define BLOCK_SIZE 256
#define FULL_MASK 0xFFFFFFFF

// Version 1: Naive interleaved reduction (has bank conflicts and divergence)
__global__ void reduce_v1_interleaved(const float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Interleaved addressing - bad!
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// Version 2: Sequential addressing (no divergence, but shared memory bank conflicts)
__global__ void reduce_v2_sequential(const float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Sequential addressing - better
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// Version 3: First add during load (halves blocks needed)
__global__ void reduce_v3_first_add(const float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    float sum = 0.0f;
    if (idx < n) sum += input[idx];
    if (idx + blockDim.x < n) sum += input[idx + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// Version 4: Unroll last warp (no sync needed within warp)
__device__ void warp_reduce(volatile float* sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void reduce_v4_unroll_warp(const float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    float sum = 0.0f;
    if (idx < n) sum += input[idx];
    if (idx + blockDim.x < n) sum += input[idx + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid < 32) {
        warp_reduce(sdata, tid);
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// Version 5: Warp shuffle (modern approach)
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

__global__ void reduce_v5_warp_shuffle(const float* input, float* output, int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;
    
    // Load and first add
    float sum = 0.0f;
    if (idx < n) sum += input[idx];
    if (idx + blockDim.x < n) sum += input[idx + blockDim.x];
    
    // Warp-level reduction
    sum = warp_reduce_sum(sum);
    
    // Write warp results to shared memory
    __shared__ float warp_sums[8];  // 256 threads / 32 = 8 warps
    if (lane == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();
    
    // Final reduction by first warp
    if (warp_id == 0) {
        sum = (tid < (blockDim.x / 32)) ? warp_sums[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (tid == 0) {
            output[blockIdx.x] = sum;
        }
    }
}

// Version 6: Grid-stride loop + atomic (single kernel)
__global__ void reduce_v6_atomic(const float* input, float* output, int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int lane = tid % 32;
    int warp_id = tid / 32;
    
    // Grid-stride accumulation
    float sum = 0.0f;
    for (int i = idx; i < n; i += stride) {
        sum += input[i];
    }
    
    // Warp reduction
    sum = warp_reduce_sum(sum);
    
    // Block reduction
    __shared__ float warp_sums[8];
    if (lane == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        sum = (tid < (blockDim.x / 32)) ? warp_sums[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (tid == 0) {
            atomicAdd(output, sum);
        }
    }
}

float reduce_cpu(const float* data, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    return (float)sum;
}

int main() {
    printf("=== Parallel Reduction Evolution ===\n\n");
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    
    float peak_bw = prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8) * 2 / 1e9;
    printf("Peak bandwidth: %.0f GB/s\n\n", peak_bw);
    
    const int N = 1 << 24;  // 16M elements
    const int TRIALS = 100;
    size_t bytes = N * sizeof(float);
    
    printf("Array size: %d elements (%.1f MB)\n\n", N, bytes / 1e6);
    
    // Allocate and initialize
    float* h_input = new float[N];
    for (int i = 0; i < N; i++) {
        h_input[i] = (rand() / (float)RAND_MAX) * 2 - 1;  // [-1, 1]
    }
    
    float cpu_result = reduce_cpu(h_input, N);
    printf("CPU result: %.6f\n\n", cpu_result);
    
    float *d_input, *d_output, *d_result;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, (N / BLOCK_SIZE + 1) * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));
    
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    printf("%-25s %-12s %-15s %-12s %-10s\n",
           "Version", "Time(ms)", "Bandwidth(GB/s)", "Efficiency", "Error");
    printf("--------------------------------------------------------------------------\n");
    
    auto benchmark = [&](const char* name, auto kernel_fn, int num_blocks, 
                        int shared_size, bool use_atomic = false) {
        float gpu_result = 0.0f;
        float ms;
        
        // Warmup
        if (use_atomic) {
            cudaMemset(d_result, 0, sizeof(float));
            kernel_fn<<<num_blocks, BLOCK_SIZE, shared_size>>>(d_input, d_result, N);
        } else {
            kernel_fn<<<num_blocks, BLOCK_SIZE, shared_size>>>(d_input, d_output, N);
        }
        cudaDeviceSynchronize();
        
        // Benchmark
        cudaEventRecord(start);
        for (int t = 0; t < TRIALS; t++) {
            if (use_atomic) {
                cudaMemset(d_result, 0, sizeof(float));
                kernel_fn<<<num_blocks, BLOCK_SIZE, shared_size>>>(d_input, d_result, N);
            } else {
                kernel_fn<<<num_blocks, BLOCK_SIZE, shared_size>>>(d_input, d_output, N);
            }
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        ms /= TRIALS;
        
        // Get result
        if (use_atomic) {
            cudaMemcpy(&gpu_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
        } else {
            // Need second pass for non-atomic versions (simplified here)
            float* h_partial = new float[num_blocks];
            cudaMemcpy(h_partial, d_output, num_blocks * sizeof(float), 
                      cudaMemcpyDeviceToHost);
            for (int i = 0; i < num_blocks; i++) {
                gpu_result += h_partial[i];
            }
            delete[] h_partial;
        }
        
        float bandwidth = bytes / ms / 1e6;  // GB/s
        float error = fabsf(gpu_result - cpu_result) / fabsf(cpu_result);
        
        printf("%-25s %-12.3f %-15.1f %-12.1f%% %-10.2e\n",
               name, ms, bandwidth, 100.0f * bandwidth / peak_bw, error);
    };
    
    int blocks_v1 = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int blocks_v2 = (N + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);
    int shared = BLOCK_SIZE * sizeof(float);
    
    benchmark("V1: Interleaved", reduce_v1_interleaved, blocks_v1, shared);
    benchmark("V2: Sequential", reduce_v2_sequential, blocks_v1, shared);
    benchmark("V3: First add", reduce_v3_first_add, blocks_v2, shared);
    benchmark("V4: Unroll warp", reduce_v4_unroll_warp, blocks_v2, shared);
    benchmark("V5: Warp shuffle", reduce_v5_warp_shuffle, blocks_v2, 8 * sizeof(float));
    benchmark("V6: Atomic (1 pass)", reduce_v6_atomic, 256, 8 * sizeof(float), true);
    
    printf("\n=== Key Insights ===\n");
    printf("1. V1 → V2: Remove divergence, sequential addressing\n");
    printf("2. V2 → V3: First add during load (halve blocks)\n");
    printf("3. V3 → V4: Unroll last warp (no sync needed)\n");
    printf("4. V4 → V5: Warp shuffle (no shared memory for warp)\n");
    printf("5. V6: Single kernel with atomics (simpler but atomic overhead)\n");
    printf("\nTarget: >70%% bandwidth efficiency for reduction\n");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_result);
    delete[] h_input;
    
    return 0;
}
