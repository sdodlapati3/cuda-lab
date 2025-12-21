/**
 * Online Softmax: The Numerical Stability + Fusion Pattern
 * 
 * Softmax = exp(x - max(x)) / sum(exp(x - max(x)))
 * 
 * Naive approach: 3 passes over data
 *   1. Find max
 *   2. Compute exp(x - max) and sum
 *   3. Divide by sum
 * 
 * Online approach: 1 pass over data (Flash Attention uses this!)
 * 
 * Master this pattern → Understand FlashAttention's core insight
 * 
 * Progression:
 *   V1: Naive 3-pass               → 3× memory traffic
 *   V2: 2-pass (fused exp+sum)     → 2× memory traffic
 *   V3: Online 1-pass              → 1× memory traffic (optimal!)
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// Warp-level reduction utilities
// ============================================================================
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// ============================================================================
// V1: Naive 3-Pass Softmax
// 
// Memory traffic: 3 × N × sizeof(float)
// Why it's slow: Memory-bound operation reads data 3 times
// ============================================================================
__global__ void softmax_naive(const float* __restrict__ input,
                              float* __restrict__ output,
                              int batch_size, int dim) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    const float* row_in = input + batch_idx * dim;
    float* row_out = output + batch_idx * dim;
    
    extern __shared__ float shared[];
    float* s_max = shared;
    float* s_sum = shared + 1;
    
    // Pass 1: Find max (for numerical stability)
    float thread_max = -FLT_MAX;
    for (int i = tid; i < dim; i += blockDim.x) {
        thread_max = fmaxf(thread_max, row_in[i]);
    }
    thread_max = warp_reduce_max(thread_max);
    
    if (tid % 32 == 0) {
        atomicMax((int*)s_max, __float_as_int(thread_max));
    }
    __syncthreads();
    float row_max = *s_max;
    
    // Pass 2: Compute exp(x - max) and sum
    float thread_sum = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float val = expf(row_in[i] - row_max);
        row_out[i] = val;  // Store intermediate
        thread_sum += val;
    }
    thread_sum = warp_reduce_sum(thread_sum);
    
    if (tid % 32 == 0) {
        atomicAdd(s_sum, thread_sum);
    }
    __syncthreads();
    float row_sum = *s_sum;
    
    // Pass 3: Normalize
    for (int i = tid; i < dim; i += blockDim.x) {
        row_out[i] /= row_sum;
    }
}

// ============================================================================
// V2: 2-Pass Softmax (Fused exp+sum)
// 
// Memory traffic: 2 × N × sizeof(float)
// Improvement: Pass 2 and 3 merged by recomputing exp
// ============================================================================
__global__ void softmax_2pass(const float* __restrict__ input,
                              float* __restrict__ output,
                              int batch_size, int dim) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;
    int num_warps = blockDim.x / 32;
    
    const float* row_in = input + batch_idx * dim;
    float* row_out = output + batch_idx * dim;
    
    extern __shared__ float shared[];
    
    // Pass 1: Find max
    float thread_max = -FLT_MAX;
    for (int i = tid; i < dim; i += blockDim.x) {
        thread_max = fmaxf(thread_max, row_in[i]);
    }
    
    // Warp reduction for max
    thread_max = warp_reduce_max(thread_max);
    if (lane == 0) shared[warp_id] = thread_max;
    __syncthreads();
    
    // First warp reduces all warp maxes
    if (warp_id == 0) {
        thread_max = (tid < num_warps) ? shared[lane] : -FLT_MAX;
        thread_max = warp_reduce_max(thread_max);
        if (lane == 0) shared[0] = thread_max;
    }
    __syncthreads();
    float row_max = shared[0];
    
    // Pass 2: Compute exp, sum, and normalize in one pass
    // (We recompute exp during normalization - compute is cheap, memory is expensive!)
    float thread_sum = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        thread_sum += expf(row_in[i] - row_max);
    }
    
    // Warp reduction for sum
    thread_sum = warp_reduce_sum(thread_sum);
    if (lane == 0) shared[warp_id] = thread_sum;
    __syncthreads();
    
    if (warp_id == 0) {
        thread_sum = (tid < num_warps) ? shared[lane] : 0.0f;
        thread_sum = warp_reduce_sum(thread_sum);
        if (lane == 0) shared[0] = thread_sum;
    }
    __syncthreads();
    float row_sum = shared[0];
    
    // Normalize (recompute exp - this is the trade-off!)
    float inv_sum = 1.0f / row_sum;
    for (int i = tid; i < dim; i += blockDim.x) {
        row_out[i] = expf(row_in[i] - row_max) * inv_sum;
    }
}

// ============================================================================
// V3: Online Softmax (Single Pass!)
// 
// THE FlashAttention INSIGHT:
// 
// We can compute softmax incrementally by tracking:
//   - Running max
//   - Running sum (adjusted when max changes)
// 
// When we see a new max m' > m:
//   sum' = sum × exp(m - m') + exp(x - m')
// 
// Memory traffic: 1 × N × sizeof(float) for computing, then 1× for storing
// ============================================================================

// Online softmax state: tracks max and sum simultaneously
struct OnlineSoftmax {
    float max_val;
    float sum;
    
    __device__ __forceinline__ OnlineSoftmax() : max_val(-FLT_MAX), sum(0.0f) {}
    
    __device__ __forceinline__ void update(float x) {
        float new_max = fmaxf(max_val, x);
        // Adjust previous sum for new max, then add new element
        sum = sum * expf(max_val - new_max) + expf(x - new_max);
        max_val = new_max;
    }
    
    __device__ __forceinline__ void merge(const OnlineSoftmax& other) {
        float new_max = fmaxf(max_val, other.max_val);
        sum = sum * expf(max_val - new_max) + other.sum * expf(other.max_val - new_max);
        max_val = new_max;
    }
};

// Warp-level merge for OnlineSoftmax
__device__ __forceinline__ OnlineSoftmax warp_reduce_online(OnlineSoftmax local) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        OnlineSoftmax other;
        other.max_val = __shfl_down_sync(0xffffffff, local.max_val, offset);
        other.sum = __shfl_down_sync(0xffffffff, local.sum, offset);
        local.merge(other);
    }
    return local;
}

__global__ void softmax_online(const float* __restrict__ input,
                               float* __restrict__ output,
                               int batch_size, int dim) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;
    int num_warps = blockDim.x / 32;
    
    const float* row_in = input + batch_idx * dim;
    float* row_out = output + batch_idx * dim;
    
    extern __shared__ char smem[];
    OnlineSoftmax* warp_results = (OnlineSoftmax*)smem;
    
    // Single pass: accumulate online softmax state
    OnlineSoftmax local;
    for (int i = tid; i < dim; i += blockDim.x) {
        local.update(row_in[i]);
    }
    
    // Warp-level reduction
    local = warp_reduce_online(local);
    if (lane == 0) {
        warp_results[warp_id] = local;
    }
    __syncthreads();
    
    // Block-level reduction (first warp)
    if (warp_id == 0) {
        local = (tid < num_warps) ? warp_results[lane] : OnlineSoftmax();
        local = warp_reduce_online(local);
        if (lane == 0) {
            warp_results[0] = local;
        }
    }
    __syncthreads();
    
    // Broadcast final result
    float row_max = warp_results[0].max_val;
    float row_sum = warp_results[0].sum;
    float inv_sum = 1.0f / row_sum;
    
    // Final pass: compute output
    for (int i = tid; i < dim; i += blockDim.x) {
        row_out[i] = expf(row_in[i] - row_max) * inv_sum;
    }
}

// ============================================================================
// CPU Reference
// ============================================================================
void softmax_cpu(const float* input, float* output, int batch_size, int dim) {
    for (int b = 0; b < batch_size; b++) {
        const float* row_in = input + b * dim;
        float* row_out = output + b * dim;
        
        // Find max
        float max_val = row_in[0];
        for (int i = 1; i < dim; i++) {
            max_val = fmaxf(max_val, row_in[i]);
        }
        
        // Exp and sum
        float sum = 0.0f;
        for (int i = 0; i < dim; i++) {
            row_out[i] = expf(row_in[i] - max_val);
            sum += row_out[i];
        }
        
        // Normalize
        for (int i = 0; i < dim; i++) {
            row_out[i] /= sum;
        }
    }
}

// ============================================================================
// Benchmarking
// ============================================================================
typedef void (*SoftmaxKernel)(const float*, float*, int, int);

void benchmark_softmax(const char* name, SoftmaxKernel kernel,
                       const float* d_input, float* d_output,
                       int batch_size, int dim, float peak_bw,
                       int block_size, int warmup, int timed) {
    int shared_size = (block_size / 32) * sizeof(OnlineSoftmax);
    if (shared_size < 128) shared_size = 128;  // Minimum for atomics
    
    // Warmup
    for (int i = 0; i < warmup; i++) {
        kernel<<<batch_size, block_size, shared_size>>>(d_input, d_output, batch_size, dim);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Timed runs
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < timed; i++) {
        kernel<<<batch_size, block_size, shared_size>>>(d_input, d_output, batch_size, dim);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / timed;
    
    // Memory: read input + write output = 2 × size
    size_t bytes = 2 * (size_t)batch_size * dim * sizeof(float);
    double gb_per_s = (bytes / 1e9) / (avg_ms / 1e3);
    double efficiency = 100.0 * gb_per_s / peak_bw;
    
    printf("%-25s: %8.2f μs | %7.1f GB/s | %5.1f%% peak\n",
           name, avg_ms * 1000, gb_per_s, efficiency);
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main(int argc, char** argv) {
    int batch_size = (argc > 1) ? atoi(argv[1]) : 1024;
    int dim = (argc > 2) ? atoi(argv[2]) : 4096;
    
    printf("\n");
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║           ONLINE SOFTMAX BENCHMARK                             ║\n");
    printf("╠════════════════════════════════════════════════════════════════╣\n");
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    float peak_bw = 2.0f * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6;
    
    printf("║ Device: %-54s ║\n", prop.name);
    printf("║ Peak Bandwidth: %7.1f GB/s                                   ║\n", peak_bw);
    printf("║ Batch: %d, Dim: %d (%d MB)                             ║\n",
           batch_size, dim, (int)(batch_size * dim * sizeof(float) / (1024 * 1024)));
    printf("╠════════════════════════════════════════════════════════════════╣\n");
    
    // Allocate
    size_t size = (size_t)batch_size * dim * sizeof(float);
    float* h_input = (float*)malloc(size);
    float* h_output = (float*)malloc(size);
    float* h_reference = (float*)malloc(size);
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, size));
    
    // Initialize with random values
    for (int i = 0; i < batch_size * dim; i++) {
        h_input[i] = ((float)rand() / RAND_MAX - 0.5f) * 10.0f;  // Range [-5, 5]
    }
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
    
    // CPU reference
    softmax_cpu(h_input, h_reference, batch_size, dim);
    
    int block_size = 256;
    int warmup = 5;
    int timed = 20;
    
    // Benchmark
    benchmark_softmax("V1: Naive (3-pass)", softmax_naive,
                      d_input, d_output, batch_size, dim, peak_bw, block_size, warmup, timed);
    
    benchmark_softmax("V2: 2-pass (fused)", softmax_2pass,
                      d_input, d_output, batch_size, dim, peak_bw, block_size, warmup, timed);
    
    benchmark_softmax("V3: Online (1-pass)", softmax_online,
                      d_input, d_output, batch_size, dim, peak_bw, block_size, warmup, timed);
    
    // Verify V3
    softmax_online<<<batch_size, block_size, (block_size / 32) * sizeof(OnlineSoftmax)>>>
        (d_input, d_output, batch_size, dim);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
    
    // Check correctness
    float max_error = 0.0f;
    for (int i = 0; i < batch_size * dim; i++) {
        float error = fabsf(h_output[i] - h_reference[i]);
        max_error = fmaxf(max_error, error);
    }
    
    printf("╠════════════════════════════════════════════════════════════════╣\n");
    printf("║ Verification: Max error = %.2e %s                       ║\n",
           max_error, max_error < 1e-5 ? "✓" : "✗");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");
    
    // Cleanup
    free(h_input);
    free(h_output);
    free(h_reference);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    
    return 0;
}

/*
 * YOUR TURN: Exercises
 * 
 * 1. Implement online softmax for attention (not just row-wise)
 *    - Input: Q, K, V matrices
 *    - Output: Attention(Q, K, V)
 *    - This is the core of FlashAttention!
 * 
 * 2. Add causal masking to online softmax
 *    - Elements above diagonal → -inf before softmax
 * 
 * 3. Fuse softmax with the previous operation (e.g., QK^T)
 *    - Don't materialize the full attention matrix
 * 
 * 4. Profile and compare memory traffic
 *    - Use Nsight Compute to verify 1-pass vs 3-pass
 * 
 * 5. Implement for half precision (fp16/bf16)
 *    - Handle numerical stability carefully
 *    - Use __expf instead of expf for speed
 */
