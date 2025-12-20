#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

#define BLOCK_SIZE 256

// =============================================================================
// TODO 1: Naive reduction with warp divergence
// Problem: if (tid % (2*s) == 0) causes divergent warps
// =============================================================================
__global__ void reduce_naive(float* out, const float* in, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load to shared memory
    sdata[tid] = (idx < n) ? in[idx] : 0.0f;
    __syncthreads();
    
    // TODO: Implement reduction with divergent warps
    // for (int s = 1; s < blockDim.x; s *= 2) {
    //     if (tid % (2*s) == 0) {
    //         sdata[tid] += sdata[tid + s];
    //     }
    //     __syncthreads();
    // }
    
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

// =============================================================================
// TODO 2: Sequential addressing (no warp divergence)
// Use: if (tid < s) instead of modulo
// =============================================================================
__global__ void reduce_sequential(float* out, const float* in, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < n) ? in[idx] : 0.0f;
    __syncthreads();
    
    // TODO: Implement with sequential addressing
    // for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    //     if (tid < s) {
    //         sdata[tid] += sdata[tid + s];
    //     }
    //     __syncthreads();
    // }
    
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

// =============================================================================
// TODO 3: Warp shuffle reduction
// Use __shfl_down_sync for final warp (no shared memory needed)
// =============================================================================
__device__ float warp_reduce(float val) {
    // TODO: Use __shfl_down_sync to reduce within warp
    // for (int offset = 16; offset > 0; offset >>= 1) {
    //     val += __shfl_down_sync(0xffffffff, val, offset);
    // }
    return val;
}

__global__ void reduce_warp_shuffle(float* out, const float* in, int n) {
    __shared__ float sdata[BLOCK_SIZE / 32];  // One element per warp
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;
    
    float val = (idx < n) ? in[idx] : 0.0f;
    
    // TODO: Warp-level reduction
    // val = warp_reduce(val);
    
    // TODO: First thread of each warp writes to shared
    // TODO: Final reduction of warp results
    
    if (tid == 0) out[blockIdx.x] = 0;  // Placeholder
}

int main() {
    const int N = 16 * 1024 * 1024;
    const int iterations = 100;
    
    printf("Parallel Reduction Exercise\n");
    printf("===========================\n");
    printf("Array size: %d elements\n\n", N);
    
    float *h_in, h_sum = 0;
    float *d_in, *d_out;
    
    h_in = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        h_in[i] = 1.0f;  // Easy to verify: sum should be N
        h_sum += h_in[i];
    }
    
    CHECK_CUDA(cudaMalloc(&d_in, N * sizeof(float)));
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    CHECK_CUDA(cudaMalloc(&d_out, numBlocks * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));
    
    printf("Expected sum: %.0f\n\n", h_sum);
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    const char* names[] = {"Naive", "Sequential", "Warp Shuffle"};
    void (*kernels[])(float*, const float*, int) = {
        reduce_naive, reduce_sequential, reduce_warp_shuffle
    };
    
    float* h_out = (float*)malloc(numBlocks * sizeof(float));
    
    for (int k = 0; k < 3; k++) {
        kernels[k]<<<numBlocks, BLOCK_SIZE>>>(d_out, d_in, N);
        CHECK_CUDA(cudaDeviceSynchronize());
        
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < iterations; i++) {
            kernels[k]<<<numBlocks, BLOCK_SIZE>>>(d_out, d_in, N);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        
        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        ms /= iterations;
        
        // Sum partial results on CPU
        CHECK_CUDA(cudaMemcpy(h_out, d_out, numBlocks * sizeof(float), cudaMemcpyDeviceToHost));
        float gpu_sum = 0;
        for (int i = 0; i < numBlocks; i++) gpu_sum += h_out[i];
        
        printf("%s: %.3f ms, sum=%.0f [%s]\n", 
               names[k], ms, gpu_sum, 
               (fabs(gpu_sum - h_sum) < 1) ? "PASS" : "FAIL");
    }
    
    free(h_in); free(h_out);
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    
    return 0;
}
