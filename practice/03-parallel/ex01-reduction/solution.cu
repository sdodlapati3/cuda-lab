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

// SOLUTION: Naive reduction with warp divergence
__global__ void reduce_naive(float* out, const float* in, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < n) ? in[idx] : 0.0f;
    __syncthreads();
    
    // Interleaved addressing - causes divergent warps!
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2*s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

// SOLUTION: Sequential addressing (no warp divergence)
__global__ void reduce_sequential(float* out, const float* in, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < n) ? in[idx] : 0.0f;
    __syncthreads();
    
    // Sequential addressing - no divergence within warps
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

// SOLUTION: Warp reduce using shuffle
__device__ float warp_reduce(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// SOLUTION: Warp shuffle reduction
__global__ void reduce_warp_shuffle(float* out, const float* in, int n) {
    __shared__ float warp_sums[BLOCK_SIZE / 32];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;
    
    float val = (idx < n) ? in[idx] : 0.0f;
    
    // Warp-level reduction (no shared memory needed)
    val = warp_reduce(val);
    
    // First thread of each warp writes to shared
    if (lane == 0) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();
    
    // First warp reduces the warp sums
    if (warp_id == 0) {
        val = (tid < BLOCK_SIZE / 32) ? warp_sums[lane] : 0.0f;
        val = warp_reduce(val);
        
        if (tid == 0) {
            out[blockIdx.x] = val;
        }
    }
}

int main() {
    const int N = 16 * 1024 * 1024;
    const int iterations = 100;
    
    printf("Parallel Reduction - SOLUTION\n");
    printf("=============================\n");
    printf("Array size: %d elements\n\n", N);
    
    float *h_in, h_sum = 0;
    float *d_in, *d_out;
    
    h_in = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        h_in[i] = 1.0f;
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
        
        CHECK_CUDA(cudaMemcpy(h_out, d_out, numBlocks * sizeof(float), cudaMemcpyDeviceToHost));
        float gpu_sum = 0;
        for (int i = 0; i < numBlocks; i++) gpu_sum += h_out[i];
        
        float bw = N * sizeof(float) / (ms * 1e6);
        printf("%s: %.3f ms, %.1f GB/s, sum=%.0f [%s]\n", 
               names[k], ms, bw, gpu_sum, 
               (fabs(gpu_sum - h_sum) < 1) ? "PASS" : "FAIL");
    }
    
    free(h_in); free(h_out);
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    
    printf("\n✓ Solution complete!\n");
    return 0;
}
