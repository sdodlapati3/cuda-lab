#include <stdio.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

__global__ void cgReduce(float* input, float* output, int n) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    int tid = block.thread_rank();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load value (0 for out-of-bounds threads)
    float val = (idx < n) ? input[idx] : 0.0f;
    
    // Warp-level reduction using CG
    float warp_sum = cg::reduce(warp, val, cg::plus<float>());
    
    // Shared memory for warp sums
    __shared__ float warp_sums[32];  // Max 32 warps per block (1024 threads)
    
    // First thread of each warp stores warp sum
    if (warp.thread_rank() == 0) {
        warp_sums[tid / 32] = warp_sum;
    }
    block.sync();
    
    // First warp reduces all warp sums
    if (tid < 32) {
        int num_warps = (blockDim.x + 31) / 32;
        val = (tid < num_warps) ? warp_sums[tid] : 0.0f;
        float block_sum = cg::reduce(warp, val, cg::plus<float>());
        
        if (tid == 0) {
            output[blockIdx.x] = block_sum;
        }
    }
}

float reduce(float* d_input, int n) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    float* d_partial;
    cudaMalloc(&d_partial, numBlocks * sizeof(float));
    
    cgReduce<<<numBlocks, blockSize>>>(d_input, d_partial, n);
    
    // Reduce partial sums on CPU
    float* h_partial = new float[numBlocks];
    cudaMemcpy(h_partial, d_partial, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);
    
    float sum = 0;
    for (int i = 0; i < numBlocks; i++) {
        sum += h_partial[i];
    }
    
    delete[] h_partial;
    cudaFree(d_partial);
    
    return sum;
}

int main() {
    const int N = 1000000;
    
    float* h_data = new float[N];
    for (int i = 0; i < N; i++) {
        h_data[i] = 1.0f;
    }
    
    float* d_data;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    
    float result = reduce(d_data, N);
    
    printf("Sum of %d ones = %.0f (expected %d)\n", N, result, N);
    printf("Test %s\n", (result == N) ? "PASSED" : "FAILED");
    
    delete[] h_data;
    cudaFree(d_data);
    
    return (result == N) ? 0 : 1;
}
