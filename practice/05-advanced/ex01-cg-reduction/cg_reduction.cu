#include <stdio.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

// TODO: Implement CG-based reduction kernel
// Requirements:
// 1. Use cg::reduce() for warp-level reduction
// 2. Accumulate warp results in shared memory
// 3. Final warp reduces the warp sums
__global__ void cgReduce(float* input, float* output, int n) {
    // TODO: Get thread block and warp tile groups
    
    // TODO: Calculate global index and load value
    
    // TODO: Warp-level reduction using cg::reduce()
    
    // TODO: Shared memory for warp sums
    
    // TODO: First thread of each warp stores warp sum
    
    // TODO: Synchronize block
    
    // TODO: First warp reduces all warp sums
    
    // TODO: Thread 0 writes block result
}

// Host function to perform complete reduction
float reduce(float* d_input, int n) {
    // TODO: Determine block and grid sizes
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    // TODO: Allocate memory for partial sums
    float* d_partial;
    cudaMalloc(&d_partial, numBlocks * sizeof(float));
    
    // TODO: Launch kernel
    cgReduce<<<numBlocks, blockSize>>>(d_input, d_partial, n);
    
    // TODO: Reduce partial sums on CPU (or recursively on GPU)
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
    
    // Allocate and initialize
    float* h_data = new float[N];
    for (int i = 0; i < N; i++) {
        h_data[i] = 1.0f;  // Sum should be N
    }
    
    float* d_data;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Perform reduction
    float result = reduce(d_data, N);
    
    printf("Sum of %d ones = %.0f (expected %d)\n", N, result, N);
    printf("Test %s\n", (result == N) ? "PASSED" : "FAILED");
    
    // Cleanup
    delete[] h_data;
    cudaFree(d_data);
    
    return (result == N) ? 0 : 1;
}
