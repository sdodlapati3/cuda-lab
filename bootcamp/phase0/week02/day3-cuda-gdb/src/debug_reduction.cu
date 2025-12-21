/**
 * Day 3: Debug a Reduction
 * 
 * Practice stepping through a reduction kernel.
 */

#include <cstdio>
#include <cuda_runtime.h>

__global__ void warp_reduce(const int* input, int* output, int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load
    int val = (idx < n) ? input[idx] : 0;
    
    // Warp-level reduction - set breakpoints here!
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
        // Breakpoint: examine 'val' for different lanes
    }
    
    // Store warp sum
    __shared__ int warp_sums[8];  // For up to 256 threads (8 warps)
    
    if (tid % 32 == 0) {
        warp_sums[tid / 32] = val;
    }
    __syncthreads();
    
    // Final reduction in first warp
    if (tid < 8) {
        val = warp_sums[tid];
        for (int offset = 4; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xff, val, offset);
        }
        
        if (tid == 0) {
            output[blockIdx.x] = val;
        }
    }
}

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

int main() {
    printf("Reduction Debugging Practice\n");
    printf("============================\n");
    printf("\nTry these in cuda-gdb:\n");
    printf("  break warp_reduce\n");
    printf("  run\n");
    printf("  cuda thread (0,0,0)\n");
    printf("  print val\n");
    printf("  next  (step through shuffle iterations)\n");
    printf("  cuda thread (1,0,0)\n");
    printf("  print val\n\n");
    
    const int N = 256;
    int *d_input, *d_output;
    
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(int)));
    
    // Initialize: all 1s, so sum should be N
    int* h_input = new int[N];
    for (int i = 0; i < N; i++) h_input[i] = 1;
    CUDA_CHECK(cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice));
    
    warp_reduce<<<1, 256>>>(d_input, d_output, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    int result;
    CUDA_CHECK(cudaMemcpy(&result, d_output, sizeof(int), cudaMemcpyDeviceToHost));
    
    printf("Sum of %d ones: %d\n", N, result);
    printf("Expected: %d, Got: %d - %s\n", 
           N, result, (result == N) ? "CORRECT" : "WRONG");
    
    delete[] h_input;
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}
