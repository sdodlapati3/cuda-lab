/**
 * Week 25, Day 5: Warp Specialization
 * Different warps for loading vs computing.
 */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>

#define CHECK_CUDA(call) { cudaError_t e = call; if(e) exit(1); }

// Warp-specialized GEMM concept demo
__global__ void warpSpecializedDemo(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[16][65], Bs[16][65];
    int warpId = threadIdx.x / 32;
    int tid = threadIdx.x;
    
    // Warp 0-1: loaders, Warp 2-7: computers (simplified demo)
    bool isLoader = (warpId < 2);
    
    if (isLoader) {
        // Loading code path
        for (int i = tid; i < 16*64; i += 64) {
            int k = i/64, m = i%64;
            As[k][m] = A[m*K + k];
        }
    }
    __syncthreads();
    
    // All warps compute
    int row = blockIdx.y * 64 + (tid / 16) * 4;
    int col = blockIdx.x * 64 + (tid % 16) * 4;
    float acc[4][4] = {{0}};
    
    for (int k = 0; k < 16; k++) {
        float a[4], b[4];
        for (int i = 0; i < 4; i++) a[i] = As[k][(tid/16)*4+i];
        for (int j = 0; j < 4; j++) b[j] = Bs[k][(tid%16)*4+j];
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                acc[i][j] += a[i] * b[j];
    }
    
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            if (row+i < M && col+j < N)
                C[(row+i)*N + col+j] = acc[i][j];
}

int main() {
    printf("Week 25 Day 5: Warp Specialization (concept demo)\n");
    printf("Concept: Different warps handle different tasks\n");
    printf("  - Loader warps: fetch data from global memory\n");
    printf("  - Compute warps: perform matrix multiply\n");
    printf("  - Enables overlapped execution\n");
    return 0;
}
