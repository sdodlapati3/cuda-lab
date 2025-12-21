/**
 * Week 25, Day 4: Warp Reduction
 */
#include <cuda_runtime.h>
#include <cstdio>

__device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

__device__ float warpReduceMax(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    return val;
}

__global__ void testReduction() {
    int lane = threadIdx.x % 32;
    float val = (float)lane;
    
    float sum = warpReduceSum(val);
    float maxVal = warpReduceMax(val);
    
    if (lane == 0) {
        printf("Sum of 0-31: %.0f (expected 496)\n", sum);
        printf("Max of 0-31: %.0f (expected 31)\n", maxVal);
    }
}

int main() {
    printf("Week 25 Day 4: Warp Reduction\n");
    testReduction<<<1, 32>>>();
    cudaDeviceSynchronize();
    return 0;
}
