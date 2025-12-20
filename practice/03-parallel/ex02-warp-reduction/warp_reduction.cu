#include <stdio.h>
#include <cuda_runtime.h>
#include <float.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// =============================================================================
// TODO 1: Sum reduction using __shfl_down_sync
// =============================================================================
__device__ float warp_reduce_sum(float val) {
    // TODO: Reduce using __shfl_down_sync
    // for (int offset = 16; offset > 0; offset >>= 1)
    //     val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// =============================================================================
// TODO 2: Sum reduction using __shfl_xor_sync (butterfly pattern)
// =============================================================================
__device__ float warp_reduce_sum_xor(float val) {
    // TODO: Reduce using butterfly pattern
    // for (int mask = 16; mask > 0; mask >>= 1)
    //     val += __shfl_xor_sync(0xffffffff, val, mask);
    return val;
}

// =============================================================================
// TODO 3: Min reduction using warp shuffles
// =============================================================================
__device__ float warp_reduce_min(float val) {
    // TODO: Use fminf with shuffle
    return val;
}

// =============================================================================
// TODO 4: Max reduction using warp shuffles
// =============================================================================
__device__ float warp_reduce_max(float val) {
    // TODO: Use fmaxf with shuffle
    return val;
}

__global__ void test_warp_reduce(float* out_sum, float* out_min, float* out_max, const float* in, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float val = (idx < n) ? in[idx] : 0.0f;
    float val_min = (idx < n) ? in[idx] : FLT_MAX;
    float val_max = (idx < n) ? in[idx] : -FLT_MAX;
    
    float sum = warp_reduce_sum(val);
    float minimum = warp_reduce_min(val_min);
    float maximum = warp_reduce_max(val_max);
    
    if (threadIdx.x % 32 == 0) {
        atomicAdd(out_sum, sum);
        atomicMin((int*)out_min, __float_as_int(minimum));
        atomicMax((int*)out_max, __float_as_int(maximum));
    }
}

int main() {
    const int N = 1024;
    
    printf("Warp Reduction Exercise\n");
    printf("=======================\n");
    
    float* h_in = (float*)malloc(N * sizeof(float));
    float h_sum = 0, h_min = FLT_MAX, h_max = -FLT_MAX;
    
    for (int i = 0; i < N; i++) {
        h_in[i] = (float)(i % 100);
        h_sum += h_in[i];
        h_min = fminf(h_min, h_in[i]);
        h_max = fmaxf(h_max, h_in[i]);
    }
    
    printf("Expected: sum=%.0f, min=%.0f, max=%.0f\n\n", h_sum, h_min, h_max);
    
    float *d_in, *d_sum, *d_min, *d_max;
    CHECK_CUDA(cudaMalloc(&d_in, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_sum, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_min, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_max, sizeof(float)));
    
    CHECK_CUDA(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_sum, 0, sizeof(float)));
    float init_min = FLT_MAX, init_max = -FLT_MAX;
    CHECK_CUDA(cudaMemcpy(d_min, &init_min, sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_max, &init_max, sizeof(float), cudaMemcpyHostToDevice));
    
    test_warp_reduce<<<(N+255)/256, 256>>>(d_sum, d_min, d_max, d_in, N);
    
    float gpu_sum, gpu_min, gpu_max;
    CHECK_CUDA(cudaMemcpy(&gpu_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&gpu_min, d_min, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&gpu_max, d_max, sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("GPU: sum=%.0f, min=%.0f, max=%.0f\n", gpu_sum, gpu_min, gpu_max);
    
    free(h_in);
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_sum));
    CHECK_CUDA(cudaFree(d_min));
    CHECK_CUDA(cudaFree(d_max));
    
    return 0;
}
