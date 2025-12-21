/**
 * Example kernels for benchmarking
 */

#include <cuda_runtime.h>

// ============================================================================
// Vector Add (memory-bound baseline)
// ============================================================================

__global__ void vector_add_naive(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Vectorized loads (float4)
__global__ void vector_add_vec4(const float4* a, const float4* b, float4* c, int n4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n4) {
        float4 va = a[idx];
        float4 vb = b[idx];
        c[idx] = make_float4(va.x + vb.x, va.y + vb.y, va.z + vb.z, va.w + vb.w);
    }
}

// ============================================================================
// Reduction (tree reduction)
// ============================================================================

__global__ void reduce_naive(const float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Naive tree reduction (with divergence)
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

__global__ void reduce_sequential(const float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Sequential addressing (no divergence, but bank conflicts)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

__global__ void reduce_warp_shuffle(const float* input, float* output, int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float val = (idx < n) ? input[idx] : 0.0f;
    
    // Warp-level reduction using shuffle
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    
    // First thread of each warp writes to shared memory
    __shared__ float warp_sums[32];  // Max 32 warps per block
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    if (lane_id == 0) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();
    
    // First warp reduces warp sums
    if (warp_id == 0) {
        val = (tid < blockDim.x / 32) ? warp_sums[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (tid == 0) output[blockIdx.x] = val;
    }
}
