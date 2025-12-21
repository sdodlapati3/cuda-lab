/**
 * Fused Layer Normalization - PyTorch CUDA Extension
 * 
 * LayerNorm(x) = (x - mean) / sqrt(var + eps) * weight + bias
 * 
 * This kernel fuses:
 * 1. Mean computation
 * 2. Variance computation  
 * 3. Normalization
 * 4. Affine transform (scale + shift)
 * 
 * Into a single kernel with online Welford algorithm.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ============================================================================
// Warp-level utilities
// ============================================================================
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Welford online algorithm state
struct WelfordState {
    float mean;
    float m2;    // sum of squared differences from mean
    int count;
    
    __device__ __forceinline__ WelfordState() : mean(0.0f), m2(0.0f), count(0) {}
    
    __device__ __forceinline__ void update(float x) {
        count++;
        float delta = x - mean;
        mean += delta / count;
        float delta2 = x - mean;
        m2 += delta * delta2;
    }
    
    __device__ __forceinline__ float variance() const {
        return count > 0 ? m2 / count : 0.0f;
    }
};

// Merge two Welford states (for parallel reduction)
__device__ __forceinline__ WelfordState welford_merge(WelfordState a, WelfordState b) {
    if (a.count == 0) return b;
    if (b.count == 0) return a;
    
    WelfordState result;
    result.count = a.count + b.count;
    float delta = b.mean - a.mean;
    result.mean = a.mean + delta * b.count / result.count;
    result.m2 = a.m2 + b.m2 + delta * delta * a.count * b.count / result.count;
    return result;
}

// Warp-level Welford reduction
__device__ __forceinline__ WelfordState warp_reduce_welford(WelfordState local) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        WelfordState other;
        other.mean = __shfl_down_sync(0xffffffff, local.mean, offset);
        other.m2 = __shfl_down_sync(0xffffffff, local.m2, offset);
        other.count = __shfl_down_sync(0xffffffff, local.count, offset);
        local = welford_merge(local, other);
    }
    return local;
}

// ============================================================================
// Forward Kernel
// ============================================================================
extern "C" __global__ void layernorm_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    float* __restrict__ mean_out,      // Save for backward
    float* __restrict__ rstd_out,      // Save for backward
    int batch_size,
    int hidden_size,
    float eps
) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;
    int num_warps = blockDim.x / 32;
    
    const float* row_in = input + batch_idx * hidden_size;
    float* row_out = output + batch_idx * hidden_size;
    
    extern __shared__ char smem[];
    WelfordState* warp_states = (WelfordState*)smem;
    
    // Compute mean and variance using Welford's algorithm
    WelfordState local;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        local.update(row_in[i]);
    }
    
    // Warp reduction
    local = warp_reduce_welford(local);
    if (lane == 0) {
        warp_states[warp_id] = local;
    }
    __syncthreads();
    
    // Block reduction (first warp)
    if (warp_id == 0) {
        local = (tid < num_warps) ? warp_states[lane] : WelfordState();
        local = warp_reduce_welford(local);
        if (lane == 0) {
            warp_states[0] = local;
        }
    }
    __syncthreads();
    
    // Broadcast mean and rstd
    float mean = warp_states[0].mean;
    float var = warp_states[0].variance();
    float rstd = rsqrtf(var + eps);
    
    // Save for backward
    if (tid == 0) {
        mean_out[batch_idx] = mean;
        rstd_out[batch_idx] = rstd;
    }
    
    // Normalize and apply affine transform
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float x_norm = (row_in[i] - mean) * rstd;
        row_out[i] = x_norm * weight[i] + bias[i];
    }
}

// ============================================================================
// Backward Kernel
// ============================================================================
extern "C" __global__ void layernorm_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    float* __restrict__ grad_input,
    float* __restrict__ grad_weight,  // Needs atomic or separate reduction
    float* __restrict__ grad_bias,    // Needs atomic or separate reduction
    int batch_size,
    int hidden_size
) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;
    int num_warps = blockDim.x / 32;
    
    const float* row_grad_out = grad_output + batch_idx * hidden_size;
    const float* row_in = input + batch_idx * hidden_size;
    float* row_grad_in = grad_input + batch_idx * hidden_size;
    
    float row_mean = mean[batch_idx];
    float row_rstd = rstd[batch_idx];
    
    extern __shared__ float sdata[];
    float* s_ds = sdata;
    float* s_db = sdata + 32;
    
    // Compute ds = sum(grad_out * weight * (x - mean))
    // Compute db = sum(grad_out * weight)
    float ds_local = 0.0f;
    float db_local = 0.0f;
    
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float x_norm = (row_in[i] - row_mean) * row_rstd;
        float dy = row_grad_out[i];
        ds_local += dy * weight[i] * x_norm;
        db_local += dy * weight[i];
        
        // Accumulate grad_weight and grad_bias (using atomics for simplicity)
        atomicAdd(&grad_weight[i], dy * x_norm);
        atomicAdd(&grad_bias[i], dy);
    }
    
    // Reduce ds and db
    ds_local = warp_reduce_sum(ds_local);
    db_local = warp_reduce_sum(db_local);
    
    if (lane == 0) {
        s_ds[warp_id] = ds_local;
        s_db[warp_id] = db_local;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        ds_local = (tid < num_warps) ? s_ds[lane] : 0.0f;
        db_local = (tid < num_warps) ? s_db[lane] : 0.0f;
        ds_local = warp_reduce_sum(ds_local);
        db_local = warp_reduce_sum(db_local);
        if (lane == 0) {
            s_ds[0] = ds_local;
            s_db[0] = db_local;
        }
    }
    __syncthreads();
    
    float ds = s_ds[0];
    float db = s_db[0];
    
    // Compute grad_input
    float inv_n = 1.0f / hidden_size;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float x_norm = (row_in[i] - row_mean) * row_rstd;
        float dy = row_grad_out[i];
        // grad_input = rstd * (dy * weight - inv_n * (db + x_norm * ds))
        row_grad_in[i] = row_rstd * (dy * weight[i] - inv_n * (db + x_norm * ds));
    }
}

// ============================================================================
// C++ wrapper functions (called from PyTorch)
// These would normally be in a separate .cpp file with pybind11
// ============================================================================

// Forward declaration for demonstration
extern "C" void layernorm_forward_cuda(
    const float* input,
    const float* weight, 
    const float* bias,
    float* output,
    float* mean,
    float* rstd,
    int batch_size,
    int hidden_size,
    float eps
) {
    int block_size = 256;
    int shared_size = (block_size / 32) * sizeof(WelfordState);
    
    layernorm_forward_kernel<<<batch_size, block_size, shared_size>>>(
        input, weight, bias, output, mean, rstd, batch_size, hidden_size, eps
    );
}

extern "C" void layernorm_backward_cuda(
    const float* grad_output,
    const float* input,
    const float* weight,
    const float* mean,
    const float* rstd,
    float* grad_input,
    float* grad_weight,
    float* grad_bias,
    int batch_size,
    int hidden_size
) {
    int block_size = 256;
    int shared_size = 64 * sizeof(float);  // For ds and db
    
    layernorm_backward_kernel<<<batch_size, block_size, shared_size>>>(
        grad_output, input, weight, mean, rstd,
        grad_input, grad_weight, grad_bias, batch_size, hidden_size
    );
}
