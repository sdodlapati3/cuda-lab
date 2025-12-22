/**
 * Week 35, Day 5: Attention Softmax
 * 
 * After QK^T + mask, apply row-wise softmax to get attention weights.
 * Each query position gets a probability distribution over keys.
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#define BLOCK_SIZE 256

__device__ float warpReduceMax(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__device__ float warpReduceSum(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ float blockReduceMax(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    
    val = warpReduceMax(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[threadIdx.x] : -INFINITY;
    if (wid == 0) val = warpReduceMax(val);
    return val;
}

__device__ float blockReduceSum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    
    val = warpReduceSum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[threadIdx.x] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}

// Row-wise softmax for attention scores
// Each block handles one row (one query position)
__global__ void attentionSoftmax(
    float* __restrict__ scores,  // [num_rows, seq_k]
    int num_rows, int seq_k
) {
    __shared__ float s_max, s_sum;
    
    int row = blockIdx.x;
    if (row >= num_rows) return;
    
    float* row_scores = scores + row * seq_k;
    
    // 1. Find max (for numerical stability)
    float local_max = -INFINITY;
    for (int k = threadIdx.x; k < seq_k; k += blockDim.x) {
        local_max = fmaxf(local_max, row_scores[k]);
    }
    local_max = blockReduceMax(local_max);
    if (threadIdx.x == 0) s_max = local_max;
    __syncthreads();
    float row_max = s_max;
    
    // 2. Compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (int k = threadIdx.x; k < seq_k; k += blockDim.x) {
        float exp_val = expf(row_scores[k] - row_max);
        row_scores[k] = exp_val;  // Store temporarily
        local_sum += exp_val;
    }
    local_sum = blockReduceSum(local_sum);
    if (threadIdx.x == 0) s_sum = local_sum;
    __syncthreads();
    float row_sum = s_sum;
    
    // 3. Normalize
    float inv_sum = 1.0f / row_sum;
    for (int k = threadIdx.x; k < seq_k; k += blockDim.x) {
        row_scores[k] *= inv_sum;
    }
}

// Online softmax (single pass for later FlashAttention)
__global__ void onlineAttentionSoftmax(
    const float* __restrict__ scores_in,
    float* __restrict__ probs_out,
    int num_rows, int seq_k
) {
    int row = blockIdx.x;
    if (row >= num_rows) return;
    
    const float* in = scores_in + row * seq_k;
    float* out = probs_out + row * seq_k;
    
    // Single pass: track max and sum simultaneously
    float row_max = -INFINITY;
    float row_sum = 0.0f;
    
    // First pass: compute max and exp sum online
    for (int k = threadIdx.x; k < seq_k; k += blockDim.x) {
        float x = in[k];
        if (x > row_max) {
            row_sum = row_sum * expf(row_max - x) + 1.0f;
            row_max = x;
        } else {
            row_sum += expf(x - row_max);
        }
    }
    
    // Reduce across threads (simplified - full version needs parallel online combine)
    __shared__ float s_max, s_sum;
    // ... (for simplicity, using two-pass here in practice)
    
    // Second pass: normalize
    for (int k = threadIdx.x; k < seq_k; k += blockDim.x) {
        out[k] = expf(in[k] - row_max) / row_sum;
    }
}

int main() {
    printf("Week 35 Day 5: Attention Softmax\n\n");
    
    printf("Attention Weights = softmax(QK^T / sqrt(d_k))\n\n");
    
    printf("Row-wise Softmax:\n");
    printf("  Each query position → probability distribution over keys\n");
    printf("  Sum of each row = 1.0\n");
    printf("  Masked positions (-inf) → 0 after softmax\n\n");
    
    const int batch_heads = 48, seq_q = 256, seq_k = 256;
    const int num_rows = batch_heads * seq_q;
    
    printf("Config: %d total rows, %d columns per row\n\n", num_rows, seq_k);
    
    float *d_scores;
    cudaMalloc(&d_scores, num_rows * seq_k * sizeof(float));
    
    // Initialize with random scores + some masked (-inf) values
    float *h_scores = new float[num_rows * seq_k];
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < seq_k; j++) {
            // Simulate causal mask for demo
            if (j > (i % seq_q)) {
                h_scores[i * seq_k + j] = -INFINITY;
            } else {
                h_scores[i * seq_k + j] = (float)(rand() % 100) / 50.0f - 1.0f;
            }
        }
    }
    cudaMemcpy(d_scores, h_scores, num_rows * seq_k * sizeof(float), cudaMemcpyHostToDevice);
    
    // Apply softmax
    attentionSoftmax<<<num_rows, BLOCK_SIZE>>>(d_scores, num_rows, seq_k);
    cudaMemcpy(h_scores, d_scores, num_rows * seq_k * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify: check row sums
    printf("Verification (first 5 rows):\n");
    for (int i = 0; i < 5; i++) {
        float sum = 0.0f;
        for (int j = 0; j < seq_k; j++) sum += h_scores[i * seq_k + j];
        printf("  Row %d sum: %.6f %s\n", i, sum, fabsf(sum - 1.0f) < 1e-5 ? "✓" : "✗");
    }
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < 1000; i++) {
        attentionSoftmax<<<num_rows, BLOCK_SIZE>>>(d_scores, num_rows, seq_k);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("\nPerformance: %.2f us/call\n", ms);
    printf("Rows processed: %d\n", num_rows);
    printf("Throughput: %.2f rows/us\n", num_rows / ms);
    
    printf("\nKey Properties After Softmax:\n");
    printf("  • Sum of each row = 1.0\n");
    printf("  • All values in [0, 1]\n");
    printf("  • -inf inputs → 0 outputs\n");
    printf("  • Can interpret as attention probability\n");
    
    delete[] h_scores;
    cudaFree(d_scores);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
