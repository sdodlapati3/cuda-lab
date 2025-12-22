/**
 * Week 36, Day 2: Standard MHA Implementation
 * 
 * Baseline implementation to understand the memory bottleneck.
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#define BLOCK_SIZE 256
#define TILE_SIZE 16

// Block reduce for softmax
__device__ float blockReduceMax(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    for (int offset = 16; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < blockDim.x / 32) ? shared[threadIdx.x] : -INFINITY;
    if (wid == 0) {
        for (int offset = 16; offset > 0; offset /= 2)
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ float blockReduceSum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < blockDim.x / 32) ? shared[threadIdx.x] : 0.0f;
    if (wid == 0) {
        for (int offset = 16; offset > 0; offset /= 2)
            val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Step 1: QK^T computation
__global__ void qktKernel(
    const float* Q, const float* K, float* S,
    int batch_heads, int seq, int d_k, float scale
) {
    int bh = blockIdx.z;
    int q_tile = blockIdx.y;
    int k_tile = blockIdx.x;
    
    __shared__ float Qs[TILE_SIZE][TILE_SIZE];
    __shared__ float Ks[TILE_SIZE][TILE_SIZE];
    
    int q_idx = q_tile * TILE_SIZE + threadIdx.y;
    int k_idx = k_tile * TILE_SIZE + threadIdx.x;
    
    const float* Q_ptr = Q + bh * seq * d_k;
    const float* K_ptr = K + bh * seq * d_k;
    float* S_ptr = S + bh * seq * seq;
    
    float sum = 0.0f;
    for (int t = 0; t < (d_k + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int d_off = t * TILE_SIZE;
        
        if (q_idx < seq && d_off + threadIdx.x < d_k)
            Qs[threadIdx.y][threadIdx.x] = Q_ptr[q_idx * d_k + d_off + threadIdx.x];
        else
            Qs[threadIdx.y][threadIdx.x] = 0;
            
        if (k_idx < seq && d_off + threadIdx.y < d_k)
            Ks[threadIdx.y][threadIdx.x] = K_ptr[k_idx * d_k + d_off + threadIdx.y];
        else
            Ks[threadIdx.y][threadIdx.x] = 0;
            
        __syncthreads();
        for (int i = 0; i < TILE_SIZE; i++)
            sum += Qs[threadIdx.y][i] * Ks[i][threadIdx.x];
        __syncthreads();
    }
    
    if (q_idx < seq && k_idx < seq)
        S_ptr[q_idx * seq + k_idx] = sum * scale;
}

// Step 2: Softmax
__global__ void softmaxKernel(float* S, int batch_heads, int seq) {
    __shared__ float s_max, s_sum;
    
    int bh = blockIdx.y;
    int row = blockIdx.x;
    
    float* row_ptr = S + bh * seq * seq + row * seq;
    
    // Find max
    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < seq; i += blockDim.x)
        local_max = fmaxf(local_max, row_ptr[i]);
    local_max = blockReduceMax(local_max);
    if (threadIdx.x == 0) s_max = local_max;
    __syncthreads();
    
    // Exp and sum
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < seq; i += blockDim.x) {
        float e = expf(row_ptr[i] - s_max);
        row_ptr[i] = e;
        local_sum += e;
    }
    local_sum = blockReduceSum(local_sum);
    if (threadIdx.x == 0) s_sum = local_sum;
    __syncthreads();
    
    // Normalize
    for (int i = threadIdx.x; i < seq; i += blockDim.x)
        row_ptr[i] /= s_sum;
}

// Step 3: PV output
__global__ void pvKernel(
    const float* P, const float* V, float* O,
    int batch_heads, int seq, int d_v
) {
    int bh = blockIdx.z;
    int q_tile = blockIdx.y;
    int d_tile = blockIdx.x;
    
    __shared__ float Ps[TILE_SIZE][TILE_SIZE];
    __shared__ float Vs[TILE_SIZE][TILE_SIZE];
    
    int q_idx = q_tile * TILE_SIZE + threadIdx.y;
    int d_idx = d_tile * TILE_SIZE + threadIdx.x;
    
    const float* P_ptr = P + bh * seq * seq;
    const float* V_ptr = V + bh * seq * d_v;
    float* O_ptr = O + bh * seq * d_v;
    
    float sum = 0.0f;
    for (int t = 0; t < (seq + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int k_off = t * TILE_SIZE;
        
        if (q_idx < seq && k_off + threadIdx.x < seq)
            Ps[threadIdx.y][threadIdx.x] = P_ptr[q_idx * seq + k_off + threadIdx.x];
        else
            Ps[threadIdx.y][threadIdx.x] = 0;
            
        if (k_off + threadIdx.y < seq && d_idx < d_v)
            Vs[threadIdx.y][threadIdx.x] = V_ptr[(k_off + threadIdx.y) * d_v + d_idx];
        else
            Vs[threadIdx.y][threadIdx.x] = 0;
            
        __syncthreads();
        for (int i = 0; i < TILE_SIZE; i++)
            sum += Ps[threadIdx.y][i] * Vs[i][threadIdx.x];
        __syncthreads();
    }
    
    if (q_idx < seq && d_idx < d_v)
        O_ptr[q_idx * d_v + d_idx] = sum;
}

void standardMHA(
    float* Q, float* K, float* V, float* O, float* S,
    int batch, int heads, int seq, int d_k, int d_v
) {
    int batch_heads = batch * heads;
    float scale = 1.0f / sqrtf((float)d_k);
    
    dim3 block_qkt(TILE_SIZE, TILE_SIZE);
    dim3 grid_qkt((seq + TILE_SIZE - 1) / TILE_SIZE, 
                   (seq + TILE_SIZE - 1) / TILE_SIZE, 
                   batch_heads);
    
    // Step 1: QK^T
    qktKernel<<<grid_qkt, block_qkt>>>(Q, K, S, batch_heads, seq, d_k, scale);
    
    // Step 2: Softmax
    softmaxKernel<<<dim3(seq, batch_heads), BLOCK_SIZE>>>(S, batch_heads, seq);
    
    // Step 3: PV
    dim3 grid_pv((d_v + TILE_SIZE - 1) / TILE_SIZE,
                  (seq + TILE_SIZE - 1) / TILE_SIZE,
                  batch_heads);
    pvKernel<<<grid_pv, block_qkt>>>(S, V, O, batch_heads, seq, d_v);
}

int main() {
    printf("Week 36 Day 2: Standard MHA Implementation\n\n");
    
    const int batch = 4, heads = 12, seq = 512, d_k = 64, d_v = 64;
    const int batch_heads = batch * heads;
    
    printf("Config: batch=%d, heads=%d, seq=%d, d_k=%d\n\n", batch, heads, seq, d_k);
    
    // Allocate
    float *d_Q, *d_K, *d_V, *d_O, *d_S;
    cudaMalloc(&d_Q, batch_heads * seq * d_k * sizeof(float));
    cudaMalloc(&d_K, batch_heads * seq * d_k * sizeof(float));
    cudaMalloc(&d_V, batch_heads * seq * d_v * sizeof(float));
    cudaMalloc(&d_O, batch_heads * seq * d_v * sizeof(float));
    cudaMalloc(&d_S, batch_heads * seq * seq * sizeof(float));  // The problem!
    
    printf("Memory allocated:\n");
    printf("  Q, K, V: %.2f MB each\n", batch_heads * seq * d_k * 4.0f / (1024*1024));
    printf("  S (attn): %.2f MB  ← O(seq²) bottleneck!\n", batch_heads * seq * seq * 4.0f / (1024*1024));
    printf("  O: %.2f MB\n\n", batch_heads * seq * d_v * 4.0f / (1024*1024));
    
    // Warmup
    standardMHA(d_Q, d_K, d_V, d_O, d_S, batch, heads, seq, d_k, d_v);
    cudaDeviceSynchronize();
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        standardMHA(d_Q, d_K, d_V, d_O, d_S, batch, heads, seq, d_k, d_v);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    printf("Performance:\n");
    printf("  Total time: %.2f ms (100 iters)\n", ms);
    printf("  Per call: %.2f us\n\n", ms * 10);
    
    printf("Memory Traffic Analysis:\n");
    long long qkt_read = 2LL * batch_heads * seq * d_k;  // Q and K
    long long qkt_write = (long long)batch_heads * seq * seq;  // S
    long long softmax_rw = 2LL * batch_heads * seq * seq;  // read + write S
    long long pv_read = (long long)batch_heads * seq * seq + batch_heads * seq * d_v;
    long long pv_write = (long long)batch_heads * seq * d_v;
    long long total = (qkt_read + qkt_write + softmax_rw + pv_read + pv_write) * 4;
    
    printf("  QK^T: read Q,K → write S\n");
    printf("  Softmax: read S → write S (in-place)\n");
    printf("  PV: read S,V → write O\n");
    printf("  Total: %.2f MB per call\n", total / (1024.0f * 1024.0f));
    printf("  S accessed 3 times! (write, softmax, read)\n\n");
    
    printf("FlashAttention insight:\n");
    printf("  Never materialize full S matrix\n");
    printf("  Compute attention in tiles, keep in SRAM\n");
    printf("  Trade recomputation for memory bandwidth\n");
    
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O); cudaFree(d_S);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    
    return 0;
}
