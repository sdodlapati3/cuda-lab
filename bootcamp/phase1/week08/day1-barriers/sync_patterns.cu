/**
 * sync_patterns.cu - Common synchronization patterns
 * 
 * Learning objectives:
 * - Producer-consumer pattern
 * - Ping-pong buffers
 * - Double-buffering
 */

#include <cuda_runtime.h>
#include <cstdio>

#define BLOCK_SIZE 256

// Pattern 1: Load-Sync-Compute
// Classic shared memory usage
__global__ void load_sync_compute(const float* input, float* output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // LOAD phase
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    
    // SYNC - wait for all loads
    __syncthreads();
    
    // COMPUTE phase - can now access any sdata element
    float result = sdata[tid];
    if (tid > 0) result += sdata[tid - 1];
    if (tid < BLOCK_SIZE - 1) result += sdata[tid + 1];
    
    // STORE phase
    if (idx < n) {
        output[idx] = result;
    }
}

// Pattern 2: Multi-pass reduction
// Shows why sync is needed at EACH iteration
__global__ void multi_pass_reduction(const float* input, float* output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initial load
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Reduction passes - MUST sync after each!
    // Pass 1: stride 128
    if (tid < 128) sdata[tid] += sdata[tid + 128];
    __syncthreads();
    
    // Pass 2: stride 64
    if (tid < 64) sdata[tid] += sdata[tid + 64];
    __syncthreads();
    
    // Pass 3: stride 32
    if (tid < 32) sdata[tid] += sdata[tid + 32];
    __syncthreads();
    
    // Last warp - can use warp-level sync
    if (tid < 16) sdata[tid] += sdata[tid + 16];
    __syncwarp();
    if (tid < 8) sdata[tid] += sdata[tid + 8];
    __syncwarp();
    if (tid < 4) sdata[tid] += sdata[tid + 4];
    __syncwarp();
    if (tid < 2) sdata[tid] += sdata[tid + 2];
    __syncwarp();
    if (tid < 1) sdata[tid] += sdata[tid + 1];
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// Pattern 3: Ping-pong buffers
// Two buffers, alternate between them
__global__ void ping_pong_pattern(float* data, int iterations) {
    __shared__ float buf_a[BLOCK_SIZE];
    __shared__ float buf_b[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    
    // Initial load
    buf_a[tid] = data[tid];
    __syncthreads();
    
    float* read_buf = buf_a;
    float* write_buf = buf_b;
    
    for (int i = 0; i < iterations; i++) {
        // Read from one buffer, write to other
        float left = (tid > 0) ? read_buf[tid - 1] : 0;
        float center = read_buf[tid];
        float right = (tid < BLOCK_SIZE - 1) ? read_buf[tid + 1] : 0;
        
        write_buf[tid] = 0.25f * left + 0.5f * center + 0.25f * right;
        
        __syncthreads();  // Wait for all writes
        
        // Swap buffers
        float* temp = read_buf;
        read_buf = write_buf;
        write_buf = temp;
    }
    
    // Final result is in read_buf
    data[tid] = read_buf[tid];
}

// Pattern 4: Block-wide operations with conditional sync
// Shows how to handle variable-length operations
__global__ void block_wide_max(const float* input, float* output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    __shared__ int shared_n;  // Actual elements in block
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load
    float val = (idx < n) ? input[idx] : -INFINITY;
    sdata[tid] = val;
    
    // First thread computes how many valid elements
    if (tid == 0) {
        int remaining = n - blockIdx.x * blockDim.x;
        shared_n = (remaining > BLOCK_SIZE) ? BLOCK_SIZE : remaining;
    }
    
    __syncthreads();  // Sync both sdata AND shared_n
    
    // Max reduction
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < shared_n) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

int main() {
    printf("=== Synchronization Patterns ===\n\n");
    
    const int N = 256;
    float* h_data = new float[N];
    float* h_result = new float[N];
    
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)i;
    }
    
    float* d_data;
    float* d_result;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMalloc(&d_result, N * sizeof(float));
    
    printf("=== Pattern 1: Load-Sync-Compute ===\n");
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    load_sync_compute<<<1, BLOCK_SIZE>>>(d_data, d_result, N);
    cudaMemcpy(h_result, d_result, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("Input[5]: %.0f, Output[5]: %.0f (= [4] + [5] + [6])\n", 
           h_data[5], h_result[5]);
    printf("Expected: %.0f\n\n", h_data[4] + h_data[5] + h_data[6]);
    
    printf("=== Pattern 2: Multi-pass Reduction ===\n");
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    multi_pass_reduction<<<1, BLOCK_SIZE>>>(d_data, d_result, N);
    cudaMemcpy(h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    
    float cpu_sum = 0;
    for (int i = 0; i < N; i++) cpu_sum += h_data[i];
    printf("GPU sum: %.0f, CPU sum: %.0f\n\n", h_result[0], cpu_sum);
    
    printf("=== Pattern 3: Ping-Pong Buffers ===\n");
    for (int i = 0; i < N; i++) h_data[i] = (i == N/2) ? 100.0f : 0.0f;
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    
    ping_pong_pattern<<<1, BLOCK_SIZE>>>(d_data, 10);
    cudaMemcpy(h_result, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("Started with spike at center, after 10 iterations of smoothing:\n");
    printf("Center region: ");
    for (int i = N/2 - 3; i <= N/2 + 3; i++) {
        printf("%.1f ", h_result[i]);
    }
    printf("\n(Gaussian-like spread from center)\n\n");
    
    printf("=== Synchronization Rules ===\n");
    printf("1. LOAD-SYNC-COMPUTE: Sync after loading shared memory\n");
    printf("2. REDUCTION: Sync after EACH reduction step\n");
    printf("3. PING-PONG: Sync before swapping buffer pointers\n");
    printf("4. SHARED STATE: Sync after writing ANY shared variable\n");
    printf("\n");
    printf("Remember: When in doubt, sync. Correctness > Performance!\n");
    
    cudaFree(d_data);
    cudaFree(d_result);
    delete[] h_data;
    delete[] h_result;
    
    return 0;
}
