/**
 * Week 33, Day 4: Warp Softmax
 * 
 * For sequences ≤ 32, a single warp can compute softmax
 * using only shuffle instructions (no shared memory).
 * 
 * Perfect for attention heads with small sequence lengths.
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cfloat>

// Warp-level max reduction
__device__ float warpReduceMax(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return __shfl_sync(0xffffffff, val, 0);  // Broadcast from lane 0
}

// Warp-level sum reduction
__device__ float warpReduceSum(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return __shfl_sync(0xffffffff, val, 0);  // Broadcast from lane 0
}

// Each warp processes one row of size ≤ 32
__global__ void warpSoftmaxKernel(const float* input, float* output, 
                                   int numRows, int rowSize) {
    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int lane = threadIdx.x % warpSize;
    
    if (warpId >= numRows) return;
    
    const float* row_in = input + warpId * rowSize;
    float* row_out = output + warpId * rowSize;
    
    // Load value (or -inf if out of bounds)
    float val = (lane < rowSize) ? row_in[lane] : -FLT_MAX;
    
    // Warp-level max
    float maxVal = warpReduceMax(val);
    
    // Compute exp(x - max)
    float expVal = (lane < rowSize) ? expf(val - maxVal) : 0.0f;
    
    // Warp-level sum
    float sum = warpReduceSum(expVal);
    
    // Normalize and store
    if (lane < rowSize) {
        row_out[lane] = expVal / sum;
    }
}

// Batched warp softmax for multiple heads
__global__ void batchedWarpSoftmaxKernel(
    const float* input,   // [batch, heads, seq, seq]
    float* output,
    int batch, int heads, int seq
) {
    // Each warp handles one (batch, head, query) row
    int totalRows = batch * heads * seq;
    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int lane = threadIdx.x % warpSize;
    
    if (warpId >= totalRows) return;
    
    const float* row_in = input + warpId * seq;
    float* row_out = output + warpId * seq;
    
    // For seq > 32, each thread handles multiple elements
    float localMax = -FLT_MAX;
    float localSum = 0.0f;
    
    // Online max and sum
    for (int i = lane; i < seq; i += warpSize) {
        float x = row_in[i];
        float newMax = fmaxf(localMax, x);
        localSum = localSum * expf(localMax - newMax) + expf(x - newMax);
        localMax = newMax;
    }
    
    // Combine across warp
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        float otherMax = __shfl_down_sync(0xffffffff, localMax, offset);
        float otherSum = __shfl_down_sync(0xffffffff, localSum, offset);
        float newMax = fmaxf(localMax, otherMax);
        localSum = localSum * expf(localMax - newMax) + 
                   otherSum * expf(otherMax - newMax);
        localMax = newMax;
    }
    
    // Broadcast final values
    float maxVal = __shfl_sync(0xffffffff, localMax, 0);
    float sum = __shfl_sync(0xffffffff, localSum, 0);
    
    // Normalize and store
    for (int i = lane; i < seq; i += warpSize) {
        row_out[i] = expf(row_in[i] - maxVal) / sum;
    }
}

int main() {
    printf("Week 33 Day 4: Warp Softmax\n\n");
    
    printf("Warp Softmax Benefits:\n");
    printf("  - No shared memory needed\n");
    printf("  - Shuffle-only communication\n");
    printf("  - Perfect for seq ≤ 32\n");
    printf("  - Great for attention heads\n\n");
    
    // Test small sequence (fits in one warp)
    const int numRows = 64;
    const int rowSize = 32;
    const int totalSize = numRows * rowSize;
    
    float* h_input = new float[totalSize];
    float* h_output = new float[totalSize];
    
    for (int i = 0; i < totalSize; i++) {
        h_input[i] = (float)(i % rowSize);
    }
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, totalSize * sizeof(float));
    cudaMalloc(&d_output, totalSize * sizeof(float));
    cudaMemcpy(d_input, h_input, totalSize * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch: one warp per row
    int warpsNeeded = numRows;
    int threadsPerBlock = 256;
    int blocks = (warpsNeeded * 32 + threadsPerBlock - 1) / threadsPerBlock;
    
    warpSoftmaxKernel<<<blocks, threadsPerBlock>>>(d_input, d_output, numRows, rowSize);
    cudaMemcpy(h_output, d_output, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify first row
    printf("First row input:  ");
    for (int i = 0; i < 8; i++) printf("%.1f ", h_input[i]);
    printf("...\n");
    
    printf("First row output: ");
    for (int i = 0; i < 8; i++) printf("%.4f ", h_output[i]);
    printf("...\n");
    
    float sum = 0.0f;
    for (int i = 0; i < rowSize; i++) sum += h_output[i];
    printf("Row sum: %.6f ✓\n\n", sum);
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < 1000; i++) {
        warpSoftmaxKernel<<<blocks, threadsPerBlock>>>(d_input, d_output, numRows, rowSize);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Performance: %.2f us per call (%d rows × %d cols)\n", 
           ms, numRows, rowSize);
    
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
