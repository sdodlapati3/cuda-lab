/**
 * Week 33, Day 5: Block Softmax
 * 
 * For larger sequences, use block-level reduction with shared memory.
 * Combines online algorithm with block-level parallelism.
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cfloat>

#define BLOCK_SIZE 256

// Block-level reduction for max
__device__ float blockReduceMax(float val) {
    __shared__ float shared[32];  // One per warp
    
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    
    // Warp reduction
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    
    // Write warp results to shared memory
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    // First warp reduces all warp results
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : -FLT_MAX;
    if (wid == 0) {
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
        }
    }
    
    return val;
}

// Block-level reduction for sum
__device__ float blockReduceSum(float val) {
    __shared__ float shared[32];
    
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;
    if (wid == 0) {
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
    }
    
    return val;
}

// One block per row softmax
__global__ void blockSoftmaxKernel(const float* input, float* output, 
                                    int numRows, int rowSize) {
    __shared__ float s_max;
    __shared__ float s_sum;
    
    int row = blockIdx.x;
    if (row >= numRows) return;
    
    const float* row_in = input + row * rowSize;
    float* row_out = output + row * rowSize;
    
    // Pass 1: Find max
    float localMax = -FLT_MAX;
    for (int i = threadIdx.x; i < rowSize; i += blockDim.x) {
        localMax = fmaxf(localMax, row_in[i]);
    }
    
    float maxVal = blockReduceMax(localMax);
    if (threadIdx.x == 0) s_max = maxVal;
    __syncthreads();
    maxVal = s_max;
    
    // Pass 2: Compute exp(x - max) and sum
    float localSum = 0.0f;
    for (int i = threadIdx.x; i < rowSize; i += blockDim.x) {
        float val = expf(row_in[i] - maxVal);
        row_out[i] = val;  // Store intermediate
        localSum += val;
    }
    
    float sum = blockReduceSum(localSum);
    if (threadIdx.x == 0) s_sum = sum;
    __syncthreads();
    sum = s_sum;
    
    // Pass 3: Normalize
    for (int i = threadIdx.x; i < rowSize; i += blockDim.x) {
        row_out[i] /= sum;
    }
}

// Optimized: fused pass with register tiling
__global__ void blockSoftmaxFusedKernel(const float* input, float* output,
                                         int numRows, int rowSize) {
    __shared__ float s_max;
    __shared__ float s_sum;
    
    int row = blockIdx.x;
    if (row >= numRows) return;
    
    const float* row_in = input + row * rowSize;
    float* row_out = output + row * rowSize;
    
    // Online max and sum in registers
    float localMax = -FLT_MAX;
    float localSum = 0.0f;
    
    for (int i = threadIdx.x; i < rowSize; i += blockDim.x) {
        float x = row_in[i];
        float newMax = fmaxf(localMax, x);
        localSum = localSum * expf(localMax - newMax) + expf(x - newMax);
        localMax = newMax;
    }
    
    // Block-level combine (simplified)
    float maxVal = blockReduceMax(localMax);
    if (threadIdx.x == 0) s_max = maxVal;
    __syncthreads();
    maxVal = s_max;
    
    // Rescale local sums to global max
    localSum = localSum * expf(localMax - maxVal);
    float sum = blockReduceSum(localSum);
    if (threadIdx.x == 0) s_sum = sum;
    __syncthreads();
    sum = s_sum;
    
    // Final normalization (one read from global)
    for (int i = threadIdx.x; i < rowSize; i += blockDim.x) {
        row_out[i] = expf(row_in[i] - maxVal) / sum;
    }
}

int main() {
    printf("Week 33 Day 5: Block Softmax\n\n");
    
    const int numRows = 128;
    const int rowSize = 1024;  // Larger sequence
    const int totalSize = numRows * rowSize;
    
    float* h_input = new float[totalSize];
    float* h_output = new float[totalSize];
    
    for (int i = 0; i < totalSize; i++) {
        h_input[i] = (float)(i % 100) / 10.0f;
    }
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, totalSize * sizeof(float));
    cudaMalloc(&d_output, totalSize * sizeof(float));
    cudaMemcpy(d_input, h_input, totalSize * sizeof(float), cudaMemcpyHostToDevice);
    
    // One block per row
    blockSoftmaxKernel<<<numRows, BLOCK_SIZE>>>(d_input, d_output, numRows, rowSize);
    cudaMemcpy(h_output, d_output, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify
    printf("Rows: %d, Cols: %d\n", numRows, rowSize);
    
    bool allValid = true;
    for (int r = 0; r < numRows; r++) {
        float sum = 0.0f;
        for (int c = 0; c < rowSize; c++) {
            sum += h_output[r * rowSize + c];
        }
        if (fabsf(sum - 1.0f) > 1e-5f) {
            printf("Row %d sum: %.6f (error!)\n", r, sum);
            allValid = false;
        }
    }
    if (allValid) printf("All rows sum to 1.0 ✓\n\n");
    
    // Benchmark both versions
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Three-pass version
    cudaEventRecord(start);
    for (int i = 0; i < 1000; i++) {
        blockSoftmaxKernel<<<numRows, BLOCK_SIZE>>>(d_input, d_output, numRows, rowSize);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms1;
    cudaEventElapsedTime(&ms1, start, stop);
    
    // Fused version
    cudaEventRecord(start);
    for (int i = 0; i < 1000; i++) {
        blockSoftmaxFusedKernel<<<numRows, BLOCK_SIZE>>>(d_input, d_output, numRows, rowSize);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms2;
    cudaEventElapsedTime(&ms2, start, stop);
    
    printf("Performance (1000 iterations):\n");
    printf("  Three-pass: %.2f ms (%.2f us/call)\n", ms1, ms1);
    printf("  Fused:      %.2f ms (%.2f us/call)\n", ms2, ms2);
    printf("  Speedup:    %.2fx\n", ms1/ms2);
    
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
