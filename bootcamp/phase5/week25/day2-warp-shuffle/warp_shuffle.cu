/**
 * Week 25, Day 2: Warp Shuffle Basics
 * 
 * Demonstrate warp shuffle operations for register communication.
 */

#include <cuda_runtime.h>
#include <cstdio>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// Demonstrate broadcast with __shfl_sync
__global__ void demonstrateBroadcast() {
    int laneId = threadIdx.x % 32;
    float myValue = (float)(laneId * 10);  // Each lane has different value
    
    // Broadcast from lane 0 to all lanes
    float broadcast = __shfl_sync(0xFFFFFFFF, myValue, 0);
    
    if (threadIdx.x < 32) {
        printf("Lane %2d: myValue=%3.0f, broadcast=%3.0f\n", 
               laneId, myValue, broadcast);
    }
}

// Demonstrate XOR shuffle for butterfly pattern
__global__ void demonstrateXOR() {
    int laneId = threadIdx.x % 32;
    float myValue = (float)laneId;
    
    // XOR with 1: swap with neighbor
    float xor1 = __shfl_xor_sync(0xFFFFFFFF, myValue, 1);
    
    // XOR with 2: swap with lane 2 away
    float xor2 = __shfl_xor_sync(0xFFFFFFFF, myValue, 2);
    
    if (threadIdx.x < 8) {
        printf("Lane %d: value=%2.0f, xor1=%2.0f, xor2=%2.0f\n",
               laneId, myValue, xor1, xor2);
    }
}

// Warp-level reduction using shuffle
__global__ void warpReduce(const float* input, float* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int laneId = threadIdx.x % 32;
    int warpId = threadIdx.x / 32;
    
    // Load value
    float val = (tid < n) ? input[tid] : 0.0f;
    
    // Warp reduction using shuffle
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    
    // Lane 0 of each warp has the sum
    if (laneId == 0) {
        atomicAdd(output, val);
    }
}

// Simple dot product using shuffle broadcast
__global__ void dotProductShuffle(const float* a, const float* b, 
                                   float* result, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int laneId = threadIdx.x % 32;
    
    float sum = 0.0f;
    
    // Each thread accumulates multiple elements
    for (int i = tid; i < n; i += gridDim.x * blockDim.x) {
        sum += a[i] * b[i];
    }
    
    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
    }
    
    // Lane 0 writes result
    if (laneId == 0) {
        atomicAdd(result, sum);
    }
}

// Broadcast A value across warp for GEMM
__global__ void gemmWithBroadcast(const float* A, const float* B, float* C,
                                   int M, int N, int K) {
    // Simplified: each warp computes one row of C
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int laneId = threadIdx.x % 32;
    
    if (row >= M) return;
    
    // Each lane handles different columns
    float sum[4] = {0, 0, 0, 0};  // 4 columns per lane
    
    for (int k = 0; k < K; k++) {
        // Lane 0 loads A[row, k], broadcast to all lanes
        float aVal = (laneId == 0) ? A[row * K + k] : 0.0f;
        aVal = __shfl_sync(0xFFFFFFFF, aVal, 0);
        
        // Each lane loads different B columns
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int col = laneId * 4 + j;
            if (col < N) {
                sum[j] += aVal * B[k * N + col];
            }
        }
    }
    
    // Write results
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        int col = laneId * 4 + j;
        if (col < N) {
            C[row * N + col] = sum[j];
        }
    }
}

int main() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║           Week 25, Day 2: Warp Shuffle Basics                ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    printf("1. Broadcast Demo (lane 0 → all lanes):\n");
    printf("─────────────────────────────────────────\n");
    demonstrateBroadcast<<<1, 32>>>();
    cudaDeviceSynchronize();
    printf("\n");
    
    printf("2. XOR Shuffle Demo (butterfly pattern):\n");
    printf("─────────────────────────────────────────\n");
    demonstrateXOR<<<1, 32>>>();
    cudaDeviceSynchronize();
    printf("\n");
    
    // Reduction test
    printf("3. Warp Reduction Test:\n");
    printf("─────────────────────────────────────────\n");
    int n = 1024;
    float *hInput = new float[n];
    float hExpected = 0;
    for (int i = 0; i < n; i++) {
        hInput[i] = 1.0f;  // Sum should be n
        hExpected += hInput[i];
    }
    
    float *dInput, *dOutput;
    CHECK_CUDA(cudaMalloc(&dInput, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dOutput, sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dInput, hInput, n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dOutput, 0, sizeof(float)));
    
    warpReduce<<<(n + 255) / 256, 256>>>(dInput, dOutput, n);
    cudaDeviceSynchronize();
    
    float result;
    CHECK_CUDA(cudaMemcpy(&result, dOutput, sizeof(float), cudaMemcpyDeviceToHost));
    printf("  Sum of %d ones: expected=%.0f, got=%.0f\n\n", n, hExpected, result);
    
    // Dot product test
    printf("4. Dot Product with Shuffle:\n");
    printf("─────────────────────────────────────────\n");
    float *hA = new float[n];
    float *hB = new float[n];
    float dotExpected = 0;
    for (int i = 0; i < n; i++) {
        hA[i] = (float)(i % 10);
        hB[i] = 1.0f;
        dotExpected += hA[i] * hB[i];
    }
    
    float *dA, *dB;
    CHECK_CUDA(cudaMalloc(&dA, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB, n * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dA, hA, n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dOutput, 0, sizeof(float)));
    
    dotProductShuffle<<<(n + 255) / 256, 256>>>(dA, dB, dOutput, n);
    cudaDeviceSynchronize();
    
    CHECK_CUDA(cudaMemcpy(&result, dOutput, sizeof(float), cudaMemcpyDeviceToHost));
    printf("  Dot product: expected=%.0f, got=%.0f\n\n", dotExpected, result);
    
    printf("Key Shuffle Operations:\n");
    printf("  __shfl_sync(mask, val, srcLane)     - Get val from srcLane\n");
    printf("  __shfl_xor_sync(mask, val, xorMask) - Butterfly exchange\n");
    printf("  __shfl_up_sync(mask, val, delta)    - Shift up (higher lanes)\n");
    printf("  __shfl_down_sync(mask, val, delta)  - Shift down (lower lanes)\n");
    
    cudaFree(dInput);
    cudaFree(dOutput);
    cudaFree(dA);
    cudaFree(dB);
    delete[] hInput;
    delete[] hA;
    delete[] hB;
    
    return 0;
}
