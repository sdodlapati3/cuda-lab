/**
 * Day 5: Column-Tiled GEMM
 * 
 * Demonstrates caching columns of B in shared memory.
 * Complements row-tiling to show need for 2D tiling.
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// Reference naive GEMM
__global__ void naiveGemm(const float* __restrict__ A,
                          const float* __restrict__ B,
                          float* __restrict__ C,
                          int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Column-Tiled GEMM: Cache columns of B in shared memory
// Each block processes entire A against TILE_N columns of B
template<int TILE_N, int BLOCK_SIZE>
__global__ void colTiledGemm(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int M, int N, int K) {
    // Shared memory for tile of B columns
    extern __shared__ float sharedB[];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int colStart = blockIdx.x * TILE_N;
    int localCol = threadIdx.x;
    int col = colStart + localCol;
    
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int numThreads = blockDim.x * blockDim.y;
    
    // Cooperatively load TILE_N columns of B (K × TILE_N elements)
    int totalElements = K * TILE_N;
    for (int i = tid; i < totalElements; i += numThreads) {
        int kIdx = i / TILE_N;
        int localColIdx = i % TILE_N;
        int globalCol = colStart + localColIdx;
        if (globalCol < N) {
            sharedB[kIdx * TILE_N + localColIdx] = B[kIdx * N + globalCol];
        } else {
            sharedB[kIdx * TILE_N + localColIdx] = 0.0f;
        }
    }
    __syncthreads();
    
    // Each thread computes one element using cached B
    if (row < M && col < N) {
        float sum = 0.0f;
        
        // A from global (slow), B from shared (fast)
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * sharedB[k * TILE_N + localCol];
        }
        C[row * N + col] = sum;
    }
}

// Unrolled version
template<int TILE_N>
__global__ void colTiledGemmUnroll(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int M, int N, int K) {
    extern __shared__ float sharedB[];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int colStart = blockIdx.x * TILE_N;
    int localCol = threadIdx.x;
    int col = colStart + localCol;
    
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int numThreads = blockDim.x * blockDim.y;
    
    // Load tile of B
    int totalElements = K * TILE_N;
    for (int i = tid; i < totalElements; i += numThreads) {
        int kIdx = i / TILE_N;
        int localColIdx = i % TILE_N;
        int globalCol = colStart + localColIdx;
        sharedB[kIdx * TILE_N + localColIdx] = (globalCol < N) ? B[kIdx * N + globalCol] : 0.0f;
    }
    __syncthreads();
    
    if (row < M && col < N) {
        float sum = 0.0f;
        
        // Unroll by 4
        int k = 0;
        for (; k + 3 < K; k += 4) {
            sum += A[row * K + k]     * sharedB[(k)     * TILE_N + localCol];
            sum += A[row * K + k + 1] * sharedB[(k + 1) * TILE_N + localCol];
            sum += A[row * K + k + 2] * sharedB[(k + 2) * TILE_N + localCol];
            sum += A[row * K + k + 3] * sharedB[(k + 3) * TILE_N + localCol];
        }
        for (; k < K; k++) {
            sum += A[row * K + k] * sharedB[k * TILE_N + localCol];
        }
        C[row * N + col] = sum;
    }
}

void initMatrix(float* mat, int size) {
    for (int i = 0; i < size; i++) {
        mat[i] = (rand() / (float)RAND_MAX - 0.5f);
    }
}

int main() {
    printf("=== Day 5: Column-Tiled GEMM ===\n\n");
    
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;
    
    printf("Matrix dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    printf("Operations: %.2f GFLOP\n\n", 2.0 * M * N * K / 1e9);
    
    // Allocate
    float *hA = new float[M * K];
    float *hB = new float[K * N];
    float *hC = new float[M * N];
    float *hRef = new float[M * N];
    
    srand(42);
    initMatrix(hA, M * K);
    initMatrix(hB, K * N);
    
    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC, M * N * sizeof(float)));
    
    CHECK_CUDA(cudaMemcpy(dA, hA, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, K * N * sizeof(float), cudaMemcpyHostToDevice));
    
    // cuBLAS reference
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float cublasMs;
    cudaEventElapsedTime(&cublasMs, start, stop);
    
    CHECK_CUDA(cudaMemcpy(hRef, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    float cublasTflops = 2.0f * M * N * K / (cublasMs * 1e9);
    printf("cuBLAS:           %7.3f ms  (%6.2f TFLOPS) [reference]\n", cublasMs, cublasTflops);
    
    // Naive
    dim3 blockNaive(16, 16);
    dim3 gridNaive((N + 15) / 16, (M + 15) / 16);
    
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        naiveGemm<<<gridNaive, blockNaive>>>(dA, dB, dC, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float naiveMs;
    cudaEventElapsedTime(&naiveMs, start, stop);
    naiveMs /= 10;
    
    CHECK_CUDA(cudaMemcpy(hC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    float maxErr = 0;
    for (int i = 0; i < M * N; i++) maxErr = fmaxf(maxErr, fabsf(hC[i] - hRef[i]));
    float naiveTflops = 2.0f * M * N * K / (naiveMs * 1e9);
    printf("Naive GEMM:       %7.3f ms  (%6.2f TFLOPS)  error: %.2e\n", naiveMs, naiveTflops, maxErr);
    
    // Column-Tiled
    constexpr int TILE_N = 16;
    size_t sharedSize = K * TILE_N * sizeof(float);
    dim3 blockCol(TILE_N, 16);  // TILE_N threads per column tile, 16 rows
    dim3 gridCol((N + TILE_N - 1) / TILE_N, (M + 15) / 16);
    
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        colTiledGemm<TILE_N, 16><<<gridCol, blockCol, sharedSize>>>(dA, dB, dC, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float colTiledMs;
    cudaEventElapsedTime(&colTiledMs, start, stop);
    colTiledMs /= 10;
    
    CHECK_CUDA(cudaMemcpy(hC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    maxErr = 0;
    for (int i = 0; i < M * N; i++) maxErr = fmaxf(maxErr, fabsf(hC[i] - hRef[i]));
    float colTiledTflops = 2.0f * M * N * K / (colTiledMs * 1e9);
    printf("Col-Tiled:        %7.3f ms  (%6.2f TFLOPS)  error: %.2e\n", colTiledMs, colTiledTflops, maxErr);
    
    // Column-Tiled Unrolled
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        colTiledGemmUnroll<TILE_N><<<gridCol, blockCol, sharedSize>>>(dA, dB, dC, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float colUnrollMs;
    cudaEventElapsedTime(&colUnrollMs, start, stop);
    colUnrollMs /= 10;
    
    CHECK_CUDA(cudaMemcpy(hC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    maxErr = 0;
    for (int i = 0; i < M * N; i++) maxErr = fmaxf(maxErr, fabsf(hC[i] - hRef[i]));
    float colUnrollTflops = 2.0f * M * N * K / (colUnrollMs * 1e9);
    printf("Col-Tiled+Unroll: %7.3f ms  (%6.2f TFLOPS)  error: %.2e\n", colUnrollMs, colUnrollTflops, maxErr);
    
    // Analysis
    printf("\n=== Analysis ===\n");
    printf("Shared memory per block: %.1f KB\n", sharedSize / 1024.0f);
    printf("\nSpeedup over naive:\n");
    printf("  Col-Tiled:        %.2fx\n", naiveMs / colTiledMs);
    printf("  Col-Tiled+Unroll: %.2fx\n", naiveMs / colUnrollMs);
    printf("\n%% of cuBLAS:\n");
    printf("  Naive:            %.1f%%\n", 100.0f * cublasMs / naiveMs);
    printf("  Col-Tiled:        %.1f%%\n", 100.0f * cublasMs / colTiledMs);
    printf("  Col-Tiled+Unroll: %.1f%%\n", 100.0f * cublasMs / colUnrollMs);
    
    printf("\n=== Key Insight ===\n");
    printf("Column tiling helps B access but A is still read M*K times!\n");
    printf("Row-tiling and column-tiling are partial solutions.\n");
    printf("True optimization requires tiling BOTH dimensions → 2D tiles.\n");
    
    // Cleanup
    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    delete[] hA;
    delete[] hB;
    delete[] hC;
    delete[] hRef;
    
    return 0;
}
