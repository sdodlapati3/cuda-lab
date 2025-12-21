/**
 * Day 6: Week 22 Tiling Performance Summary
 * 
 * Comprehensive benchmark of all tiling optimizations.
 */

#include <cuda_runtime.h>
#include <cuda_pipeline.h>
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

// Naive GEMM (Week 21)
__global__ void naiveGemm(const float* A, const float* B, float* C, int M, int N, int K) {
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

// Basic tiled (Day 2)
template<int TILE>
__global__ void tiledBasic(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int t = 0; t < K; t += TILE) {
        As[threadIdx.y][threadIdx.x] = (row < M && t + threadIdx.x < K) ? 
            A[row * K + t + threadIdx.x] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (t + threadIdx.y < K && col < N) ? 
            B[(t + threadIdx.y) * N + col] : 0.0f;
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    
    if (row < M && col < N) C[row * N + col] = sum;
}

// Tiled with padding (Day 4)
template<int TILE>
__global__ void tiledPadded(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE][TILE + 1];
    __shared__ float Bs[TILE][TILE + 1];
    
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int t = 0; t < K; t += TILE) {
        As[threadIdx.y][threadIdx.x] = (row < M && t + threadIdx.x < K) ? 
            A[row * K + t + threadIdx.x] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (t + threadIdx.y < K && col < N) ? 
            B[(t + threadIdx.y) * N + col] : 0.0f;
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    
    if (row < M && col < N) C[row * N + col] = sum;
}

// Double buffered + padded (Day 5)
template<int TILE>
__global__ void tiledDouble(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[2][TILE][TILE + 1];
    __shared__ float Bs[2][TILE][TILE + 1];
    
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    
    float sum = 0.0f;
    int numTiles = (K + TILE - 1) / TILE;
    
    // Load first tile
    As[0][threadIdx.y][threadIdx.x] = (row < M && threadIdx.x < K) ? 
        A[row * K + threadIdx.x] : 0.0f;
    Bs[0][threadIdx.y][threadIdx.x] = (threadIdx.y < K && col < N) ? 
        B[threadIdx.y * N + col] : 0.0f;
    __syncthreads();
    
    for (int t = 1; t < numTiles; t++) {
        int curr = (t - 1) % 2;
        int next = t % 2;
        int tK = t * TILE;
        
        As[next][threadIdx.y][threadIdx.x] = (row < M && tK + threadIdx.x < K) ? 
            A[row * K + tK + threadIdx.x] : 0.0f;
        Bs[next][threadIdx.y][threadIdx.x] = (tK + threadIdx.y < K && col < N) ? 
            B[(tK + threadIdx.y) * N + col] : 0.0f;
        
        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            sum += As[curr][threadIdx.y][k] * Bs[curr][k][threadIdx.x];
        }
        __syncthreads();
    }
    
    int last = (numTiles - 1) % 2;
    #pragma unroll
    for (int k = 0; k < TILE; k++) {
        sum += As[last][threadIdx.y][k] * Bs[last][k][threadIdx.x];
    }
    
    if (row < M && col < N) C[row * N + col] = sum;
}

void initMatrix(float* mat, int size) {
    for (int i = 0; i < size; i++) {
        mat[i] = (rand() / (float)RAND_MAX - 0.5f);
    }
}

int main() {
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║           Week 22: Tiling Optimizations Summary                  ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n\n", prop.name);
    
    const int M = 4096;
    const int N = 4096;
    const int K = 4096;
    constexpr int TILE = 32;
    
    printf("Matrix: %d × %d × %d\n", M, N, K);
    double gflops = 2.0 * M * N * K / 1e9;
    printf("Operations: %.2f GFLOP\n\n", gflops);
    
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
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int iterations = 10;
    
    // cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;
    
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float cublasMs;
    cudaEventElapsedTime(&cublasMs, start, stop);
    cublasMs /= iterations;
    
    CHECK_CUDA(cudaMemcpy(hRef, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("┌───────────────────────────┬──────────┬──────────┬──────────┬───────────┐\n");
    printf("│ Implementation            │ Time(ms) │ TFLOPS   │ %%cuBLAS  │ Speedup   │\n");
    printf("├───────────────────────────┼──────────┼──────────┼──────────┼───────────┤\n");
    
    float cublasTflops = gflops / cublasMs;
    printf("│ cuBLAS (reference)        │ %8.3f │ %8.2f │   100.0%% │     --    │\n",
           cublasMs, cublasTflops);
    
    // Naive
    dim3 blockN(16, 16);
    dim3 gridN((N + 15) / 16, (M + 15) / 16);
    
    naiveGemm<<<gridN, blockN>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        naiveGemm<<<gridN, blockN>>>(dA, dB, dC, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float naiveMs;
    cudaEventElapsedTime(&naiveMs, start, stop);
    naiveMs /= iterations;
    
    printf("│ Naive (Week 21)           │ %8.3f │ %8.2f │ %7.1f%% │   1.00x   │\n",
           naiveMs, gflops / naiveMs, 100.0f * cublasMs / naiveMs);
    
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    
    // Basic tiled
    tiledBasic<TILE><<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        tiledBasic<TILE><<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float basicMs;
    cudaEventElapsedTime(&basicMs, start, stop);
    basicMs /= iterations;
    
    printf("│ 2D Tiled (Day 2)          │ %8.3f │ %8.2f │ %7.1f%% │ %6.2fx   │\n",
           basicMs, gflops / basicMs, 100.0f * cublasMs / basicMs, naiveMs / basicMs);
    
    // Padded
    tiledPadded<TILE><<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        tiledPadded<TILE><<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float paddedMs;
    cudaEventElapsedTime(&paddedMs, start, stop);
    paddedMs /= iterations;
    
    printf("│ + Padding (Day 4)         │ %8.3f │ %8.2f │ %7.1f%% │ %6.2fx   │\n",
           paddedMs, gflops / paddedMs, 100.0f * cublasMs / paddedMs, naiveMs / paddedMs);
    
    // Double buffered
    tiledDouble<TILE><<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        tiledDouble<TILE><<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float doubleMs;
    cudaEventElapsedTime(&doubleMs, start, stop);
    doubleMs /= iterations;
    
    CHECK_CUDA(cudaMemcpy(hC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    float maxErr = 0;
    for (int i = 0; i < M * N; i++) maxErr = fmaxf(maxErr, fabsf(hC[i] - hRef[i]));
    
    printf("│ + Double Buffer (Day 5)   │ %8.3f │ %8.2f │ %7.1f%% │ %6.2fx   │\n",
           doubleMs, gflops / doubleMs, 100.0f * cublasMs / doubleMs, naiveMs / doubleMs);
    
    printf("└───────────────────────────┴──────────┴──────────┴──────────┴───────────┘\n\n");
    
    printf("Max error: %.2e\n\n", maxErr);
    
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║                    Week 22 Summary                               ║\n");
    printf("╠══════════════════════════════════════════════════════════════════╣\n");
    printf("║ Total speedup over naive: %.1fx                                  ║\n", naiveMs / doubleMs);
    printf("║ Current %% of cuBLAS: %.1f%%                                       ║\n", 100.0f * cublasMs / doubleMs);
    printf("╠══════════════════════════════════════════════════════════════════╣\n");
    printf("║ Week 23 Focus: Register blocking, vectorized access              ║\n");
    printf("║ Target: 50-60%% of cuBLAS performance                             ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n");
    
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
