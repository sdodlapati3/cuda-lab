/**
 * Day 2: Basic Tiled GEMM
 * 
 * Implementation of 2D tiled matrix multiplication using shared memory.
 * This is the foundational high-performance GEMM kernel.
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

// Naive GEMM for comparison
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

// Basic 2D Tiled GEMM
template<int TILE_SIZE>
__global__ void tiledGemm(const float* __restrict__ A,
                          const float* __restrict__ B,
                          float* __restrict__ C,
                          int M, int N, int K) {
    // Shared memory tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    // Global row and column for this thread
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    // Accumulator for dot product
    float sum = 0.0f;
    
    // Number of tiles to iterate over
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; t++) {
        // Column in A for this tile
        int aCol = t * TILE_SIZE + threadIdx.x;
        // Row in B for this tile
        int bRow = t * TILE_SIZE + threadIdx.y;
        
        // Collaborative load of A tile
        if (row < M && aCol < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + aCol];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Collaborative load of B tile
        if (bRow < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[bRow * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Synchronize to ensure tile is fully loaded
        __syncthreads();
        
        // Compute partial dot product from shared memory
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        // Synchronize before loading next tile
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Alternative: Tiled GEMM with explicit bounds in inner loop
template<int TILE_SIZE>
__global__ void tiledGemmSafe(const float* __restrict__ A,
                               const float* __restrict__ B,
                               float* __restrict__ C,
                               int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int tileK = 0; tileK < K; tileK += TILE_SIZE) {
        // Load with bounds checking
        int aCol = tileK + threadIdx.x;
        int bRow = tileK + threadIdx.y;
        
        As[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        
        __syncthreads();
        
        // Compute with bounds on k
        int kEnd = min(TILE_SIZE, K - tileK);
        for (int k = 0; k < kEnd; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void initMatrix(float* mat, int size) {
    for (int i = 0; i < size; i++) {
        mat[i] = (rand() / (float)RAND_MAX - 0.5f) * 2.0f;
    }
}

int main() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║               Day 2: Basic Tiled GEMM                        ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    const int M = 2048;
    const int N = 2048;
    const int K = 2048;
    
    printf("Matrix dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    double gflops = 2.0 * M * N * K / 1e9;
    printf("Operations: %.2f GFLOP\n\n", gflops);
    
    // Allocate host memory
    float *hA = new float[M * K];
    float *hB = new float[K * N];
    float *hC = new float[M * N];
    float *hRef = new float[M * N];
    
    srand(42);
    initMatrix(hA, M * K);
    initMatrix(hB, K * N);
    
    // Allocate device memory
    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC, M * N * sizeof(float)));
    
    CHECK_CUDA(cudaMemcpy(dA, hA, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, K * N * sizeof(float), cudaMemcpyHostToDevice));
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    int iterations = 20;
    
    // cuBLAS reference
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;
    
    // Warmup
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
    float cublasTflops = gflops / cublasMs;
    printf("cuBLAS:         %7.3f ms  (%6.2f TFLOPS) [reference]\n", cublasMs, cublasTflops);
    
    // Naive GEMM
    dim3 blockNaive(16, 16);
    dim3 gridNaive((N + 15) / 16, (M + 15) / 16);
    
    naiveGemm<<<gridNaive, blockNaive>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        naiveGemm<<<gridNaive, blockNaive>>>(dA, dB, dC, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float naiveMs;
    cudaEventElapsedTime(&naiveMs, start, stop);
    naiveMs /= iterations;
    
    CHECK_CUDA(cudaMemcpy(hC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    float maxErr = 0;
    for (int i = 0; i < M * N; i++) {
        maxErr = fmaxf(maxErr, fabsf(hC[i] - hRef[i]));
    }
    float naiveTflops = gflops / naiveMs;
    printf("Naive:          %7.3f ms  (%6.2f TFLOPS)  error: %.2e\n", naiveMs, naiveTflops, maxErr);
    
    // Tiled GEMM with different tile sizes
    printf("\n--- Tiled GEMM (varying tile sizes) ---\n");
    
    // Tile 16x16
    {
        constexpr int TILE = 16;
        dim3 block(TILE, TILE);
        dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
        
        tiledGemm<TILE><<<grid, block>>>(dA, dB, dC, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());
        
        cudaEventRecord(start);
        for (int i = 0; i < iterations; i++) {
            tiledGemm<TILE><<<grid, block>>>(dA, dB, dC, M, N, K);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        ms /= iterations;
        
        CHECK_CUDA(cudaMemcpy(hC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));
        maxErr = 0;
        for (int i = 0; i < M * N; i++) maxErr = fmaxf(maxErr, fabsf(hC[i] - hRef[i]));
        
        printf("Tiled 16×16:    %7.3f ms  (%6.2f TFLOPS)  error: %.2e  speedup: %.1fx\n",
               ms, gflops / ms, maxErr, naiveMs / ms);
    }
    
    // Tile 32x32
    {
        constexpr int TILE = 32;
        dim3 block(TILE, TILE);
        dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
        
        tiledGemm<TILE><<<grid, block>>>(dA, dB, dC, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());
        
        cudaEventRecord(start);
        for (int i = 0; i < iterations; i++) {
            tiledGemm<TILE><<<grid, block>>>(dA, dB, dC, M, N, K);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        ms /= iterations;
        
        CHECK_CUDA(cudaMemcpy(hC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));
        maxErr = 0;
        for (int i = 0; i < M * N; i++) maxErr = fmaxf(maxErr, fabsf(hC[i] - hRef[i]));
        
        printf("Tiled 32×32:    %7.3f ms  (%6.2f TFLOPS)  error: %.2e  speedup: %.1fx\n",
               ms, gflops / ms, maxErr, naiveMs / ms);
    }
    
    // Summary
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║                        Summary                               ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║ 2D tiling provides significant speedup over naive:           ║\n");
    printf("║ - Shared memory reduces global memory traffic                ║\n");
    printf("║ - Data reuse factor ≈ tile size                              ║\n");
    printf("║ - 32×32 tiles often optimal for SM occupancy                 ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    
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
