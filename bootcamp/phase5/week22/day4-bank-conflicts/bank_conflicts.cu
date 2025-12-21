/**
 * Day 4: Bank Conflicts in Shared Memory
 * 
 * Demonstrates how bank conflicts affect GEMM performance
 * and how padding eliminates them.
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

// Tiled GEMM WITHOUT padding (potential bank conflicts)
template<int TILE>
__global__ void tiledGemmNoPad(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int M, int N, int K) {
    __shared__ float As[TILE][TILE];      // No padding
    __shared__ float Bs[TILE][TILE];
    
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int t = 0; t < K; t += TILE) {
        int aCol = t + threadIdx.x;
        int bRow = t + threadIdx.y;
        
        As[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Tiled GEMM WITH padding (no bank conflicts)
template<int TILE>
__global__ void tiledGemmPadded(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C,
                                 int M, int N, int K) {
    __shared__ float As[TILE][TILE + 1];  // +1 padding
    __shared__ float Bs[TILE][TILE + 1];
    
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int t = 0; t < K; t += TILE) {
        int aCol = t + threadIdx.x;
        int bRow = t + threadIdx.y;
        
        As[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Tiled GEMM with transposed B tile (different access pattern)
template<int TILE>
__global__ void tiledGemmTransB(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C,
                                 int M, int N, int K) {
    __shared__ float As[TILE][TILE + 1];
    __shared__ float BsT[TILE][TILE + 1];  // Transposed storage
    
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int t = 0; t < K; t += TILE) {
        int aCol = t + threadIdx.x;
        int bRow = t + threadIdx.y;
        
        As[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        // Store B transposed: BsT[x][y] instead of Bs[y][x]
        BsT[threadIdx.x][threadIdx.y] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            sum += As[threadIdx.y][k] * BsT[threadIdx.x][k];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void initMatrix(float* mat, int size) {
    for (int i = 0; i < size; i++) {
        mat[i] = (rand() / (float)RAND_MAX - 0.5f);
    }
}

int main() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║          Day 4: Bank Conflicts Study                         ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    // Bank explanation
    printf("Shared Memory Banks (32 banks, 4 bytes each):\n");
    printf("┌────────────────────────────────────────────────────────────┐\n");
    printf("│ Address:  0   4   8  12  16  20 ... 124  128 132 ...      │\n");
    printf("│ Bank:     0   1   2   3   4   5 ...  31    0   1  ...     │\n");
    printf("└────────────────────────────────────────────────────────────┘\n\n");
    
    const int M = 2048;
    const int N = 2048;
    const int K = 2048;
    constexpr int TILE = 32;
    
    printf("Matrix: %d × %d × %d, Tile: %d × %d\n\n", M, N, K, TILE, TILE);
    
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
    
    int iterations = 20;
    double gflops = 2.0 * M * N * K / 1e9;
    
    // cuBLAS reference
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
    
    printf("Performance Results:\n");
    printf("┌─────────────────────────┬──────────┬──────────┬──────────┐\n");
    printf("│ Implementation          │ Time(ms) │ TFLOPS   │ %%cuBLAS  │\n");
    printf("├─────────────────────────┼──────────┼──────────┼──────────┤\n");
    printf("│ cuBLAS                  │ %8.3f │ %8.2f │   100.0%% │\n",
           cublasMs, gflops / cublasMs);
    
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    
    // Without padding
    tiledGemmNoPad<TILE><<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        tiledGemmNoPad<TILE><<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float noPadMs;
    cudaEventElapsedTime(&noPadMs, start, stop);
    noPadMs /= iterations;
    
    CHECK_CUDA(cudaMemcpy(hC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    float maxErr = 0;
    for (int i = 0; i < M * N; i++) maxErr = fmaxf(maxErr, fabsf(hC[i] - hRef[i]));
    
    printf("│ No Padding (conflicts)  │ %8.3f │ %8.2f │ %7.1f%% │\n",
           noPadMs, gflops / noPadMs, 100.0f * cublasMs / noPadMs);
    
    // With padding
    tiledGemmPadded<TILE><<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        tiledGemmPadded<TILE><<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float paddedMs;
    cudaEventElapsedTime(&paddedMs, start, stop);
    paddedMs /= iterations;
    
    CHECK_CUDA(cudaMemcpy(hC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    maxErr = 0;
    for (int i = 0; i < M * N; i++) maxErr = fmaxf(maxErr, fabsf(hC[i] - hRef[i]));
    
    printf("│ With Padding            │ %8.3f │ %8.2f │ %7.1f%% │\n",
           paddedMs, gflops / paddedMs, 100.0f * cublasMs / paddedMs);
    
    // Transposed B
    tiledGemmTransB<TILE><<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        tiledGemmTransB<TILE><<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float transBMs;
    cudaEventElapsedTime(&transBMs, start, stop);
    transBMs /= iterations;
    
    printf("│ Padded + Trans B        │ %8.3f │ %8.2f │ %7.1f%% │\n",
           transBMs, gflops / transBMs, 100.0f * cublasMs / transBMs);
    
    printf("└─────────────────────────┴──────────┴──────────┴──────────┘\n\n");
    
    // Analysis
    printf("Analysis:\n");
    printf("  Padding improvement: %.1f%%\n", 100.0f * (noPadMs - paddedMs) / noPadMs);
    printf("  Shared memory overhead: %d bytes/block → %d bytes/block\n",
           2 * TILE * TILE * 4, 2 * TILE * (TILE + 1) * 4);
    
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║                    Key Takeaways                             ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║ 1. Bank conflicts can significantly hurt performance         ║\n");
    printf("║ 2. Adding 1 column of padding eliminates conflicts           ║\n");
    printf("║ 3. Small memory overhead is worth the performance gain       ║\n");
    printf("║ 4. Use Nsight Compute to diagnose bank conflicts             ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    
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
