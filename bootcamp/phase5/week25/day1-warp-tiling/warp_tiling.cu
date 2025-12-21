/**
 * Week 25, Day 1: Warp Tiling Introduction
 * 
 * Understanding warp-level GEMM organization.
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

// Constants for warp-tiled GEMM
constexpr int WARP_SIZE = 32;
constexpr int BM = 128;   // Block tile M
constexpr int BN = 128;   // Block tile N
constexpr int BK = 16;    // Block tile K
constexpr int WM = 64;    // Warp tile M
constexpr int WN = 64;    // Warp tile N
constexpr int TM = 8;     // Thread tile M
constexpr int TN = 8;     // Thread tile N

// Warps per block: (BM/WM) × (BN/WN) = 2 × 2 = 4 warps
// Threads per warp tile: (WM/TM) × (WN/TN) = 8 × 8 = 64 threads? 
// But warp has 32 threads, so we use (WM/TM) × (WN/TN) / 2 layout

__global__ void warpTiledGemm(const float* __restrict__ A, 
                               const float* __restrict__ B, 
                               float* __restrict__ C, 
                               int M, int N, int K) {
    __shared__ float As[BK][BM + 1];
    __shared__ float Bs[BK][BN + 1];
    
    // Warp and lane indices
    const int warpId = threadIdx.x / WARP_SIZE;
    const int laneId = threadIdx.x % WARP_SIZE;
    
    // Number of warps: 256 threads / 32 = 8 warps
    // Warp layout: 2 × 4 (along M × N)
    const int warpRow = warpId / 4;  // 0-1
    const int warpCol = warpId % 4;  // 0-3
    
    // Each warp computes 64×32 output
    // Thread layout within warp: 4×8 (4 along M, 8 along N)
    // Each thread: 16×4 outputs
    const int threadRowInWarp = laneId / 8;  // 0-3
    const int threadColInWarp = laneId % 8;  // 0-7
    
    // Thread tile: 16 rows × 4 cols
    constexpr int threadTileM = 16;
    constexpr int threadTileN = 4;
    
    // Global position
    const int blockRowStart = blockIdx.y * BM;
    const int blockColStart = blockIdx.x * BN;
    
    // Warp position within block
    const int warpRowStart = warpRow * 64;    // 0 or 64
    const int warpColStart = warpCol * 32;    // 0, 32, 64, or 96
    
    // Thread position within warp tile
    const int threadRowStart = blockRowStart + warpRowStart + threadRowInWarp * threadTileM;
    const int threadColStart = blockColStart + warpColStart + threadColInWarp * threadTileN;
    
    // Accumulators
    float acc[threadTileM][threadTileN] = {{0}};
    
    // Thread block has 256 threads
    const int tid = threadIdx.x;
    const int numThreads = blockDim.x;
    
    for (int tileK = 0; tileK < K; tileK += BK) {
        // Collaborative load
        for (int i = tid; i < BK * BM; i += numThreads) {
            int loadK = i / BM, loadM = i % BM;
            int globalK = tileK + loadK, globalM = blockRowStart + loadM;
            As[loadK][loadM] = (globalK < K && globalM < M) ? A[globalM * K + globalK] : 0.0f;
        }
        for (int i = tid; i < BK * BN; i += numThreads) {
            int loadK = i / BN, loadN = i % BN;
            int globalK = tileK + loadK, globalN = blockColStart + loadN;
            Bs[loadK][loadN] = (globalK < K && globalN < N) ? B[globalK * N + globalN] : 0.0f;
        }
        __syncthreads();
        
        // Compute
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            // Load A fragment for this thread
            float regA[threadTileM];
            #pragma unroll
            for (int i = 0; i < threadTileM; i++) {
                int m = warpRowStart + threadRowInWarp * threadTileM + i;
                regA[i] = As[k][m];
            }
            
            // Load B fragment for this thread
            float regB[threadTileN];
            #pragma unroll
            for (int j = 0; j < threadTileN; j++) {
                int n = warpColStart + threadColInWarp * threadTileN + j;
                regB[j] = Bs[k][n];
            }
            
            // Outer product
            #pragma unroll
            for (int i = 0; i < threadTileM; i++)
                #pragma unroll
                for (int j = 0; j < threadTileN; j++)
                    acc[i][j] += regA[i] * regB[j];
        }
        __syncthreads();
    }
    
    // Store results
    #pragma unroll
    for (int i = 0; i < threadTileM; i++)
        #pragma unroll
        for (int j = 0; j < threadTileN; j++) {
            int row = threadRowStart + i;
            int col = threadColStart + j;
            if (row < M && col < N)
                C[row * N + col] = acc[i][j];
        }
}

void initMatrix(float* mat, int size) {
    for (int i = 0; i < size; i++) mat[i] = (rand() / (float)RAND_MAX - 0.5f);
}

int main() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║         Week 25, Day 1: Warp Tiling Introduction             ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    printf("Warp Tile Design:\n");
    printf("  Block tile: %d × %d\n", BM, BN);
    printf("  Warps per block: %d\n", (BM/64) * (BN/32));
    printf("  Thread tile: 16 × 4 (64 outputs per thread)\n\n");
    
    const int M = 2048, N = 2048, K = 2048;
    double gflops = 2.0 * M * N * K / 1e9;
    printf("Matrix: %d × %d × %d (%.2f GFLOP)\n\n", M, N, K, gflops);
    
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
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;
    
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);
    cudaDeviceSynchronize();
    
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
    
    printf("Performance:\n");
    printf("┌─────────────────────────┬──────────┬──────────┬──────────┐\n");
    printf("│ Implementation          │ Time(ms) │ TFLOPS   │ %%cuBLAS  │\n");
    printf("├─────────────────────────┼──────────┼──────────┼──────────┤\n");
    printf("│ cuBLAS                  │ %8.3f │ %8.2f │   100.0%% │\n", 
           cublasMs, gflops / cublasMs);
    
    // Warp-tiled GEMM
    dim3 block(256);  // 8 warps
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    
    warpTiledGemm<<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        warpTiledGemm<<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float warpMs;
    cudaEventElapsedTime(&warpMs, start, stop);
    warpMs /= iterations;
    
    CHECK_CUDA(cudaMemcpy(hC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    float maxErr = 0;
    for (int i = 0; i < M * N; i++) maxErr = fmaxf(maxErr, fabsf(hC[i] - hRef[i]));
    
    printf("│ Warp-Tiled GEMM         │ %8.3f │ %8.2f │ %7.1f%% │\n",
           warpMs, gflops / warpMs, 100.0f * cublasMs / warpMs);
    printf("└─────────────────────────┴──────────┴──────────┴──────────┘\n\n");
    
    printf("Max error: %.2e\n\n", maxErr);
    
    printf("Warp Tiling Benefits:\n");
    printf("  - No sync needed within warp\n");
    printf("  - Foundation for shuffle operations\n");
    printf("  - Better register utilization potential\n");
    
    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    delete[] hA; delete[] hB; delete[] hC; delete[] hRef;
    
    return 0;
}
