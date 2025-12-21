/**
 * Week 23, Day 3: 4×4 Thread Tile
 * 
 * Each thread computes a 4×4 tile of outputs (16 elements).
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

// 2×2 Thread Tile (reference)
template<int BM, int BN, int BK>
__global__ void threadTile2x2Gemm(const float* A, const float* B, float* C, 
                                   int M, int N, int K) {
    __shared__ float As[BK][BM + 1];
    __shared__ float Bs[BK][BN + 1];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int rowStart = blockIdx.y * BM + ty * 2;
    const int colStart = blockIdx.x * BN + tx * 2;
    
    float c00 = 0.0f, c01 = 0.0f, c10 = 0.0f, c11 = 0.0f;
    
    const int numThreads = (BM / 2) * (BN / 2);
    const int tid = ty * (BN / 2) + tx;
    
    for (int tileK = 0; tileK < K; tileK += BK) {
        for (int i = tid; i < BK * BM; i += numThreads) {
            int loadK = i / BM, loadM = i % BM;
            int globalK = tileK + loadK, globalM = blockIdx.y * BM + loadM;
            As[loadK][loadM] = (globalK < K && globalM < M) ? A[globalM * K + globalK] : 0.0f;
        }
        for (int i = tid; i < BK * BN; i += numThreads) {
            int loadK = i / BN, loadN = i % BN;
            int globalK = tileK + loadK, globalN = blockIdx.x * BN + loadN;
            Bs[loadK][loadN] = (globalK < K && globalN < N) ? B[globalK * N + globalN] : 0.0f;
        }
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            float a0 = As[k][ty * 2], a1 = As[k][ty * 2 + 1];
            float b0 = Bs[k][tx * 2], b1 = Bs[k][tx * 2 + 1];
            c00 += a0 * b0; c01 += a0 * b1; c10 += a1 * b0; c11 += a1 * b1;
        }
        __syncthreads();
    }
    
    if (rowStart < M && colStart < N) C[rowStart * N + colStart] = c00;
    if (rowStart < M && colStart + 1 < N) C[rowStart * N + colStart + 1] = c01;
    if (rowStart + 1 < M && colStart < N) C[(rowStart + 1) * N + colStart] = c10;
    if (rowStart + 1 < M && colStart + 1 < N) C[(rowStart + 1) * N + colStart + 1] = c11;
}

// 4×4 Thread Tile GEMM
template<int BM, int BN, int BK>
__global__ void threadTile4x4Gemm(const float* A, const float* B, float* C, 
                                   int M, int N, int K) {
    __shared__ float As[BK][BM + 1];
    __shared__ float Bs[BK][BN + 1];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // Each thread computes 4×4 outputs
    const int TM = 4, TN = 4;
    const int rowStart = blockIdx.y * BM + ty * TM;
    const int colStart = blockIdx.x * BN + tx * TN;
    
    // 16 accumulators in registers
    float acc[TM][TN] = {{0}};
    
    // Thread block size
    const int threadsX = BN / TN;
    const int threadsY = BM / TM;
    const int numThreads = threadsX * threadsY;
    const int tid = ty * threadsX + tx;
    
    for (int tileK = 0; tileK < K; tileK += BK) {
        // Load A tile
        for (int i = tid; i < BK * BM; i += numThreads) {
            int loadK = i / BM, loadM = i % BM;
            int globalK = tileK + loadK, globalM = blockIdx.y * BM + loadM;
            As[loadK][loadM] = (globalK < K && globalM < M) ? A[globalM * K + globalK] : 0.0f;
        }
        // Load B tile
        for (int i = tid; i < BK * BN; i += numThreads) {
            int loadK = i / BN, loadN = i % BN;
            int globalK = tileK + loadK, globalN = blockIdx.x * BN + loadN;
            Bs[loadK][loadN] = (globalK < K && globalN < N) ? B[globalK * N + globalN] : 0.0f;
        }
        __syncthreads();
        
        // Compute outer products
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            // Load A fragment (4 elements)
            float regA[TM];
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                regA[i] = As[k][ty * TM + i];
            }
            
            // Load B fragment (4 elements)
            float regB[TN];
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                regB[j] = Bs[k][tx * TN + j];
            }
            
            // Outer product: 16 FMAs
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                    acc[i][j] += regA[i] * regB[j];
                }
            }
        }
        __syncthreads();
    }
    
    // Store 4×4 output
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int row = rowStart + i;
            int col = colStart + j;
            if (row < M && col < N) {
                C[row * N + col] = acc[i][j];
            }
        }
    }
}

void initMatrix(float* mat, int size) {
    for (int i = 0; i < size; i++) mat[i] = (rand() / (float)RAND_MAX - 0.5f);
}

int main() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║           Week 23, Day 3: 4×4 Thread Tile                    ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
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
    
    // 2×2 Thread Tile
    constexpr int BM2 = 64, BN2 = 64, BK2 = 16;
    dim3 block2x2(BN2 / 2, BM2 / 2);
    dim3 grid2x2((N + BN2 - 1) / BN2, (M + BM2 - 1) / BM2);
    
    threadTile2x2Gemm<BM2, BN2, BK2><<<grid2x2, block2x2>>>(dA, dB, dC, M, N, K);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        threadTile2x2Gemm<BM2, BN2, BK2><<<grid2x2, block2x2>>>(dA, dB, dC, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float tile2x2Ms;
    cudaEventElapsedTime(&tile2x2Ms, start, stop);
    tile2x2Ms /= iterations;
    
    printf("│ 2×2 Thread Tile         │ %8.3f │ %8.2f │ %7.1f%% │\n",
           tile2x2Ms, gflops / tile2x2Ms, 100.0f * cublasMs / tile2x2Ms);
    
    // 4×4 Thread Tile
    constexpr int BM4 = 64, BN4 = 64, BK4 = 16;
    dim3 block4x4(BN4 / 4, BM4 / 4);  // 16 × 16 = 256 threads
    dim3 grid4x4((N + BN4 - 1) / BN4, (M + BM4 - 1) / BM4);
    
    threadTile4x4Gemm<BM4, BN4, BK4><<<grid4x4, block4x4>>>(dA, dB, dC, M, N, K);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        threadTile4x4Gemm<BM4, BN4, BK4><<<grid4x4, block4x4>>>(dA, dB, dC, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float tile4x4Ms;
    cudaEventElapsedTime(&tile4x4Ms, start, stop);
    tile4x4Ms /= iterations;
    
    CHECK_CUDA(cudaMemcpy(hC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    float maxErr = 0;
    for (int i = 0; i < M * N; i++) maxErr = fmaxf(maxErr, fabsf(hC[i] - hRef[i]));
    
    printf("│ 4×4 Thread Tile         │ %8.3f │ %8.2f │ %7.1f%% │\n",
           tile4x4Ms, gflops / tile4x4Ms, 100.0f * cublasMs / tile4x4Ms);
    printf("└─────────────────────────┴──────────┴──────────┴──────────┘\n\n");
    
    printf("Speedup: 4×4 is %.2fx faster than 2×2\n", tile2x2Ms / tile4x4Ms);
    printf("Max error: %.2e\n\n", maxErr);
    
    printf("Compute Intensity Comparison:\n");
    printf("  2×2: 4 FMA / (2+2) loads = 1.0\n");
    printf("  4×4: 16 FMA / (4+4) loads = 2.0\n");
    printf("  Improvement: 2× compute intensity\n");
    
    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    delete[] hA; delete[] hB; delete[] hC; delete[] hRef;
    
    return 0;
}
