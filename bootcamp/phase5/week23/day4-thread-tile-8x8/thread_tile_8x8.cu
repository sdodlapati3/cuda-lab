/**
 * Week 23, Day 4: 8×8 Thread Tile
 * 
 * Each thread computes 64 outputs - maximum practical register blocking.
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

// 4×4 Thread Tile
template<int BM, int BN, int BK, int TM, int TN>
__global__ void threadTileGemm(const float* A, const float* B, float* C, 
                               int M, int N, int K) {
    __shared__ float As[BK][BM + 1];
    __shared__ float Bs[BK][BN + 1];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int rowStart = blockIdx.y * BM + ty * TM;
    const int colStart = blockIdx.x * BN + tx * TN;
    
    float acc[TM][TN] = {{0}};
    
    const int threadsX = BN / TN;
    const int threadsY = BM / TM;
    const int numThreads = threadsX * threadsY;
    const int tid = ty * threadsX + tx;
    
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
            float regA[TM], regB[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++) regA[i] = As[k][ty * TM + i];
            #pragma unroll
            for (int j = 0; j < TN; j++) regB[j] = Bs[k][tx * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] += regA[i] * regB[j];
        }
        __syncthreads();
    }
    
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
            if (rowStart + i < M && colStart + j < N)
                C[(rowStart + i) * N + colStart + j] = acc[i][j];
}

// 8×8 Thread Tile (high register pressure)
template<int BM, int BN, int BK>
__global__ void threadTile8x8Gemm(const float* A, const float* B, float* C, 
                                   int M, int N, int K) {
    __shared__ float As[BK][BM + 1];
    __shared__ float Bs[BK][BN + 1];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    constexpr int TM = 8, TN = 8;
    
    const int rowStart = blockIdx.y * BM + ty * TM;
    const int colStart = blockIdx.x * BN + tx * TN;
    
    // 64 accumulators in registers!
    float acc[8][8];
    #pragma unroll
    for (int i = 0; i < 8; i++)
        #pragma unroll
        for (int j = 0; j < 8; j++)
            acc[i][j] = 0.0f;
    
    const int threadsX = BN / TN;
    const int threadsY = BM / TM;
    const int numThreads = threadsX * threadsY;
    const int tid = ty * threadsX + tx;
    
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
            // 8 registers for A fragment
            float regA[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) regA[i] = As[k][ty * 8 + i];
            
            // 8 registers for B fragment
            float regB[8];
            #pragma unroll
            for (int j = 0; j < 8; j++) regB[j] = Bs[k][tx * 8 + j];
            
            // 64 FMAs in outer product
            #pragma unroll
            for (int i = 0; i < 8; i++)
                #pragma unroll
                for (int j = 0; j < 8; j++)
                    acc[i][j] += regA[i] * regB[j];
        }
        __syncthreads();
    }
    
    // Store 64 outputs
    #pragma unroll
    for (int i = 0; i < 8; i++)
        #pragma unroll
        for (int j = 0; j < 8; j++)
            if (rowStart + i < M && colStart + j < N)
                C[(rowStart + i) * N + colStart + j] = acc[i][j];
}

void initMatrix(float* mat, int size) {
    for (int i = 0; i < size; i++) mat[i] = (rand() / (float)RAND_MAX - 0.5f);
}

int main() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║           Week 23, Day 4: 8×8 Thread Tile                    ║\n");
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
    
    printf("Performance Comparison:\n");
    printf("┌─────────────────────────┬──────────┬──────────┬──────────┬───────────┐\n");
    printf("│ Implementation          │ Time(ms) │ TFLOPS   │ %%cuBLAS  │ Intensity │\n");
    printf("├─────────────────────────┼──────────┼──────────┼──────────┼───────────┤\n");
    printf("│ cuBLAS                  │ %8.3f │ %8.2f │   100.0%% │     -     │\n", 
           cublasMs, gflops / cublasMs);
    
    // 4×4 Thread Tile
    constexpr int BM4 = 64, BN4 = 64, BK4 = 16;
    dim3 block4(BN4 / 4, BM4 / 4);  // 16 × 16 = 256
    dim3 grid4((N + BN4 - 1) / BN4, (M + BM4 - 1) / BM4);
    
    threadTileGemm<BM4, BN4, BK4, 4, 4><<<grid4, block4>>>(dA, dB, dC, M, N, K);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        threadTileGemm<BM4, BN4, BK4, 4, 4><<<grid4, block4>>>(dA, dB, dC, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float tile4Ms;
    cudaEventElapsedTime(&tile4Ms, start, stop);
    tile4Ms /= iterations;
    
    printf("│ 4×4 Thread Tile         │ %8.3f │ %8.2f │ %7.1f%% │    2.0    │\n",
           tile4Ms, gflops / tile4Ms, 100.0f * cublasMs / tile4Ms);
    
    // 8×8 Thread Tile
    constexpr int BM8 = 128, BN8 = 128, BK8 = 8;
    dim3 block8(BN8 / 8, BM8 / 8);  // 16 × 16 = 256
    dim3 grid8((N + BN8 - 1) / BN8, (M + BM8 - 1) / BM8);
    
    threadTile8x8Gemm<BM8, BN8, BK8><<<grid8, block8>>>(dA, dB, dC, M, N, K);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        threadTile8x8Gemm<BM8, BN8, BK8><<<grid8, block8>>>(dA, dB, dC, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float tile8Ms;
    cudaEventElapsedTime(&tile8Ms, start, stop);
    tile8Ms /= iterations;
    
    CHECK_CUDA(cudaMemcpy(hC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    float maxErr = 0;
    for (int i = 0; i < M * N; i++) maxErr = fmaxf(maxErr, fabsf(hC[i] - hRef[i]));
    
    printf("│ 8×8 Thread Tile         │ %8.3f │ %8.2f │ %7.1f%% │    4.0    │\n",
           tile8Ms, gflops / tile8Ms, 100.0f * cublasMs / tile8Ms);
    printf("└─────────────────────────┴──────────┴──────────┴──────────┴───────────┘\n\n");
    
    printf("Register Analysis:\n");
    printf("  4×4: 16 acc + 8 frag = ~24 registers\n");
    printf("  8×8: 64 acc + 16 frag = ~80 registers\n\n");
    
    printf("Speedup: 8×8 is %.2fx vs 4×4\n", tile4Ms / tile8Ms);
    printf("Max error: %.2e\n", maxErr);
    
    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    delete[] hA; delete[] hB; delete[] hC; delete[] hRef;
    
    return 0;
}
