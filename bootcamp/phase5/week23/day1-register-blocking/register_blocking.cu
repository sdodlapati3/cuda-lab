/**
 * Week 23, Day 1: Register Blocking Introduction
 * 
 * Demonstrates the concept of each thread computing multiple outputs
 * using registers to store partial results.
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

// Basic tiled GEMM (Week 22 reference)
template<int TILE>
__global__ void tiledGemm(const float* A, const float* B, float* C, int M, int N, int K) {
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

// Register blocking: 2x2 outputs per thread
template<int BM, int BN, int BK, int TM, int TN>
__global__ void regBlockGemm(const float* A, const float* B, float* C, int M, int N, int K) {
    // Thread block computes BM x BN output tile
    // Each thread computes TM x TN outputs
    
    __shared__ float As[BK][BM + 1];
    __shared__ float Bs[BK][BN + 1];
    
    // Thread position in block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int numThreadsX = BN / TN;  // threads along N
    const int numThreadsY = BM / TM;  // threads along M
    
    // This thread's output tile position
    const int threadRow = ty * TM;
    const int threadCol = tx * TN;
    
    // Global starting position
    const int rowStart = blockIdx.y * BM + threadRow;
    const int colStart = blockIdx.x * BN + threadCol;
    
    // Register accumulators (TM x TN outputs)
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            acc[i][j] = 0.0f;
        }
    }
    
    // Register fragments for A and B
    float regA[TM];
    float regB[TN];
    
    // Loop over K tiles
    for (int tileK = 0; tileK < K; tileK += BK) {
        // Collaborative load of A tile (BK x BM) into As
        // Each thread loads multiple elements
        for (int loadOffset = ty * numThreadsX + tx; 
             loadOffset < BK * BM; 
             loadOffset += numThreadsX * numThreadsY) {
            int loadK = loadOffset / BM;
            int loadM = loadOffset % BM;
            int globalK = tileK + loadK;
            int globalM = blockIdx.y * BM + loadM;
            As[loadK][loadM] = (globalK < K && globalM < M) ? 
                A[globalM * K + globalK] : 0.0f;
        }
        
        // Collaborative load of B tile (BK x BN) into Bs
        for (int loadOffset = ty * numThreadsX + tx; 
             loadOffset < BK * BN; 
             loadOffset += numThreadsX * numThreadsY) {
            int loadK = loadOffset / BN;
            int loadN = loadOffset % BN;
            int globalK = tileK + loadK;
            int globalN = blockIdx.x * BN + loadN;
            Bs[loadK][loadN] = (globalK < K && globalN < N) ? 
                B[globalK * N + globalN] : 0.0f;
        }
        
        __syncthreads();
        
        // Compute using register blocking
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            // Load A fragment into registers
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                regA[i] = As[k][threadRow + i];
            }
            
            // Load B fragment into registers
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                regB[j] = Bs[k][threadCol + j];
            }
            
            // Compute TM x TN outer product
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
    
    // Write results
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int outRow = rowStart + i;
            int outCol = colStart + j;
            if (outRow < M && outCol < N) {
                C[outRow * N + outCol] = acc[i][j];
            }
        }
    }
}

void initMatrix(float* mat, int size) {
    for (int i = 0; i < size; i++) {
        mat[i] = (rand() / (float)RAND_MAX - 0.5f);
    }
}

int main() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║       Week 23, Day 1: Register Blocking Introduction         ║\n");
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
    
    // cuBLAS reference
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
    printf("┌─────────────────────────┬──────────┬──────────┬──────────┐\n");
    printf("│ Implementation          │ Time(ms) │ TFLOPS   │ %%cuBLAS  │\n");
    printf("├─────────────────────────┼──────────┼──────────┼──────────┤\n");
    printf("│ cuBLAS                  │ %8.3f │ %8.2f │   100.0%% │\n", 
           cublasMs, gflops / cublasMs);
    
    // Basic tiled
    constexpr int TILE = 32;
    dim3 blockBasic(TILE, TILE);
    dim3 gridBasic((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    
    tiledGemm<TILE><<<gridBasic, blockBasic>>>(dA, dB, dC, M, N, K);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        tiledGemm<TILE><<<gridBasic, blockBasic>>>(dA, dB, dC, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float basicMs;
    cudaEventElapsedTime(&basicMs, start, stop);
    basicMs /= iterations;
    
    printf("│ Basic Tiled (32×32)     │ %8.3f │ %8.2f │ %7.1f%% │\n",
           basicMs, gflops / basicMs, 100.0f * cublasMs / basicMs);
    
    // Register blocked 2x2
    constexpr int BM = 64, BN = 64, BK = 8, TM = 4, TN = 4;
    dim3 blockReg(BN / TN, BM / TM);  // 16 x 16 = 256 threads
    dim3 gridReg((N + BN - 1) / BN, (M + BM - 1) / BM);
    
    regBlockGemm<BM, BN, BK, TM, TN><<<gridReg, blockReg>>>(dA, dB, dC, M, N, K);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        regBlockGemm<BM, BN, BK, TM, TN><<<gridReg, blockReg>>>(dA, dB, dC, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float regMs;
    cudaEventElapsedTime(&regMs, start, stop);
    regMs /= iterations;
    
    CHECK_CUDA(cudaMemcpy(hC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    float maxErr = 0;
    for (int i = 0; i < M * N; i++) maxErr = fmaxf(maxErr, fabsf(hC[i] - hRef[i]));
    
    printf("│ Reg Blocked (%dx%d)       │ %8.3f │ %8.2f │ %7.1f%% │\n",
           TM, TN, regMs, gflops / regMs, 100.0f * cublasMs / regMs);
    
    printf("└─────────────────────────┴──────────┴──────────┴──────────┘\n\n");
    
    printf("Speedup from register blocking: %.2fx\n", basicMs / regMs);
    printf("Max error: %.2e\n\n", maxErr);
    
    printf("Register Analysis (TM=%d, TN=%d):\n", TM, TN);
    printf("  Accumulators: %d registers (float)\n", TM * TN);
    printf("  A fragment: %d registers\n", TM);
    printf("  B fragment: %d registers\n", TN);
    printf("  Total: ~%d registers per thread\n", TM * TN + TM + TN + 10);
    printf("  Compute per shared load: %d FMAs\n", TM * TN);
    
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
