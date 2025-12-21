/**
 * Week 24, Day 3: Async Copy with cp.async
 * 
 * Use asynchronous copy for overlapping memory transfers.
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

// Synchronous load GEMM
template<int BM, int BN, int BK, int TM, int TN>
__global__ void syncLoadGemm(const float* __restrict__ A, 
                              const float* __restrict__ B, 
                              float* __restrict__ C, 
                              int M, int N, int K) {
    __shared__ float As[BK][BM + 1];
    __shared__ float Bs[BK][BN + 1];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int rowStart = blockIdx.y * BM + ty * TM;
    const int colStart = blockIdx.x * BN + tx * TN;
    
    float acc[TM][TN] = {{0}};
    
    const int threadsX = BN / TN;
    const int numThreads = (BM / TM) * threadsX;
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

// Async copy GEMM (requires sm_80+)
template<int BM, int BN, int BK, int TM, int TN>
__global__ void asyncCopyGemm(const float* __restrict__ A, 
                               const float* __restrict__ B, 
                               float* __restrict__ C, 
                               int M, int N, int K) {
    // Double buffer for async pipeline
    __shared__ float As[2][BK][BM + 1];
    __shared__ float Bs[2][BK][BN + 1];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int rowStart = blockIdx.y * BM + ty * TM;
    const int colStart = blockIdx.x * BN + tx * TN;
    
    float acc[TM][TN] = {{0}};
    
    const int threadsX = BN / TN;
    const int numThreads = (BM / TM) * threadsX;
    const int tid = ty * threadsX + tx;
    
    int numTiles = (K + BK - 1) / BK;
    int writeBuffer = 0;
    int readBuffer = 0;
    
    // Prefetch first tile asynchronously
    for (int i = tid; i < BK * BM; i += numThreads) {
        int loadK = i / BM, loadM = i % BM;
        int globalK = loadK, globalM = blockIdx.y * BM + loadM;
        if (globalK < K && globalM < M) {
            __pipeline_memcpy_async(&As[writeBuffer][loadK][loadM], 
                                     &A[globalM * K + globalK], sizeof(float));
        }
    }
    for (int i = tid; i < BK * BN; i += numThreads) {
        int loadK = i / BN, loadN = i % BN;
        int globalK = loadK, globalN = blockIdx.x * BN + loadN;
        if (globalK < K && globalN < N) {
            __pipeline_memcpy_async(&Bs[writeBuffer][loadK][loadN], 
                                     &B[globalK * N + globalN], sizeof(float));
        }
    }
    __pipeline_commit();
    writeBuffer ^= 1;
    
    for (int tile = 0; tile < numTiles; tile++) {
        int nextTile = tile + 1;
        
        // Start loading next tile (if exists) into write buffer
        if (nextTile < numTiles) {
            int tileK = nextTile * BK;
            for (int i = tid; i < BK * BM; i += numThreads) {
                int loadK = i / BM, loadM = i % BM;
                int globalK = tileK + loadK, globalM = blockIdx.y * BM + loadM;
                if (globalK < K && globalM < M) {
                    __pipeline_memcpy_async(&As[writeBuffer][loadK][loadM], 
                                             &A[globalM * K + globalK], sizeof(float));
                }
            }
            for (int i = tid; i < BK * BN; i += numThreads) {
                int loadK = i / BN, loadN = i % BN;
                int globalK = tileK + loadK, globalN = blockIdx.x * BN + loadN;
                if (globalK < K && globalN < N) {
                    __pipeline_memcpy_async(&Bs[writeBuffer][loadK][loadN], 
                                             &B[globalK * N + globalN], sizeof(float));
                }
            }
            __pipeline_commit();
        }
        
        // Wait for current tile to arrive
        __pipeline_wait_prior(1);
        __syncthreads();
        
        // Compute on read buffer
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            float regA[TM], regB[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++) regA[i] = As[readBuffer][k][ty * TM + i];
            #pragma unroll
            for (int j = 0; j < TN; j++) regB[j] = Bs[readBuffer][k][tx * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] += regA[i] * regB[j];
        }
        
        // Swap buffers
        readBuffer ^= 1;
        writeBuffer ^= 1;
        __syncthreads();
    }
    
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
            if (rowStart + i < M && colStart + j < N)
                C[(rowStart + i) * N + colStart + j] = acc[i][j];
}

void initMatrix(float* mat, int size) {
    for (int i = 0; i < size; i++) mat[i] = (rand() / (float)RAND_MAX - 0.5f);
}

int main() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║           Week 24, Day 3: Async Copy (cp.async)              ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);
    printf("Device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    
    if (prop.major < 8) {
        printf("Warning: cp.async requires sm_80+. Results may vary.\n");
    }
    printf("\n");
    
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
    
    constexpr int BM = 64, BN = 64, BK = 16, TM = 4, TN = 4;
    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    
    // Synchronous
    syncLoadGemm<BM, BN, BK, TM, TN><<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        syncLoadGemm<BM, BN, BK, TM, TN><<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float syncMs;
    cudaEventElapsedTime(&syncMs, start, stop);
    syncMs /= iterations;
    
    printf("│ Sync Load               │ %8.3f │ %8.2f │ %7.1f%% │\n",
           syncMs, gflops / syncMs, 100.0f * cublasMs / syncMs);
    
    // Async copy
    asyncCopyGemm<BM, BN, BK, TM, TN><<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        asyncCopyGemm<BM, BN, BK, TM, TN><<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float asyncMs;
    cudaEventElapsedTime(&asyncMs, start, stop);
    asyncMs /= iterations;
    
    CHECK_CUDA(cudaMemcpy(hC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    float maxErr = 0;
    for (int i = 0; i < M * N; i++) maxErr = fmaxf(maxErr, fabsf(hC[i] - hRef[i]));
    
    printf("│ Async Copy (cp.async)   │ %8.3f │ %8.2f │ %7.1f%% │\n",
           asyncMs, gflops / asyncMs, 100.0f * cublasMs / asyncMs);
    printf("└─────────────────────────┴──────────┴──────────┴──────────┘\n\n");
    
    printf("Async speedup: %.2fx over sync\n", syncMs / asyncMs);
    printf("Max error: %.2e\n\n");
    
    printf("Key Benefits:\n");
    printf("  - Overlaps memory loads with computation\n");
    printf("  - Double buffering hides load latency\n");
    printf("  - Direct global→shared transfer (bypasses L1)\n");
    
    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    delete[] hA; delete[] hB; delete[] hC; delete[] hRef;
    
    return 0;
}
