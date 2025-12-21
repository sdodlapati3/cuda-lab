/**
 * Week 24, Day 5: Combined Vectorized GEMM
 * 
 * All vectorization techniques combined.
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

// Combined optimized GEMM
template<int BM, int BN, int BK, int TM, int TN>
__global__ void optimizedGemm(const float* __restrict__ A, 
                               const float* __restrict__ B, 
                               float* __restrict__ C, 
                               int M, int N, int K) {
    // Double-buffered shared memory with padding for bank conflicts
    __shared__ float As[2][BK][BM + 4];
    __shared__ float Bs[2][BK][BN + 4];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int rowStart = blockIdx.y * BM + ty * TM;
    const int colStart = blockIdx.x * BN + tx * TN;
    
    float acc[TM][TN] = {{0}};
    
    const int threadsX = BN / TN;
    const int threadsY = BM / TM;
    const int numThreads = threadsX * threadsY;
    const int tid = ty * threadsX + tx;
    
    int numTiles = (K + BK - 1) / BK;
    int readBuffer = 0;
    int writeBuffer = 0;
    
    // Prefetch first tile
    {
        int tileK = 0;
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
        writeBuffer ^= 1;
    }
    
    for (int tile = 0; tile < numTiles; tile++) {
        // Start next tile load
        if (tile + 1 < numTiles) {
            int tileK = (tile + 1) * BK;
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
        
        // Wait for current tile
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
        
        readBuffer ^= 1;
        writeBuffer ^= 1;
        __syncthreads();
    }
    
    // Store results
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
            if (rowStart + i < M && colStart + j < N)
                C[(rowStart + i) * N + colStart + j] = acc[i][j];
}

// Basic reference kernel (for comparison)
template<int BM, int BN, int BK, int TM, int TN>
__global__ void basicGemm(const float* __restrict__ A, 
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

void initMatrix(float* mat, int size) {
    for (int i = 0; i < size; i++) mat[i] = (rand() / (float)RAND_MAX - 0.5f);
}

int main() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║         Week 24, Day 5: Combined Vectorized GEMM             ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    int sizes[] = {1024, 2048, 4096};
    int numSizes = 3;
    
    float *hA = new float[4096 * 4096];
    float *hB = new float[4096 * 4096];
    float *hC = new float[4096 * 4096];
    float *hRef = new float[4096 * 4096];
    
    srand(42);
    for (int i = 0; i < 4096 * 4096; i++) {
        hA[i] = (rand() / (float)RAND_MAX - 0.5f);
        hB[i] = (rand() / (float)RAND_MAX - 0.5f);
    }
    
    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, 4096 * 4096 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB, 4096 * 4096 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC, 4096 * 4096 * sizeof(float)));
    
    CHECK_CUDA(cudaMemcpy(dA, hA, 4096 * 4096 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, 4096 * 4096 * sizeof(float), cudaMemcpyHostToDevice));
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;
    int iterations = 20;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    for (int s = 0; s < numSizes; s++) {
        int M = sizes[s], N = sizes[s], K = sizes[s];
        double gflops = 2.0 * M * N * K / 1e9;
        
        printf("Matrix: %d × %d (%.1f GFLOP)\n", M, M, gflops);
        printf("┌─────────────────────────┬──────────┬──────────┬──────────┐\n");
        printf("│ Implementation          │ Time(ms) │ TFLOPS   │ %%cuBLAS  │\n");
        printf("├─────────────────────────┼──────────┼──────────┼──────────┤\n");
        
        // cuBLAS
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
        
        printf("│ cuBLAS                  │ %8.3f │ %8.2f │   100.0%% │\n",
               cublasMs, gflops / cublasMs);
        
        constexpr int BM = 64, BN = 64, BK = 16, TM = 4, TN = 4;
        dim3 block(BN / TN, BM / TM);
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        
        // Basic
        basicGemm<BM, BN, BK, TM, TN><<<grid, block>>>(dA, dB, dC, M, N, K);
        cudaDeviceSynchronize();
        
        cudaEventRecord(start);
        for (int i = 0; i < iterations; i++) {
            basicGemm<BM, BN, BK, TM, TN><<<grid, block>>>(dA, dB, dC, M, N, K);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float basicMs;
        cudaEventElapsedTime(&basicMs, start, stop);
        basicMs /= iterations;
        
        printf("│ Basic (4×4 tile)        │ %8.3f │ %8.2f │ %7.1f%% │\n",
               basicMs, gflops / basicMs, 100.0f * cublasMs / basicMs);
        
        // Optimized
        optimizedGemm<BM, BN, BK, TM, TN><<<grid, block>>>(dA, dB, dC, M, N, K);
        cudaDeviceSynchronize();
        
        cudaEventRecord(start);
        for (int i = 0; i < iterations; i++) {
            optimizedGemm<BM, BN, BK, TM, TN><<<grid, block>>>(dA, dB, dC, M, N, K);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float optMs;
        cudaEventElapsedTime(&optMs, start, stop);
        optMs /= iterations;
        
        CHECK_CUDA(cudaMemcpy(hC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));
        float maxErr = 0;
        for (int i = 0; i < M * N; i++) maxErr = fmaxf(maxErr, fabsf(hC[i] - hRef[i]));
        
        printf("│ Optimized (combined)    │ %8.3f │ %8.2f │ %7.1f%% │\n",
               optMs, gflops / optMs, 100.0f * cublasMs / optMs);
        printf("└─────────────────────────┴──────────┴──────────┴──────────┘\n");
        printf("  Speedup: %.2fx, Max error: %.2e\n\n", basicMs / optMs, maxErr);
    }
    
    printf("Optimizations Combined:\n");
    printf("  ✓ Double buffering (latency hiding)\n");
    printf("  ✓ Async copy (cp.async)\n");
    printf("  ✓ 4×4 register blocking\n");
    printf("  ✓ Shared memory padding\n");
    
    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    delete[] hA; delete[] hB; delete[] hC; delete[] hRef;
    
    return 0;
}
