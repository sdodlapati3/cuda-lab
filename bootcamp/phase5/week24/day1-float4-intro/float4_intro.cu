/**
 * Week 24, Day 1: float4 Vector Loads
 * 
 * Compare scalar vs vectorized memory access for GEMM.
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

// Scalar load GEMM
template<int BM, int BN, int BK, int TM, int TN>
__global__ void scalarLoadGemm(const float* __restrict__ A, 
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
    const int threadsY = BM / TM;
    const int numThreads = threadsX * threadsY;
    const int tid = ty * threadsX + tx;
    
    for (int tileK = 0; tileK < K; tileK += BK) {
        // Scalar loads - one float at a time
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

// float4 vector load GEMM
// Assumes K is multiple of 4 for aligned loads
template<int BM, int BN, int BK, int TM, int TN>
__global__ void float4LoadGemm(const float* __restrict__ A, 
                                const float* __restrict__ B, 
                                float* __restrict__ C, 
                                int M, int N, int K) {
    __shared__ float As[BK][BM + 4];  // Extra padding for float4
    __shared__ float Bs[BK][BN + 4];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int rowStart = blockIdx.y * BM + ty * TM;
    const int colStart = blockIdx.x * BN + tx * TN;
    
    float acc[TM][TN] = {{0}};
    
    const int threadsX = BN / TN;
    const int threadsY = BM / TM;
    const int numThreads = threadsX * threadsY;
    const int tid = ty * threadsX + tx;
    
    // Number of float4 loads needed
    const int numFloat4A = (BK * BM + 3) / 4;
    const int numFloat4B = (BK * BN + 3) / 4;
    
    for (int tileK = 0; tileK < K; tileK += BK) {
        // Vector loads for A
        for (int i = tid; i < numFloat4A; i += numThreads) {
            int idx = i * 4;
            int loadK = idx / BM;
            int loadM = idx % BM;
            int globalK = tileK + loadK;
            int globalM = blockIdx.y * BM + loadM;
            
            if (globalK < K && globalM + 3 < M && loadM + 3 < BM) {
                // Aligned float4 load from global memory
                float4 tmp = reinterpret_cast<const float4*>(&A[globalM * K + globalK])[0];
                As[loadK][loadM] = tmp.x;
                As[loadK][loadM + 1] = tmp.y;
                As[loadK][loadM + 2] = tmp.z;
                As[loadK][loadM + 3] = tmp.w;
            } else {
                // Fall back to scalar for boundary
                for (int j = 0; j < 4 && loadM + j < BM; j++) {
                    int gM = globalM + j;
                    As[loadK][loadM + j] = (globalK < K && gM < M) ? A[gM * K + globalK] : 0.0f;
                }
            }
        }
        
        // Vector loads for B (along N dimension for coalescing)
        for (int i = tid; i < numFloat4B; i += numThreads) {
            int idx = i * 4;
            int loadK = idx / BN;
            int loadN = idx % BN;
            int globalK = tileK + loadK;
            int globalN = blockIdx.x * BN + loadN;
            
            if (globalK < K && globalN + 3 < N && loadN + 3 < BN) {
                float4 tmp = reinterpret_cast<const float4*>(&B[globalK * N + globalN])[0];
                Bs[loadK][loadN] = tmp.x;
                Bs[loadK][loadN + 1] = tmp.y;
                Bs[loadK][loadN + 2] = tmp.z;
                Bs[loadK][loadN + 3] = tmp.w;
            } else {
                for (int j = 0; j < 4 && loadN + j < BN; j++) {
                    int gN = globalN + j;
                    Bs[loadK][loadN + j] = (globalK < K && gN < N) ? B[globalK * N + gN] : 0.0f;
                }
            }
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
    printf("║           Week 24, Day 1: float4 Vector Loads                ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    printf("float4 loads: 128-bit (4 floats) in single transaction\n");
    printf("Benefits: 4× fewer instructions, better memory efficiency\n\n");
    
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
    
    // Scalar loads
    constexpr int BM = 64, BN = 64, BK = 16, TM = 4, TN = 4;
    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    
    scalarLoadGemm<BM, BN, BK, TM, TN><<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        scalarLoadGemm<BM, BN, BK, TM, TN><<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float scalarMs;
    cudaEventElapsedTime(&scalarMs, start, stop);
    scalarMs /= iterations;
    
    printf("│ Scalar Loads            │ %8.3f │ %8.2f │ %7.1f%% │\n",
           scalarMs, gflops / scalarMs, 100.0f * cublasMs / scalarMs);
    
    // float4 loads
    float4LoadGemm<BM, BN, BK, TM, TN><<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        float4LoadGemm<BM, BN, BK, TM, TN><<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float float4Ms;
    cudaEventElapsedTime(&float4Ms, start, stop);
    float4Ms /= iterations;
    
    CHECK_CUDA(cudaMemcpy(hC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    float maxErr = 0;
    for (int i = 0; i < M * N; i++) maxErr = fmaxf(maxErr, fabsf(hC[i] - hRef[i]));
    
    printf("│ float4 Loads            │ %8.3f │ %8.2f │ %7.1f%% │\n",
           float4Ms, gflops / float4Ms, 100.0f * cublasMs / float4Ms);
    printf("└─────────────────────────┴──────────┴──────────┴──────────┘\n\n");
    
    printf("float4 speedup: %.2fx over scalar\n", scalarMs / float4Ms);
    printf("Max error: %.2e\n\n");
    
    printf("Memory Access Analysis:\n");
    printf("  Scalar: %d load instructions per tile\n", BK * BM + BK * BN);
    printf("  float4: %d load instructions per tile\n", (BK * BM + BK * BN) / 4);
    
    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    delete[] hA; delete[] hB; delete[] hC; delete[] hRef;
    
    return 0;
}
