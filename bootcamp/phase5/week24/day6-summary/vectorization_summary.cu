/**
 * Week 24, Day 6: Vectorization Summary
 * 
 * Compare all vectorization techniques.
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

// Best optimized kernel from this week
template<int BM, int BN, int BK, int TM, int TN>
__global__ void optimizedGemm(const float* __restrict__ A, 
                               const float* __restrict__ B, 
                               float* __restrict__ C, 
                               int M, int N, int K) {
    __shared__ float As[2][BK][BM + 4];
    __shared__ float Bs[2][BK][BN + 4];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int rowStart = blockIdx.y * BM + ty * TM;
    const int colStart = blockIdx.x * BN + tx * TN;
    
    float acc[TM][TN] = {{0}};
    
    const int threadsX = BN / TN;
    const int numThreads = (BM / TM) * threadsX;
    const int tid = ty * threadsX + tx;
    
    int numTiles = (K + BK - 1) / BK;
    int buf = 0;
    
    // Prefetch first tile
    for (int i = tid; i < BK * BM; i += numThreads) {
        int loadK = i / BM, loadM = i % BM;
        int globalK = loadK, globalM = blockIdx.y * BM + loadM;
        if (globalK < K && globalM < M)
            __pipeline_memcpy_async(&As[buf][loadK][loadM], &A[globalM * K + globalK], 4);
    }
    for (int i = tid; i < BK * BN; i += numThreads) {
        int loadK = i / BN, loadN = i % BN;
        int globalK = loadK, globalN = blockIdx.x * BN + loadN;
        if (globalK < K && globalN < N)
            __pipeline_memcpy_async(&Bs[buf][loadK][loadN], &B[globalK * N + globalN], 4);
    }
    __pipeline_commit();
    
    for (int tile = 0; tile < numTiles; tile++) {
        int nextBuf = buf ^ 1;
        
        if (tile + 1 < numTiles) {
            int tileK = (tile + 1) * BK;
            for (int i = tid; i < BK * BM; i += numThreads) {
                int loadK = i / BM, loadM = i % BM;
                int globalK = tileK + loadK, globalM = blockIdx.y * BM + loadM;
                if (globalK < K && globalM < M)
                    __pipeline_memcpy_async(&As[nextBuf][loadK][loadM], &A[globalM * K + globalK], 4);
            }
            for (int i = tid; i < BK * BN; i += numThreads) {
                int loadK = i / BN, loadN = i % BN;
                int globalK = tileK + loadK, globalN = blockIdx.x * BN + loadN;
                if (globalK < K && globalN < N)
                    __pipeline_memcpy_async(&Bs[nextBuf][loadK][loadN], &B[globalK * N + globalN], 4);
            }
            __pipeline_commit();
        }
        
        __pipeline_wait_prior(1);
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            float regA[TM], regB[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++) regA[i] = As[buf][k][ty * TM + i];
            #pragma unroll
            for (int j = 0; j < TN; j++) regB[j] = Bs[buf][k][tx * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] += regA[i] * regB[j];
        }
        
        buf = nextBuf;
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
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║         Week 24, Day 6: Vectorization Summary                    ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");
    
    printf("Week 24 Techniques:\n");
    printf("  Day 1: float4 vector loads (4× fewer instructions)\n");
    printf("  Day 2: Transposed loads (better coalescing)\n");
    printf("  Day 3: Async copy (overlap memory & compute)\n");
    printf("  Day 4: Swizzled shared memory (zero bank conflicts)\n");
    printf("  Day 5: Combined optimizations\n\n");
    
    const int M = 4096, N = 4096, K = 4096;
    double gflops = 2.0 * M * N * K / 1e9;
    printf("Benchmark: %d × %d × %d (%.1f GFLOP)\n\n", M, N, K, gflops);
    
    float *hA = new float[M * K];
    float *hB = new float[K * N];
    float *hRef = new float[M * N];
    float *hC = new float[M * N];
    
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
    
    // Our optimized kernel
    constexpr int BM = 64, BN = 64, BK = 16, TM = 4, TN = 4;
    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    
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
    
    printf("Final Results:\n");
    printf("┌─────────────────────────┬──────────┬──────────┬──────────┐\n");
    printf("│ Implementation          │ Time(ms) │ TFLOPS   │ %%cuBLAS  │\n");
    printf("├─────────────────────────┼──────────┼──────────┼──────────┤\n");
    printf("│ cuBLAS                  │ %8.3f │ %8.2f │   100.0%% │\n",
           cublasMs, gflops / cublasMs);
    printf("│ Week 24 Optimized       │ %8.3f │ %8.2f │ %7.1f%% │\n",
           optMs, gflops / optMs, 100.0f * cublasMs / optMs);
    printf("└─────────────────────────┴──────────┴──────────┴──────────┘\n\n");
    
    printf("Max error: %.2e\n\n", maxErr);
    
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("Week 24 Key Takeaways:\n");
    printf("───────────────────────────────────────────────────────────────────\n");
    printf("1. Vector loads reduce memory instruction count\n");
    printf("2. Async copy overlaps memory latency with compute\n");
    printf("3. Double buffering enables continuous execution\n");
    printf("4. Proper memory layout essential for coalescing\n");
    printf("5. Next: Warp-level optimizations (Week 25)\n");
    printf("═══════════════════════════════════════════════════════════════════\n");
    
    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    delete[] hA; delete[] hB; delete[] hC; delete[] hRef;
    
    return 0;
}
