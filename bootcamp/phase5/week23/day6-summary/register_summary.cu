/**
 * Week 23, Day 6: Register Blocking Summary
 * 
 * Comprehensive comparison of all register blocking strategies.
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

// Generic thread-tiled GEMM
template<int BM, int BN, int BK, int TM, int TN>
__global__ void threadTileGemm(const float* __restrict__ A, 
                                const float* __restrict__ B, 
                                float* __restrict__ C, 
                                int M, int N, int K) {
    __shared__ float As[BK][BM + 1];
    __shared__ float Bs[BK][BN + 1];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int rowStart = blockIdx.y * BM + ty * TM;
    const int colStart = blockIdx.x * BN + tx * TN;
    
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
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

template<int BM, int BN, int BK, int TM, int TN>
float benchConfig(const float* dA, const float* dB, float* dC,
                  int M, int N, int K, int iterations) {
    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    
    threadTileGemm<BM, BN, BK, TM, TN><<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaDeviceSynchronize();
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        threadTileGemm<BM, BN, BK, TM, TN><<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / iterations;
}

void initMatrix(float* mat, int size) {
    for (int i = 0; i < size; i++) mat[i] = (rand() / (float)RAND_MAX - 0.5f);
}

int main() {
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║       Week 23, Day 6: Register Blocking Summary                  ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");
    
    // Test multiple matrix sizes
    int sizes[] = {1024, 2048, 4096};
    int numSizes = 3;
    
    float *hA = new float[4096 * 4096];
    float *hB = new float[4096 * 4096];
    
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
    
    for (int s = 0; s < numSizes; s++) {
        int M = sizes[s], N = sizes[s], K = sizes[s];
        double gflops = 2.0 * M * N * K / 1e9;
        
        printf("Matrix Size: %d × %d × %d (%.2f GFLOP)\n", M, N, K, gflops);
        printf("┌──────────────┬──────────┬──────────┬──────────┬─────────┐\n");
        printf("│ Config       │ Time(ms) │ TFLOPS   │ %%cuBLAS  │ Eff/Thr │\n");
        printf("├──────────────┼──────────┼──────────┼──────────┼─────────┤\n");
        
        // cuBLAS reference
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);
        cudaDeviceSynchronize();
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        for (int i = 0; i < iterations; i++) {
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float cublasMs;
        cudaEventElapsedTime(&cublasMs, start, stop);
        cublasMs /= iterations;
        
        printf("│ cuBLAS       │ %8.3f │ %8.2f │   100.0%% │    -    │\n",
               cublasMs, gflops / cublasMs);
        
        // Test configurations
        float ms1 = benchConfig<32, 32, 8, 1, 1>(dA, dB, dC, M, N, K, iterations);
        printf("│ 1×1 Tile     │ %8.3f │ %8.2f │ %7.1f%% │   1 out │\n",
               ms1, gflops / ms1, 100.0f * cublasMs / ms1);
        
        float ms2 = benchConfig<64, 64, 16, 2, 2>(dA, dB, dC, M, N, K, iterations);
        printf("│ 2×2 Tile     │ %8.3f │ %8.2f │ %7.1f%% │   4 out │\n",
               ms2, gflops / ms2, 100.0f * cublasMs / ms2);
        
        float ms4 = benchConfig<64, 64, 16, 4, 4>(dA, dB, dC, M, N, K, iterations);
        printf("│ 4×4 Tile     │ %8.3f │ %8.2f │ %7.1f%% │  16 out │\n",
               ms4, gflops / ms4, 100.0f * cublasMs / ms4);
        
        float ms8 = benchConfig<128, 128, 8, 8, 8>(dA, dB, dC, M, N, K, iterations);
        printf("│ 8×8 Tile     │ %8.3f │ %8.2f │ %7.1f%% │  64 out │\n",
               ms8, gflops / ms8, 100.0f * cublasMs / ms8);
        
        printf("└──────────────┴──────────┴──────────┴──────────┴─────────┘\n\n");
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("Week 23 Key Takeaways:\n");
    printf("───────────────────────────────────────────────────────────────────\n");
    printf("1. Register blocking dramatically improves compute intensity\n");
    printf("2. 4×4 tiles offer excellent balance (16 outputs, ~24 registers)\n");
    printf("3. 8×8 tiles maximize intensity but reduce occupancy\n");
    printf("4. Optimal tile size depends on matrix dimensions\n");
    printf("5. Next: Vectorized memory access (Week 24)\n");
    printf("═══════════════════════════════════════════════════════════════════\n");
    
    cublasDestroy(handle);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    delete[] hA; delete[] hB;
    
    return 0;
}
