/**
 * Week 23, Day 5: Register Pressure Analysis
 * 
 * Analyze how register usage affects GEMM performance.
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

// Template for configurable thread tile
template<int BM, int BN, int BK, int TM, int TN>
__global__ void configurableGemm(const float* __restrict__ A, 
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

// Wrapper for kernel attributes
template<int BM, int BN, int BK, int TM, int TN>
void getKernelInfo(const char* name) {
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, configurableGemm<BM, BN, BK, TM, TN>);
    
    int threadsPerBlock = (BM / TM) * (BN / TN);
    int maxWarps = attr.maxThreadsPerBlock / 32;
    
    // Calculate theoretical occupancy
    int sharedMem = (BK * (BM + 1) + BK * (BN + 1)) * sizeof(float);
    
    printf("│ %-12s │ %3d×%-3d │ %4d │ %6d │ %5d B │ %3d × %-3d │\n",
           name, TM, TN, threadsPerBlock, attr.numRegs, sharedMem, BM, BN);
}

template<int BM, int BN, int BK, int TM, int TN>
float benchKernel(const float* dA, const float* dB, float* dC, 
                  int M, int N, int K, int iterations) {
    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    
    // Warmup
    configurableGemm<BM, BN, BK, TM, TN><<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaDeviceSynchronize();
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        configurableGemm<BM, BN, BK, TM, TN><<<grid, block>>>(dA, dB, dC, M, N, K);
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
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║         Week 23, Day 5: Register Pressure Analysis           ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    printf("Kernel Configuration Analysis:\n");
    printf("┌──────────────┬─────────┬──────┬────────┬─────────┬───────────┐\n");
    printf("│ Name         │ Tile    │ Thds │ Regs   │ Shared  │ Block     │\n");
    printf("├──────────────┼─────────┼──────┼────────┼─────────┼───────────┤\n");
    
    getKernelInfo<64, 64, 16, 2, 2>("2x2 Tile");
    getKernelInfo<64, 64, 16, 4, 4>("4x4 Tile");
    getKernelInfo<128, 128, 8, 8, 8>("8x8 Tile");
    
    printf("└──────────────┴─────────┴──────┴────────┴─────────┴───────────┘\n\n");
    
    const int M = 2048, N = 2048, K = 2048;
    double gflops = 2.0 * M * N * K / 1e9;
    printf("Benchmarking %d × %d × %d (%.2f GFLOP)\n\n", M, N, K, gflops);
    
    float *hA = new float[M * K];
    float *hB = new float[K * N];
    
    srand(42);
    initMatrix(hA, M * K);
    initMatrix(hB, K * N);
    
    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC, M * N * sizeof(float)));
    
    CHECK_CUDA(cudaMemcpy(dA, hA, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, K * N * sizeof(float), cudaMemcpyHostToDevice));
    
    int iterations = 20;
    
    // cuBLAS reference
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;
    
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
    
    printf("Performance Results:\n");
    printf("┌──────────────┬──────────┬──────────┬──────────┬──────────┐\n");
    printf("│ Config       │ Time(ms) │ TFLOPS   │ %%cuBLAS  │ FMA/Load │\n");
    printf("├──────────────┼──────────┼──────────┼──────────┼──────────┤\n");
    printf("│ cuBLAS       │ %8.3f │ %8.2f │   100.0%% │    -     │\n",
           cublasMs, gflops / cublasMs);
    
    // Benchmark each configuration
    float ms2x2 = benchKernel<64, 64, 16, 2, 2>(dA, dB, dC, M, N, K, iterations);
    printf("│ 2×2 Tile     │ %8.3f │ %8.2f │ %7.1f%% │   1.0    │\n",
           ms2x2, gflops / ms2x2, 100.0f * cublasMs / ms2x2);
    
    float ms4x4 = benchKernel<64, 64, 16, 4, 4>(dA, dB, dC, M, N, K, iterations);
    printf("│ 4×4 Tile     │ %8.3f │ %8.2f │ %7.1f%% │   2.0    │\n",
           ms4x4, gflops / ms4x4, 100.0f * cublasMs / ms4x4);
    
    float ms8x8 = benchKernel<128, 128, 8, 8, 8>(dA, dB, dC, M, N, K, iterations);
    printf("│ 8×8 Tile     │ %8.3f │ %8.2f │ %7.1f%% │   4.0    │\n",
           ms8x8, gflops / ms8x8, 100.0f * cublasMs / ms8x8);
    
    printf("└──────────────┴──────────┴──────────┴──────────┴──────────┘\n\n");
    
    printf("Analysis:\n");
    printf("  - 2×2: Low register pressure, high occupancy, memory-bound\n");
    printf("  - 4×4: Balanced registers/occupancy, good compute intensity\n");
    printf("  - 8×8: High registers, lower occupancy, compute-dense\n\n");
    
    printf("Optimal tile depends on matrix size and GPU architecture.\n");
    printf("Use nvcc --ptxas-options=-v to verify actual register counts.\n");
    
    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    delete[] hA; delete[] hB;
    
    return 0;
}
