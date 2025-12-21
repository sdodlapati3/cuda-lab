/**
 * Week 23, Day 2: 2×2 Thread Tile
 * 
 * Each thread computes a 2×2 tile of outputs using register blocking.
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

// Basic tiled GEMM (reference)
template<int BK>
__global__ void tiledGemm(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[BK][BK + 1];
    __shared__ float Bs[BK][BK + 1];
    
    int row = blockIdx.y * BK + threadIdx.y;
    int col = blockIdx.x * BK + threadIdx.x;
    float sum = 0.0f;
    
    for (int t = 0; t < K; t += BK) {
        As[threadIdx.y][threadIdx.x] = (row < M && t + threadIdx.x < K) ? 
            A[row * K + t + threadIdx.x] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (t + threadIdx.y < K && col < N) ? 
            B[(t + threadIdx.y) * N + col] : 0.0f;
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    
    if (row < M && col < N) C[row * N + col] = sum;
}

// 2×2 Thread Tile GEMM
// Block: BM × BN output tile
// Thread: 2 × 2 output elements
template<int BM, int BN, int BK>
__global__ void threadTile2x2Gemm(const float* A, const float* B, float* C, 
                                   int M, int N, int K) {
    // Shared memory for tiles
    __shared__ float As[BK][BM + 1];
    __shared__ float Bs[BK][BN + 1];
    
    // Thread block computes BM × BN, each thread computes 2×2
    // So we need (BM/2) × (BN/2) threads
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // Output positions for this thread's 2×2 tile
    const int rowStart = blockIdx.y * BM + ty * 2;
    const int colStart = blockIdx.x * BN + tx * 2;
    
    // Register accumulators for 2×2 output
    float c00 = 0.0f, c01 = 0.0f;
    float c10 = 0.0f, c11 = 0.0f;
    
    // Number of threads for loading
    const int numThreads = (BM / 2) * (BN / 2);
    const int tid = ty * (BN / 2) + tx;
    
    // Loop over K tiles
    for (int tileK = 0; tileK < K; tileK += BK) {
        // Collaborative load of A tile (BK × BM)
        for (int i = tid; i < BK * BM; i += numThreads) {
            int loadK = i / BM;
            int loadM = i % BM;
            int globalK = tileK + loadK;
            int globalM = blockIdx.y * BM + loadM;
            As[loadK][loadM] = (globalK < K && globalM < M) ? 
                A[globalM * K + globalK] : 0.0f;
        }
        
        // Collaborative load of B tile (BK × BN)
        for (int i = tid; i < BK * BN; i += numThreads) {
            int loadK = i / BN;
            int loadN = i % BN;
            int globalK = tileK + loadK;
            int globalN = blockIdx.x * BN + loadN;
            Bs[loadK][loadN] = (globalK < K && globalN < N) ? 
                B[globalK * N + globalN] : 0.0f;
        }
        
        __syncthreads();
        
        // Compute 2×2 output using outer products
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            // Load A fragment (2 elements)
            float a0 = As[k][ty * 2];
            float a1 = As[k][ty * 2 + 1];
            
            // Load B fragment (2 elements)
            float b0 = Bs[k][tx * 2];
            float b1 = Bs[k][tx * 2 + 1];
            
            // Outer product: 4 FMAs
            c00 += a0 * b0;
            c01 += a0 * b1;
            c10 += a1 * b0;
            c11 += a1 * b1;
        }
        
        __syncthreads();
    }
    
    // Write 2×2 output tile
    if (rowStart < M && colStart < N) {
        C[rowStart * N + colStart] = c00;
    }
    if (rowStart < M && colStart + 1 < N) {
        C[rowStart * N + colStart + 1] = c01;
    }
    if (rowStart + 1 < M && colStart < N) {
        C[(rowStart + 1) * N + colStart] = c10;
    }
    if (rowStart + 1 < M && colStart + 1 < N) {
        C[(rowStart + 1) * N + colStart + 1] = c11;
    }
}

void initMatrix(float* mat, int size) {
    for (int i = 0; i < size; i++) {
        mat[i] = (rand() / (float)RAND_MAX - 0.5f);
    }
}

int main() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║           Week 23, Day 2: 2×2 Thread Tile                    ║\n");
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
    
    printf("Performance:\n");
    printf("┌─────────────────────────┬──────────┬──────────┬──────────┐\n");
    printf("│ Implementation          │ Time(ms) │ TFLOPS   │ %%cuBLAS  │\n");
    printf("├─────────────────────────┼──────────┼──────────┼──────────┤\n");
    printf("│ cuBLAS                  │ %8.3f │ %8.2f │   100.0%% │\n", 
           cublasMs, gflops / cublasMs);
    
    // Basic tiled (32×32)
    constexpr int BK = 32;
    dim3 blockBasic(BK, BK);
    dim3 gridBasic((N + BK - 1) / BK, (M + BK - 1) / BK);
    
    tiledGemm<BK><<<gridBasic, blockBasic>>>(dA, dB, dC, M, N, K);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        tiledGemm<BK><<<gridBasic, blockBasic>>>(dA, dB, dC, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float basicMs;
    cudaEventElapsedTime(&basicMs, start, stop);
    basicMs /= iterations;
    
    printf("│ Basic Tiled (32×32)     │ %8.3f │ %8.2f │ %7.1f%% │\n",
           basicMs, gflops / basicMs, 100.0f * cublasMs / basicMs);
    
    // 2×2 Thread Tile
    constexpr int BM = 64, BN = 64, BK2 = 16;
    // Each thread computes 2×2, so block is (BN/2) × (BM/2)
    dim3 block2x2(BN / 2, BM / 2);  // 32 × 32 = 1024 threads
    dim3 grid2x2((N + BN - 1) / BN, (M + BM - 1) / BM);
    
    threadTile2x2Gemm<BM, BN, BK2><<<grid2x2, block2x2>>>(dA, dB, dC, M, N, K);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        threadTile2x2Gemm<BM, BN, BK2><<<grid2x2, block2x2>>>(dA, dB, dC, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float tile2x2Ms;
    cudaEventElapsedTime(&tile2x2Ms, start, stop);
    tile2x2Ms /= iterations;
    
    CHECK_CUDA(cudaMemcpy(hC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    float maxErr = 0;
    for (int i = 0; i < M * N; i++) maxErr = fmaxf(maxErr, fabsf(hC[i] - hRef[i]));
    
    printf("│ 2×2 Thread Tile         │ %8.3f │ %8.2f │ %7.1f%% │\n",
           tile2x2Ms, gflops / tile2x2Ms, 100.0f * cublasMs / tile2x2Ms);
    printf("└─────────────────────────┴──────────┴──────────┴──────────┘\n\n");
    
    printf("Speedup over basic: %.2fx\n", basicMs / tile2x2Ms);
    printf("Max error: %.2e\n\n", maxErr);
    
    printf("Analysis:\n");
    printf("  Basic:   1 output/thread, 2 loads/FMA → intensity 0.5\n");
    printf("  2×2:     4 outputs/thread, 4 loads/4 FMAs → intensity 1.0\n");
    printf("  Improvement: 2× compute intensity\n");
    
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
