/**
 * Day 4: Row-Tiled GEMM
 * 
 * Demonstrates caching rows of A in shared memory.
 * This is a partial optimization - A reuse improves, B still naive.
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

// Naive GEMM for reference
__global__ void naiveGemm(const float* __restrict__ A,
                          const float* __restrict__ B,
                          float* __restrict__ C,
                          int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Row-Tiled GEMM: Cache rows of A in shared memory
// Each block processes TILE_M rows × entire width of B
template<int TILE_M, int BLOCK_SIZE>
__global__ void rowTiledGemm(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int M, int N, int K) {
    // Shared memory for tile of A rows
    // We cache TILE_M rows of A
    extern __shared__ float sharedA[];
    
    // Which tile of rows this block processes
    int rowStart = blockIdx.y * TILE_M;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Thread index within block
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int numThreads = blockDim.x * blockDim.y;
    
    // Cooperatively load TILE_M rows of A into shared memory
    // Each row has K elements, total = TILE_M * K elements
    int totalElements = TILE_M * K;
    for (int i = tid; i < totalElements; i += numThreads) {
        int localRow = i / K;
        int localCol = i % K;
        int globalRow = rowStart + localRow;
        if (globalRow < M) {
            sharedA[i] = A[globalRow * K + localCol];
        } else {
            sharedA[i] = 0.0f;
        }
    }
    __syncthreads();
    
    // Each thread computes one element using cached A
    int localRow = threadIdx.y;
    int globalRow = rowStart + localRow;
    
    if (globalRow < M && col < N) {
        float sum = 0.0f;
        float* rowA = sharedA + localRow * K;  // Pointer to this row in shared
        
        // A from shared memory (fast), B from global (slow)
        for (int k = 0; k < K; k++) {
            sum += rowA[k] * B[k * N + col];
        }
        C[globalRow * N + col] = sum;
    }
}

// Optimized version: Unroll inner loop
template<int TILE_M>
__global__ void rowTiledGemmUnroll(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int M, int N, int K) {
    extern __shared__ float sharedA[];
    
    int rowStart = blockIdx.y * TILE_M;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int numThreads = blockDim.x * blockDim.y;
    
    // Load tile of A
    int totalElements = TILE_M * K;
    for (int i = tid; i < totalElements; i += numThreads) {
        int localRow = i / K;
        int localCol = i % K;
        int globalRow = rowStart + localRow;
        sharedA[i] = (globalRow < M) ? A[globalRow * K + localCol] : 0.0f;
    }
    __syncthreads();
    
    int localRow = threadIdx.y;
    int globalRow = rowStart + localRow;
    
    if (globalRow < M && col < N) {
        float sum = 0.0f;
        float* rowA = sharedA + localRow * K;
        
        // Unroll by 4
        int k = 0;
        for (; k + 3 < K; k += 4) {
            sum += rowA[k]     * B[(k)     * N + col];
            sum += rowA[k + 1] * B[(k + 1) * N + col];
            sum += rowA[k + 2] * B[(k + 2) * N + col];
            sum += rowA[k + 3] * B[(k + 3) * N + col];
        }
        for (; k < K; k++) {
            sum += rowA[k] * B[k * N + col];
        }
        C[globalRow * N + col] = sum;
    }
}

void initMatrix(float* mat, int size, float scale = 1.0f) {
    for (int i = 0; i < size; i++) {
        mat[i] = (rand() / (float)RAND_MAX - 0.5f) * scale;
    }
}

float benchmarkKernel(void (*launcher)(const float*, const float*, float*, int, int, int),
                      const float* dA, const float* dB, float* dC,
                      int M, int N, int K, int iterations = 10) {
    // Warmup
    launcher(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        launcher(dA, dB, dC, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return ms / iterations;
}

int main() {
    printf("=== Day 4: Row-Tiled GEMM ===\n\n");
    
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;
    
    printf("Matrix dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    printf("Operations: %.2f GFLOP\n\n", 2.0 * M * N * K / 1e9);
    
    // Allocate host memory
    float *hA = new float[M * K];
    float *hB = new float[K * N];
    float *hC = new float[M * N];
    float *hRef = new float[M * N];
    
    srand(42);
    initMatrix(hA, M * K);
    initMatrix(hB, K * N);
    
    // Allocate device memory
    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC, M * N * sizeof(float)));
    
    CHECK_CUDA(cudaMemcpy(dA, hA, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, K * N * sizeof(float), cudaMemcpyHostToDevice));
    
    // cuBLAS reference
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float cublasMs;
    cudaEventElapsedTime(&cublasMs, start, stop);
    
    CHECK_CUDA(cudaMemcpy(hRef, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    float cublasTflops = 2.0f * M * N * K / (cublasMs * 1e9);
    printf("cuBLAS:         %7.3f ms  (%6.2f TFLOPS) [reference]\n", cublasMs, cublasTflops);
    
    // Naive GEMM
    auto launchNaive = [](const float* A, const float* B, float* C, int M, int N, int K) {
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (M + 15) / 16);
        naiveGemm<<<grid, block>>>(A, B, C, M, N, K);
    };
    
    float naiveMs = benchmarkKernel(launchNaive, dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaMemcpy(hC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    float maxErr = 0;
    for (int i = 0; i < M * N; i++) {
        maxErr = fmaxf(maxErr, fabsf(hC[i] - hRef[i]));
    }
    float naiveTflops = 2.0f * M * N * K / (naiveMs * 1e9);
    printf("Naive GEMM:     %7.3f ms  (%6.2f TFLOPS)  error: %.2e\n", naiveMs, naiveTflops, maxErr);
    
    // Row-Tiled GEMM
    constexpr int TILE_M = 16;
    size_t sharedSize = TILE_M * K * sizeof(float);
    
    auto launchRowTiled = [&](const float* A, const float* B, float* C, int M, int N, int K) {
        dim3 block(16, TILE_M);  // 16 threads per row, TILE_M rows
        dim3 grid((N + 15) / 16, (M + TILE_M - 1) / TILE_M);
        rowTiledGemm<TILE_M, 16><<<grid, block, sharedSize>>>(A, B, C, M, N, K);
    };
    
    float rowTiledMs = benchmarkKernel(launchRowTiled, dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaMemcpy(hC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    maxErr = 0;
    for (int i = 0; i < M * N; i++) {
        maxErr = fmaxf(maxErr, fabsf(hC[i] - hRef[i]));
    }
    float rowTiledTflops = 2.0f * M * N * K / (rowTiledMs * 1e9);
    printf("Row-Tiled:      %7.3f ms  (%6.2f TFLOPS)  error: %.2e\n", rowTiledMs, rowTiledTflops, maxErr);
    
    // Row-Tiled Unrolled
    auto launchRowTiledUnroll = [&](const float* A, const float* B, float* C, int M, int N, int K) {
        dim3 block(16, TILE_M);
        dim3 grid((N + 15) / 16, (M + TILE_M - 1) / TILE_M);
        rowTiledGemmUnroll<TILE_M><<<grid, block, sharedSize>>>(A, B, C, M, N, K);
    };
    
    float rowUnrollMs = benchmarkKernel(launchRowTiledUnroll, dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaMemcpy(hC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    maxErr = 0;
    for (int i = 0; i < M * N; i++) {
        maxErr = fmaxf(maxErr, fabsf(hC[i] - hRef[i]));
    }
    float rowUnrollTflops = 2.0f * M * N * K / (rowUnrollMs * 1e9);
    printf("Row-Tiled+Unroll: %5.3f ms  (%6.2f TFLOPS)  error: %.2e\n", rowUnrollMs, rowUnrollTflops, maxErr);
    
    // Analysis
    printf("\n=== Analysis ===\n");
    printf("Shared memory per block: %.1f KB\n", sharedSize / 1024.0f);
    printf("\nSpeedup over naive:\n");
    printf("  Row-Tiled:        %.2fx\n", naiveMs / rowTiledMs);
    printf("  Row-Tiled+Unroll: %.2fx\n", naiveMs / rowUnrollMs);
    printf("\n%% of cuBLAS:\n");
    printf("  Naive:            %.1f%%\n", 100.0f * cublasMs / naiveMs);
    printf("  Row-Tiled:        %.1f%%\n", 100.0f * cublasMs / rowTiledMs);
    printf("  Row-Tiled+Unroll: %.1f%%\n", 100.0f * cublasMs / rowUnrollMs);
    
    printf("\n=== Key Insight ===\n");
    printf("Row tiling helps A access but B is still read M*K times from global!\n");
    printf("Need to also tile B for significant improvement.\n");
    printf("This motivates the 2D tiled approach (Week 22).\n");
    
    // Cleanup
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
