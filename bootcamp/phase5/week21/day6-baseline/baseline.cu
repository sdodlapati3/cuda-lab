/**
 * Day 6: Week 21 Performance Baseline
 * 
 * Comprehensive comparison of all Week 21 implementations.
 * Establishes baseline for measuring progress in Weeks 22-28.
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// === KERNEL IMPLEMENTATIONS ===

// 1. Naive GEMM
__global__ void naiveGemm(const float* A, const float* B, float* C, int M, int N, int K) {
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

// 2. Naive Unrolled
__global__ void naiveGemmUnroll(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        int k = 0;
        for (; k + 3 < K; k += 4) {
            sum += A[row * K + k]     * B[(k)     * N + col];
            sum += A[row * K + k + 1] * B[(k + 1) * N + col];
            sum += A[row * K + k + 2] * B[(k + 2) * N + col];
            sum += A[row * K + k + 3] * B[(k + 3) * N + col];
        }
        for (; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// 3. Row-Tiled GEMM
template<int TILE_M>
__global__ void rowTiledGemm(const float* A, const float* B, float* C, int M, int N, int K) {
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
        for (int k = 0; k < K; k++) {
            sum += rowA[k] * B[k * N + col];
        }
        C[globalRow * N + col] = sum;
    }
}

// 4. Column-Tiled GEMM
template<int TILE_N>
__global__ void colTiledGemm(const float* A, const float* B, float* C, int M, int N, int K) {
    extern __shared__ float sharedB[];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int colStart = blockIdx.x * TILE_N;
    int localCol = threadIdx.x;
    int col = colStart + localCol;
    
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int numThreads = blockDim.x * blockDim.y;
    
    // Load tile of B
    int totalElements = K * TILE_N;
    for (int i = tid; i < totalElements; i += numThreads) {
        int kIdx = i / TILE_N;
        int localColIdx = i % TILE_N;
        int globalCol = colStart + localColIdx;
        sharedB[kIdx * TILE_N + localColIdx] = (globalCol < N) ? B[kIdx * N + globalCol] : 0.0f;
    }
    __syncthreads();
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * sharedB[k * TILE_N + localCol];
        }
        C[row * N + col] = sum;
    }
}

// === BENCHMARK INFRASTRUCTURE ===

struct BenchmarkResult {
    std::string name;
    float timeMs;
    float tflops;
    float percentCublas;
    float maxError;
};

void initMatrix(float* mat, int size) {
    for (int i = 0; i < size; i++) {
        mat[i] = (rand() / (float)RAND_MAX - 0.5f);
    }
}

float computeMaxError(const float* a, const float* b, int size) {
    float maxErr = 0;
    for (int i = 0; i < size; i++) {
        maxErr = fmaxf(maxErr, fabsf(a[i] - b[i]));
    }
    return maxErr;
}

int main() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║         Week 21: GEMM Performance Baseline                   ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    // Get device info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    float peakTflops = 2.0f * prop.clockRate * 1e3 * prop.multiProcessorCount * 64 / 1e12;
    printf("Estimated Peak FP32: %.1f TFLOPS (simplified estimate)\n\n", peakTflops);
    
    // Test multiple sizes
    std::vector<int> sizes = {512, 1024, 2048, 4096};
    
    for (int size : sizes) {
        int M = size, N = size, K = size;
        double gflops = 2.0 * M * N * K / 1e9;
        
        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        printf("Matrix Size: %d × %d × %d  (%.2f GFLOP)\n", M, N, K, gflops);
        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        
        // Allocate
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
        
        std::vector<BenchmarkResult> results;
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        int iterations = 10;
        
        // cuBLAS Reference
        cublasHandle_t handle;
        cublasCreate(&handle);
        float alpha = 1.0f, beta = 0.0f;
        
        // Warmup
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);
        CHECK_CUDA(cudaDeviceSynchronize());
        
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
        float cublasTflops = gflops / cublasMs;
        
        results.push_back({"cuBLAS (reference)", cublasMs, cublasTflops, 100.0f, 0.0f});
        
        // Naive GEMM
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (M + 15) / 16);
        
        naiveGemm<<<grid, block>>>(dA, dB, dC, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());
        
        cudaEventRecord(start);
        for (int i = 0; i < iterations; i++) {
            naiveGemm<<<grid, block>>>(dA, dB, dC, M, N, K);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float naiveMs;
        cudaEventElapsedTime(&naiveMs, start, stop);
        naiveMs /= iterations;
        
        CHECK_CUDA(cudaMemcpy(hC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));
        float naiveErr = computeMaxError(hC, hRef, M * N);
        float naiveTflops = gflops / naiveMs;
        
        results.push_back({"Naive GEMM", naiveMs, naiveTflops, 100.0f * cublasMs / naiveMs, naiveErr});
        
        // Naive Unrolled
        naiveGemmUnroll<<<grid, block>>>(dA, dB, dC, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());
        
        cudaEventRecord(start);
        for (int i = 0; i < iterations; i++) {
            naiveGemmUnroll<<<grid, block>>>(dA, dB, dC, M, N, K);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float unrollMs;
        cudaEventElapsedTime(&unrollMs, start, stop);
        unrollMs /= iterations;
        
        CHECK_CUDA(cudaMemcpy(hC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));
        float unrollErr = computeMaxError(hC, hRef, M * N);
        float unrollTflops = gflops / unrollMs;
        
        results.push_back({"Naive Unrolled", unrollMs, unrollTflops, 100.0f * cublasMs / unrollMs, unrollErr});
        
        // Row-Tiled (only for smaller sizes due to shared memory)
        if (K <= 1024) {
            constexpr int TILE_M = 16;
            size_t sharedSize = TILE_M * K * sizeof(float);
            dim3 blockRow(16, TILE_M);
            dim3 gridRow((N + 15) / 16, (M + TILE_M - 1) / TILE_M);
            
            rowTiledGemm<TILE_M><<<gridRow, blockRow, sharedSize>>>(dA, dB, dC, M, N, K);
            CHECK_CUDA(cudaDeviceSynchronize());
            
            cudaEventRecord(start);
            for (int i = 0; i < iterations; i++) {
                rowTiledGemm<TILE_M><<<gridRow, blockRow, sharedSize>>>(dA, dB, dC, M, N, K);
            }
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            
            float rowMs;
            cudaEventElapsedTime(&rowMs, start, stop);
            rowMs /= iterations;
            
            CHECK_CUDA(cudaMemcpy(hC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));
            float rowErr = computeMaxError(hC, hRef, M * N);
            float rowTflops = gflops / rowMs;
            
            results.push_back({"Row-Tiled", rowMs, rowTflops, 100.0f * cublasMs / rowMs, rowErr});
            
            // Col-Tiled
            constexpr int TILE_N = 16;
            sharedSize = K * TILE_N * sizeof(float);
            dim3 blockCol(TILE_N, 16);
            dim3 gridCol((N + TILE_N - 1) / TILE_N, (M + 15) / 16);
            
            colTiledGemm<TILE_N><<<gridCol, blockCol, sharedSize>>>(dA, dB, dC, M, N, K);
            CHECK_CUDA(cudaDeviceSynchronize());
            
            cudaEventRecord(start);
            for (int i = 0; i < iterations; i++) {
                colTiledGemm<TILE_N><<<gridCol, blockCol, sharedSize>>>(dA, dB, dC, M, N, K);
            }
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            
            float colMs;
            cudaEventElapsedTime(&colMs, start, stop);
            colMs /= iterations;
            
            CHECK_CUDA(cudaMemcpy(hC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));
            float colErr = computeMaxError(hC, hRef, M * N);
            float colTflops = gflops / colMs;
            
            results.push_back({"Col-Tiled", colMs, colTflops, 100.0f * cublasMs / colMs, colErr});
        }
        
        // Print results table
        printf("┌────────────────────┬──────────┬──────────┬──────────┬──────────┐\n");
        printf("│ Implementation     │ Time(ms) │ TFLOPS   │ %%cuBLAS  │ MaxError │\n");
        printf("├────────────────────┼──────────┼──────────┼──────────┼──────────┤\n");
        
        for (const auto& r : results) {
            printf("│ %-18s │ %8.3f │ %8.2f │ %7.1f%% │ %.2e │\n",
                   r.name.c_str(), r.timeMs, r.tflops, r.percentCublas, r.maxError);
        }
        printf("└────────────────────┴──────────┴──────────┴──────────┴──────────┘\n\n");
        
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
    }
    
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║                    Week 21 Summary                           ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║ Key Observations:                                            ║\n");
    printf("║ 1. Naive GEMM achieves only 5-10%% of cuBLAS performance      ║\n");
    printf("║ 2. 1D tiling provides modest improvement                     ║\n");
    printf("║ 3. Memory bandwidth is the primary bottleneck                ║\n");
    printf("║ 4. Need 2D tiling for significant gains (Week 22)            ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    
    return 0;
}
