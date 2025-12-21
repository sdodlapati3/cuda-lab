/**
 * Day 5: Double Buffering
 * 
 * Demonstrates overlapping memory loads with computation
 * using double buffering technique.
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

// Single-buffered tiled GEMM (baseline)
template<int TILE>
__global__ void tiledGemmSingle(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C,
                                 int M, int N, int K) {
    __shared__ float As[TILE][TILE + 1];
    __shared__ float Bs[TILE][TILE + 1];
    
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int t = 0; t < K; t += TILE) {
        int aCol = t + threadIdx.x;
        int bRow = t + threadIdx.y;
        
        As[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Double-buffered tiled GEMM
template<int TILE>
__global__ void tiledGemmDouble(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C,
                                 int M, int N, int K) {
    // Double buffers
    __shared__ float As[2][TILE][TILE + 1];
    __shared__ float Bs[2][TILE][TILE + 1];
    
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    
    float sum = 0.0f;
    
    int numTiles = (K + TILE - 1) / TILE;
    
    // Load first tile into buffer 0
    {
        int aCol = threadIdx.x;
        int bRow = threadIdx.y;
        As[0][threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        Bs[0][threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
    }
    __syncthreads();
    
    // Main loop: load next tile while computing current
    for (int t = 1; t < numTiles; t++) {
        int currBuf = (t - 1) % 2;
        int nextBuf = t % 2;
        
        // Load next tile into nextBuf
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;
        As[nextBuf][threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        Bs[nextBuf][threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        
        // Compute from currBuf (overlapped with load)
        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            sum += As[currBuf][threadIdx.y][k] * Bs[currBuf][k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Compute last tile
    int lastBuf = (numTiles - 1) % 2;
    #pragma unroll
    for (int k = 0; k < TILE; k++) {
        sum += As[lastBuf][threadIdx.y][k] * Bs[lastBuf][k][threadIdx.x];
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Double-buffered with async copy (requires sm_80+)
template<int TILE>
__global__ void tiledGemmAsync(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int M, int N, int K) {
    __shared__ float As[2][TILE][TILE + 1];
    __shared__ float Bs[2][TILE][TILE + 1];
    
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    
    float sum = 0.0f;
    int numTiles = (K + TILE - 1) / TILE;
    
    // Async load first tile
    {
        int aCol = threadIdx.x;
        int bRow = threadIdx.y;
        
        if (row < M && aCol < K) {
            __pipeline_memcpy_async(&As[0][threadIdx.y][threadIdx.x],
                                    &A[row * K + aCol], sizeof(float));
        } else {
            As[0][threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (bRow < K && col < N) {
            __pipeline_memcpy_async(&Bs[0][threadIdx.y][threadIdx.x],
                                    &B[bRow * N + col], sizeof(float));
        } else {
            Bs[0][threadIdx.y][threadIdx.x] = 0.0f;
        }
        __pipeline_commit();
    }
    
    // Main loop
    for (int t = 1; t < numTiles; t++) {
        int currBuf = (t - 1) % 2;
        int nextBuf = t % 2;
        
        // Initiate async load of next tile
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;
        
        if (row < M && aCol < K) {
            __pipeline_memcpy_async(&As[nextBuf][threadIdx.y][threadIdx.x],
                                    &A[row * K + aCol], sizeof(float));
        } else {
            As[nextBuf][threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (bRow < K && col < N) {
            __pipeline_memcpy_async(&Bs[nextBuf][threadIdx.y][threadIdx.x],
                                    &B[bRow * N + col], sizeof(float));
        } else {
            Bs[nextBuf][threadIdx.y][threadIdx.x] = 0.0f;
        }
        __pipeline_commit();
        
        // Wait for current buffer to be ready
        __pipeline_wait_prior(1);
        __syncthreads();
        
        // Compute from current buffer
        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            sum += As[currBuf][threadIdx.y][k] * Bs[currBuf][k][threadIdx.x];
        }
    }
    
    // Wait for and compute last tile
    __pipeline_wait_prior(0);
    __syncthreads();
    
    int lastBuf = (numTiles - 1) % 2;
    #pragma unroll
    for (int k = 0; k < TILE; k++) {
        sum += As[lastBuf][threadIdx.y][k] * Bs[lastBuf][k][threadIdx.x];
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void initMatrix(float* mat, int size) {
    for (int i = 0; i < size; i++) {
        mat[i] = (rand() / (float)RAND_MAX - 0.5f);
    }
}

int main() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║              Day 5: Double Buffering                         ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    const int M = 4096;
    const int N = 4096;
    const int K = 4096;
    constexpr int TILE = 32;
    
    printf("Matrix: %d × %d × %d, Tile: %d × %d\n\n", M, N, K, TILE, TILE);
    double gflops = 2.0 * M * N * K / 1e9;
    
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
    
    int iterations = 10;
    
    // cuBLAS reference
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;
    
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
    
    printf("Performance Comparison:\n");
    printf("┌────────────────────────┬──────────┬──────────┬──────────┐\n");
    printf("│ Implementation         │ Time(ms) │ TFLOPS   │ %%cuBLAS  │\n");
    printf("├────────────────────────┼──────────┼──────────┼──────────┤\n");
    printf("│ cuBLAS                 │ %8.3f │ %8.2f │   100.0%% │\n",
           cublasMs, gflops / cublasMs);
    
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    
    // Single buffered
    tiledGemmSingle<TILE><<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        tiledGemmSingle<TILE><<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float singleMs;
    cudaEventElapsedTime(&singleMs, start, stop);
    singleMs /= iterations;
    
    printf("│ Single Buffer          │ %8.3f │ %8.2f │ %7.1f%% │\n",
           singleMs, gflops / singleMs, 100.0f * cublasMs / singleMs);
    
    // Double buffered
    tiledGemmDouble<TILE><<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        tiledGemmDouble<TILE><<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float doubleMs;
    cudaEventElapsedTime(&doubleMs, start, stop);
    doubleMs /= iterations;
    
    CHECK_CUDA(cudaMemcpy(hC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    float maxErr = 0;
    for (int i = 0; i < M * N; i++) maxErr = fmaxf(maxErr, fabsf(hC[i] - hRef[i]));
    
    printf("│ Double Buffer          │ %8.3f │ %8.2f │ %7.1f%% │\n",
           doubleMs, gflops / doubleMs, 100.0f * cublasMs / doubleMs);
    
    // Async double buffered
    tiledGemmAsync<TILE><<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        tiledGemmAsync<TILE><<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float asyncMs;
    cudaEventElapsedTime(&asyncMs, start, stop);
    asyncMs /= iterations;
    
    printf("│ Double + Async         │ %8.3f │ %8.2f │ %7.1f%% │\n",
           asyncMs, gflops / asyncMs, 100.0f * cublasMs / asyncMs);
    
    printf("└────────────────────────┴──────────┴──────────┴──────────┘\n\n");
    
    printf("Speedup from double buffering: %.1f%%\n", 
           100.0f * (singleMs - doubleMs) / singleMs);
    printf("Max error: %.2e\n", maxErr);
    
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║                    Key Takeaways                             ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║ 1. Double buffering overlaps memory loads with compute       ║\n");
    printf("║ 2. Async copy (cp.async) further improves latency hiding     ║\n");
    printf("║ 3. Trade-off: 2× shared memory usage                         ║\n");
    printf("║ 4. Most beneficial for memory-bound kernels                  ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    
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
