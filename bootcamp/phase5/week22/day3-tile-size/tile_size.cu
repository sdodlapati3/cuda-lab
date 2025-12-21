/**
 * Day 3: Tile Size Selection
 * 
 * Comprehensive study of how tile size affects GEMM performance.
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

// Template for different tile sizes
template<int TILE_M, int TILE_N, int TILE_K>
__global__ void tiledGemm(const float* __restrict__ A,
                          const float* __restrict__ B,
                          float* __restrict__ C,
                          int M, int N, int K) {
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];
    
    int row = blockIdx.y * TILE_M + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int t = 0; t < K; t += TILE_K) {
        // Load A tile
        if (threadIdx.x < TILE_K && threadIdx.y < TILE_M) {
            int aCol = t + threadIdx.x;
            As[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        }
        
        // Load B tile
        if (threadIdx.y < TILE_K && threadIdx.x < TILE_N) {
            int bRow = t + threadIdx.y;
            Bs[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Specialized square tile version
template<int TILE>
__global__ void tiledGemmSquare(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C,
                                 int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    
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

void initMatrix(float* mat, int size) {
    for (int i = 0; i < size; i++) {
        mat[i] = (rand() / (float)RAND_MAX - 0.5f);
    }
}

template<int TILE>
float benchmarkTile(const float* dA, const float* dB, float* dC,
                    int M, int N, int K, int iterations) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    
    // Warmup
    tiledGemmSquare<TILE><<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaDeviceSynchronize();
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        tiledGemmSquare<TILE><<<grid, block>>>(dA, dB, dC, M, N, K);
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
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║             Day 3: Tile Size Selection Study                 ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    // Get device info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("Shared memory per SM: %zu KB\n", prop.sharedMemPerMultiprocessor / 1024);
    printf("Max threads per block: %d\n\n", prop.maxThreadsPerBlock);
    
    int iterations = 20;
    
    // Test different matrix sizes
    int sizes[] = {512, 1024, 2048, 4096};
    
    for (int size : sizes) {
        int M = size, N = size, K = size;
        double gflops = 2.0 * M * N * K / 1e9;
        
        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        printf("Matrix Size: %d × %d × %d  (%.2f GFLOP)\n", M, N, K, gflops);
        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        
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
        
        printf("\n┌──────────┬──────────┬──────────┬──────────┬──────────────┐\n");
        printf("│ Tile     │ Time(ms) │ TFLOPS   │ %%cuBLAS  │ SharedMem/Blk│\n");
        printf("├──────────┼──────────┼──────────┼──────────┼──────────────┤\n");
        
        printf("│ cuBLAS   │ %8.3f │ %8.2f │   100.0%% │     N/A      │\n",
               cublasMs, gflops / cublasMs);
        
        // Test tile 8
        float ms8 = benchmarkTile<8>(dA, dB, dC, M, N, K, iterations);
        printf("│   8×8    │ %8.3f │ %8.2f │ %7.1f%% │ %5d bytes  │\n",
               ms8, gflops / ms8, 100.0f * cublasMs / ms8, 2 * 8 * 8 * 4);
        
        // Test tile 16
        float ms16 = benchmarkTile<16>(dA, dB, dC, M, N, K, iterations);
        printf("│  16×16   │ %8.3f │ %8.2f │ %7.1f%% │ %5d bytes  │\n",
               ms16, gflops / ms16, 100.0f * cublasMs / ms16, 2 * 16 * 16 * 4);
        
        // Test tile 32
        float ms32 = benchmarkTile<32>(dA, dB, dC, M, N, K, iterations);
        printf("│  32×32   │ %8.3f │ %8.2f │ %7.1f%% │ %5d bytes  │\n",
               ms32, gflops / ms32, 100.0f * cublasMs / ms32, 2 * 32 * 32 * 4);
        
        printf("└──────────┴──────────┴──────────┴──────────┴──────────────┘\n");
        
        // Find best
        float bestMs = fminf(ms8, fminf(ms16, ms32));
        const char* bestTile = (bestMs == ms8) ? "8×8" : ((bestMs == ms16) ? "16×16" : "32×32");
        printf("\nBest tile size: %s (%.3f ms, %.2f TFLOPS)\n\n", 
               bestTile, bestMs, gflops / bestMs);
        
        cublasDestroy(handle);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
        delete[] hA;
        delete[] hB;
    }
    
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║                    Key Findings                              ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║ 1. 32×32 tiles often best for large matrices                 ║\n");
    printf("║ 2. Smaller tiles better for small matrices (occupancy)       ║\n");
    printf("║ 3. Optimal tile size depends on matrix dimensions            ║\n");
    printf("║ 4. Balance between data reuse and occupancy is key           ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    
    return 0;
}
