/**
 * Memory Access Analysis for GEMM
 * 
 * Demonstrates different memory layouts and their impact:
 * - Row-major A × Row-major B
 * - Row-major A × Column-major B (transposed)
 * - Column-major A × Row-major B
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// Naive GEMM: Row-major A, Row-major B
// A access: strided (bad), B access: coalesced (good)
__global__ void gemm_RR(const float* A, const float* B, float* C,
                         int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            // A[row, k] - threads in warp access different rows (strided)
            // B[k, col] - threads in warp access consecutive columns (coalesced)
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// GEMM: Row-major A, Column-major B (B transposed in memory)
// A access: strided (bad), B access: strided (bad!)
__global__ void gemm_RC(const float* A, const float* B_col, float* C,
                         int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            // A[row, k] - strided
            // B_col[k, col] = B_col[col * K + k] - also strided!
            sum += A[row * K + k] * B_col[col * K + k];
        }
        C[row * N + col] = sum;
    }
}

// GEMM: Column-major A, Row-major B
// A access: coalesced (good!), B access: coalesced (good!)
__global__ void gemm_CR(const float* A_col, const float* B, float* C,
                         int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            // A_col[row, k] = A_col[k * M + row] - threads access consecutive rows (coalesced!)
            // B[k, col] - threads access consecutive columns (coalesced)
            sum += A_col[k * M + row] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Transpose matrix on GPU
__global__ void transpose(const float* in, float* out, int rows, int cols) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < rows && j < cols) {
        out[j * rows + i] = in[i * cols + j];
    }
}

void initMatrix(float* mat, int size, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < size; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

float benchmark(void (*kernel)(const float*, const float*, float*, int, int, int),
                 const float* d_A, const float* d_B, float* d_C,
                 int M, int N, int K, int iterations) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    
    // Warm-up
    kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return ms / iterations;
}

int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 2048;
    int N = (argc > 2) ? atoi(argv[2]) : 2048;
    int K = (argc > 3) ? atoi(argv[3]) : 2048;
    
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║            Memory Access Pattern Analysis                ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");
    
    printf("Dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    double flops = 2.0 * M * N * K;
    printf("FLOPs: %.2f GFLOP\n\n", flops / 1e9);
    
    // Allocate
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);
    
    float* h_A = (float*)malloc(sizeA);
    float* h_B = (float*)malloc(sizeB);
    
    initMatrix(h_A, M * K, 42);
    initMatrix(h_B, K * N, 43);
    
    float *d_A, *d_B, *d_C;
    float *d_A_col, *d_B_col;  // Transposed versions
    
    CHECK_CUDA(cudaMalloc(&d_A, sizeA));
    CHECK_CUDA(cudaMalloc(&d_B, sizeB));
    CHECK_CUDA(cudaMalloc(&d_C, sizeC));
    CHECK_CUDA(cudaMalloc(&d_A_col, sizeA));
    CHECK_CUDA(cudaMalloc(&d_B_col, sizeB));
    
    CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));
    
    // Create transposed versions
    dim3 tBlock(16, 16);
    dim3 tGridA((K + 15) / 16, (M + 15) / 16);
    dim3 tGridB((N + 15) / 16, (K + 15) / 16);
    
    transpose<<<tGridA, tBlock>>>(d_A, d_A_col, M, K);
    transpose<<<tGridB, tBlock>>>(d_B, d_B_col, K, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    int iterations = 10;
    
    printf("=== Memory Access Pattern Comparison ===\n\n");
    printf("Layout    | A Access  | B Access  | Time (ms) | TFLOPS | Note\n");
    printf("----------|-----------|-----------|-----------|--------|------------------\n");
    
    // Test RR (standard row-major)
    float ms_RR = benchmark(gemm_RR, d_A, d_B, d_C, M, N, K, iterations);
    double tflops_RR = (flops / ms_RR) / 1e9;
    printf("A_row×B_row | Strided  | Coalesced | %9.3f | %6.2f | Baseline\n", 
           ms_RR, tflops_RR);
    
    // Test RC (B column-major)
    float ms_RC = benchmark(gemm_RC, d_A, d_B_col, d_C, M, N, K, iterations);
    double tflops_RC = (flops / ms_RC) / 1e9;
    printf("A_row×B_col | Strided  | Strided   | %9.3f | %6.2f | Worst case\n",
           ms_RC, tflops_RC);
    
    // Test CR (A column-major)
    float ms_CR = benchmark(gemm_CR, d_A_col, d_B, d_C, M, N, K, iterations);
    double tflops_CR = (flops / ms_CR) / 1e9;
    printf("A_col×B_row | Coalesced| Coalesced | %9.3f | %6.2f | Best naive\n",
           ms_CR, tflops_CR);
    
    printf("\n=== Analysis ===\n");
    printf("Speedup (A_col×B_row vs A_row×B_row): %.2fx\n", ms_RR / ms_CR);
    printf("Speedup (A_col×B_row vs A_row×B_col): %.2fx\n", ms_RC / ms_CR);
    
    printf("\n=== Key Insights ===\n");
    printf("1. Memory coalescing has HUGE impact on performance\n");
    printf("2. Column-major A enables coalesced access\n");
    printf("3. But we still read each element M×N times!\n");
    printf("4. Next step: Use shared memory to cache data (Week 22)\n");
    
    // Calculate theoretical bandwidth
    double bytesRead = 2.0 * M * N * K * sizeof(float);  // Naive reads
    double achievedBW_CR = bytesRead / (ms_CR / 1000.0) / 1e9;  // GB/s
    printf("\n=== Memory Bandwidth ===\n");
    printf("Total bytes read (naive): %.2f GB\n", bytesRead / 1e9);
    printf("Achieved bandwidth (A_col×B_row): %.0f GB/s\n", achievedBW_CR);
    
    // Cleanup
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_A_col));
    CHECK_CUDA(cudaFree(d_B_col));
    free(h_A);
    free(h_B);
    
    return 0;
}
