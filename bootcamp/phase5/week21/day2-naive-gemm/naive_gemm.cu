/**
 * Naive GEMM Implementation
 * 
 * One thread computes one element of the output matrix.
 * This is intentionally inefficient to establish a baseline.
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d\n", __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// ============================================================================
// Naive GEMM Kernel
// ============================================================================

/**
 * Naive GEMM: One thread per output element
 * 
 * C[M,N] = A[M,K] × B[K,N]
 * 
 * Each thread:
 * - Reads one full row of A (K elements)
 * - Reads one full column of B (K elements)
 * - Computes one element of C
 */
__global__ void naiveGemm(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C,
                           int M, int N, int K) {
    // 2D thread mapping to output matrix
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        
        // Dot product of row of A with column of B
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        
        C[row * N + col] = sum;
    }
}

// Slightly optimized naive: unroll inner loop
__global__ void naiveGemmUnroll(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C,
                                 int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        
        int k = 0;
        // Unroll by 4
        for (; k + 3 < K; k += 4) {
            sum += A[row * K + k] * B[k * N + col];
            sum += A[row * K + k + 1] * B[(k + 1) * N + col];
            sum += A[row * K + k + 2] * B[(k + 2) * N + col];
            sum += A[row * K + k + 3] * B[(k + 3) * N + col];
        }
        // Handle remainder
        for (; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        
        C[row * N + col] = sum;
    }
}

// ============================================================================
// Verification and Benchmarking
// ============================================================================

void initMatrix(float* mat, int rows, int cols, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
}

bool verifyResult(float* result, float* reference, int size, float tolerance = 1e-3f) {
    int errors = 0;
    float maxError = 0.0f;
    
    for (int i = 0; i < size; i++) {
        float err = fabs(result[i] - reference[i]);
        maxError = fmaxf(maxError, err);
        
        // Relative error check
        float relErr = err / (fabs(reference[i]) + 1e-6f);
        if (relErr > tolerance) {
            if (errors < 5) {
                printf("  Mismatch at %d: got %.6f, expected %.6f (rel err: %.6f)\n",
                       i, result[i], reference[i], relErr);
            }
            errors++;
        }
    }
    
    printf("  Max absolute error: %.6e\n", maxError);
    printf("  Errors: %d / %d\n", errors, size);
    return errors == 0;
}

float benchmarkKernel(void (*launch)(const float*, const float*, float*, int, int, int, dim3, dim3),
                       const float* d_A, const float* d_B, float* d_C,
                       int M, int N, int K, dim3 grid, dim3 block,
                       int iterations) {
    // Warm-up
    launch(d_A, d_B, d_C, M, N, K, grid, block);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        launch(d_A, d_B, d_C, M, N, K, grid, block);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float totalMs;
    CHECK_CUDA(cudaEventElapsedTime(&totalMs, start, stop));
    
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    return totalMs / iterations;
}

// Kernel launch wrappers
void launchNaive(const float* A, const float* B, float* C, 
                  int M, int N, int K, dim3 grid, dim3 block) {
    naiveGemm<<<grid, block>>>(A, B, C, M, N, K);
}

void launchNaiveUnroll(const float* A, const float* B, float* C,
                        int M, int N, int K, dim3 grid, dim3 block) {
    naiveGemmUnroll<<<grid, block>>>(A, B, C, M, N, K);
}

int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 2048;
    int N = (argc > 2) ? atoi(argv[2]) : 2048;
    int K = (argc > 3) ? atoi(argv[3]) : 2048;
    
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║               Naive GEMM Implementation                  ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");
    
    printf("Matrix dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    double flops = 2.0 * M * N * K;
    printf("FLOP count: %.2f GFLOP\n\n", flops / 1e9);
    
    // Allocate memory
    size_t sizeA = (size_t)M * K * sizeof(float);
    size_t sizeB = (size_t)K * N * sizeof(float);
    size_t sizeC = (size_t)M * N * sizeof(float);
    
    float* h_A = (float*)malloc(sizeA);
    float* h_B = (float*)malloc(sizeB);
    float* h_C = (float*)malloc(sizeC);
    float* h_C_ref = (float*)malloc(sizeC);
    
    float *d_A, *d_B, *d_C, *d_C_ref;
    CHECK_CUDA(cudaMalloc(&d_A, sizeA));
    CHECK_CUDA(cudaMalloc(&d_B, sizeB));
    CHECK_CUDA(cudaMalloc(&d_C, sizeC));
    CHECK_CUDA(cudaMalloc(&d_C_ref, sizeC));
    
    // Initialize
    initMatrix(h_A, M, K, 42);
    initMatrix(h_B, K, N, 43);
    
    CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));
    
    // cuBLAS reference
    printf("=== cuBLAS Reference ===\n");
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    
    float alpha = 1.0f, beta = 0.0f;
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    // Warm-up and compute reference
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C_ref, N));
    
    int iterations = 10;
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C_ref, N));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float cuBlasMs;
    CHECK_CUDA(cudaEventElapsedTime(&cuBlasMs, start, stop));
    cuBlasMs /= iterations;
    
    double cublasTflops = (flops / cuBlasMs) / 1e9;
    printf("cuBLAS time: %.3f ms (%.2f TFLOPS)\n\n", cuBlasMs, cublasTflops);
    
    CHECK_CUDA(cudaMemcpy(h_C_ref, d_C_ref, sizeC, cudaMemcpyDeviceToHost));
    
    // Naive GEMM
    printf("=== Naive GEMM ===\n");
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    printf("Grid: (%d, %d), Block: (%d, %d)\n", grid.x, grid.y, block.x, block.y);
    
    float naiveMs = benchmarkKernel(launchNaive, d_A, d_B, d_C, M, N, K, grid, block, iterations);
    double naiveTflops = (flops / naiveMs) / 1e9;
    
    printf("Naive time: %.3f ms (%.2f TFLOPS)\n", naiveMs, naiveTflops);
    printf("Efficiency vs cuBLAS: %.1f%%\n", 100.0 * naiveTflops / cublasTflops);
    
    // Verify
    CHECK_CUDA(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));
    printf("Verification:\n");
    bool pass = verifyResult(h_C, h_C_ref, M * N);
    printf("Result: %s\n\n", pass ? "PASS" : "FAIL");
    
    // Naive with unroll
    printf("=== Naive GEMM (Unrolled) ===\n");
    float unrollMs = benchmarkKernel(launchNaiveUnroll, d_A, d_B, d_C, M, N, K, grid, block, iterations);
    double unrollTflops = (flops / unrollMs) / 1e9;
    
    printf("Unrolled time: %.3f ms (%.2f TFLOPS)\n", unrollMs, unrollTflops);
    printf("Efficiency vs cuBLAS: %.1f%%\n", 100.0 * unrollTflops / cublasTflops);
    printf("Speedup vs basic naive: %.2fx\n\n", naiveMs / unrollMs);
    
    // Summary
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║                      Summary                             ║\n");
    printf("╠══════════════════════════════════════════════════════════╣\n");
    printf("║ Implementation      Time (ms)    TFLOPS    vs cuBLAS    ║\n");
    printf("╠══════════════════════════════════════════════════════════╣\n");
    printf("║ cuBLAS             %8.3f    %7.2f      100.0%%       ║\n", cuBlasMs, cublasTflops);
    printf("║ Naive              %8.3f    %7.2f      %5.1f%%       ║\n", naiveMs, naiveTflops, 100.0*naiveTflops/cublasTflops);
    printf("║ Naive (unroll)     %8.3f    %7.2f      %5.1f%%       ║\n", unrollMs, unrollTflops, 100.0*unrollTflops/cublasTflops);
    printf("╚══════════════════════════════════════════════════════════╝\n");
    
    // Cleanup
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_C_ref));
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    
    return 0;
}
