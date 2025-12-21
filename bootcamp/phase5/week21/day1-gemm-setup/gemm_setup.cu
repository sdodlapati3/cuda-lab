/**
 * GEMM Problem Setup and Analysis
 * 
 * This file establishes the foundation for GEMM optimization:
 * - Matrix dimensions and memory layout
 * - FLOP counting and arithmetic intensity
 * - cuBLAS baseline for comparison
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

// Default dimensions for GEMM
#define DEFAULT_M 4096
#define DEFAULT_N 4096
#define DEFAULT_K 4096

// Error checking macros
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
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
        exit(1); \
    } \
} while(0)

// Initialize matrix with random values
void initMatrix(float* mat, int rows, int cols, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;  // Range [-1, 1]
    }
}

// Initialize matrix with zeros
void zeroMatrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = 0.0f;
    }
}

// Calculate theoretical metrics
void analyzeGEMM(int M, int N, int K) {
    printf("=== GEMM Analysis ===\n");
    printf("Dimensions: M=%d, N=%d, K=%d\n\n", M, N, K);
    
    // FLOP count
    double flops = 2.0 * M * N * K;
    printf("Compute:\n");
    printf("  Total FLOPs: %.2e (%.2f GFLOP)\n", flops, flops / 1e9);
    
    // Memory traffic (FP32)
    double bytesA = (double)M * K * sizeof(float);
    double bytesB = (double)K * N * sizeof(float);
    double bytesC = (double)M * N * sizeof(float);
    double totalBytes = bytesA + bytesB + 2 * bytesC;  // Read and write C
    
    printf("\nMemory:\n");
    printf("  Matrix A: %.2f MB\n", bytesA / 1e6);
    printf("  Matrix B: %.2f MB\n", bytesB / 1e6);
    printf("  Matrix C: %.2f MB\n", bytesC / 1e6);
    printf("  Total traffic: %.2f MB\n", totalBytes / 1e6);
    
    // Arithmetic intensity
    double ai = flops / totalBytes;
    printf("\nArithmetic Intensity: %.2f FLOPs/byte\n", ai);
    
    // A100 roofline analysis
    double peakFlops = 19.5e12;  // FP32 TFLOPS
    double peakBW = 2039e9;      // GB/s
    double ridgePoint = peakFlops / peakBW;
    
    printf("\nRoofline Analysis (A100):\n");
    printf("  Peak FP32: %.1f TFLOPS\n", peakFlops / 1e12);
    printf("  Peak BW: %.0f GB/s\n", peakBW / 1e9);
    printf("  Ridge point: %.1f FLOPs/byte\n", ridgePoint);
    
    double achievableFlops;
    if (ai < ridgePoint) {
        achievableFlops = peakBW * ai;
        printf("  Status: MEMORY-BOUND (AI < ridge point)\n");
    } else {
        achievableFlops = peakFlops;
        printf("  Status: COMPUTE-BOUND (AI > ridge point)\n");
    }
    printf("  Max achievable: %.2f TFLOPS\n", achievableFlops / 1e12);
    printf("\n");
}

// Run cuBLAS GEMM for baseline
float benchmarkCuBLAS(cublasHandle_t handle, 
                       float* d_A, float* d_B, float* d_C,
                       int M, int N, int K,
                       int iterations) {
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Warm-up
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              N, M, K,  // Note: cuBLAS uses column-major
                              &alpha,
                              d_B, N,
                              d_A, K,
                              &beta,
                              d_C, N));
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Benchmark
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  N, M, K,
                                  &alpha,
                                  d_B, N,
                                  d_A, K,
                                  &beta,
                                  d_C, N));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float totalMs;
    CHECK_CUDA(cudaEventElapsedTime(&totalMs, start, stop));
    
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    return totalMs / iterations;
}

int main(int argc, char** argv) {
    // Parse dimensions
    int M = (argc > 1) ? atoi(argv[1]) : DEFAULT_M;
    int N = (argc > 2) ? atoi(argv[2]) : DEFAULT_N;
    int K = (argc > 3) ? atoi(argv[3]) : DEFAULT_K;
    
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║            GEMM Problem Setup & Analysis                 ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");
    
    // Print device info
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s\n", prop.name);
    printf("Compute: %d.%d, SMs: %d, Memory: %.1f GB\n\n",
           prop.major, prop.minor, prop.multiProcessorCount,
           prop.totalGlobalMem / 1e9);
    
    // Analyze GEMM problem
    analyzeGEMM(M, N, K);
    
    // Allocate matrices
    size_t sizeA = (size_t)M * K * sizeof(float);
    size_t sizeB = (size_t)K * N * sizeof(float);
    size_t sizeC = (size_t)M * N * sizeof(float);
    
    printf("Allocating matrices...\n");
    float* h_A = (float*)malloc(sizeA);
    float* h_B = (float*)malloc(sizeB);
    float* h_C = (float*)malloc(sizeC);
    
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, sizeA));
    CHECK_CUDA(cudaMalloc(&d_B, sizeB));
    CHECK_CUDA(cudaMalloc(&d_C, sizeC));
    
    // Initialize
    printf("Initializing matrices...\n");
    initMatrix(h_A, M, K, 42);
    initMatrix(h_B, K, N, 43);
    zeroMatrix(h_C, M, N);
    
    CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C, sizeC, cudaMemcpyHostToDevice));
    
    // cuBLAS baseline
    printf("\n=== cuBLAS Baseline ===\n");
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    
    int iterations = 10;
    float cuBlasMs = benchmarkCuBLAS(handle, d_A, d_B, d_C, M, N, K, iterations);
    
    double flops = 2.0 * M * N * K;
    double tflops = (flops / cuBlasMs) / 1e9;  // ms to s, then to TFLOPS
    
    printf("cuBLAS time: %.3f ms\n", cuBlasMs);
    printf("cuBLAS TFLOPS: %.2f\n", tflops);
    printf("\nThis is your optimization target!\n");
    
    // Cleanup
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    
    printf("\n=== Next Steps ===\n");
    printf("1. Implement naive GEMM kernel (Day 2)\n");
    printf("2. Analyze memory access patterns (Day 3)\n");
    printf("3. Compare your kernel against cuBLAS baseline\n");
    
    return 0;
}
