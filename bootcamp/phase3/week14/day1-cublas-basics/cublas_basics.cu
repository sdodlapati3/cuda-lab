/**
 * cublas_basics.cu - Introduction to cuBLAS
 * 
 * Learning objectives:
 * - Set up cuBLAS handle
 * - Vector operations (AXPY, DOT)
 * - Matrix multiply (GEMM)
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cmath>

#define CHECK_CUBLAS(call) { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        printf("cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
        exit(1); \
    } \
}

int main() {
    printf("=== cuBLAS Basics Demo ===\n\n");
    
    // Create cuBLAS handle
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    
    // ========================================================================
    // Part 1: Vector Operations
    // ========================================================================
    {
        printf("1. Vector Operations\n");
        printf("─────────────────────────────────────────\n");
        
        const int N = 1000;
        float *h_x = new float[N];
        float *h_y = new float[N];
        
        for (int i = 0; i < N; i++) {
            h_x[i] = 1.0f;
            h_y[i] = 2.0f;
        }
        
        float *d_x, *d_y;
        cudaMalloc(&d_x, N * sizeof(float));
        cudaMalloc(&d_y, N * sizeof(float));
        cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice);
        
        // DOT product: result = x · y
        float dot_result;
        CHECK_CUBLAS(cublasSdot(handle, N, d_x, 1, d_y, 1, &dot_result));
        printf("   DOT(x, y) = %.0f (expected %d)\n", dot_result, N * 2);
        
        // AXPY: y = alpha * x + y
        float alpha = 3.0f;
        CHECK_CUBLAS(cublasSaxpy(handle, N, &alpha, d_x, 1, d_y, 1));
        cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
        printf("   AXPY: y = 3*x + y → y[0] = %.0f (expected 5)\n", h_y[0]);
        
        // NORM: ||x||
        float norm;
        CHECK_CUBLAS(cublasSnrm2(handle, N, d_x, 1, &norm));
        printf("   NORM(x) = %.2f (expected %.2f)\n\n", norm, sqrtf((float)N));
        
        cudaFree(d_x);
        cudaFree(d_y);
        delete[] h_x;
        delete[] h_y;
    }
    
    // ========================================================================
    // Part 2: Matrix Multiply (GEMM)
    // ========================================================================
    {
        printf("2. Matrix Multiply (GEMM)\n");
        printf("─────────────────────────────────────────\n");
        
        // C = α * A * B + β * C
        // A: M x K
        // B: K x N
        // C: M x N
        
        const int M = 4, N = 4, K = 4;
        
        // Allocate host matrices (column-major for cuBLAS!)
        float h_A[M * K], h_B[K * N], h_C[M * N];
        
        // Initialize A = identity, B = identity
        for (int i = 0; i < M * K; i++) h_A[i] = 0.0f;
        for (int i = 0; i < K * N; i++) h_B[i] = 0.0f;
        for (int i = 0; i < M * N; i++) h_C[i] = 0.0f;
        
        // Set diagonals (column-major: A[i,j] = A[i + j*M])
        for (int i = 0; i < M && i < K; i++) h_A[i + i * M] = 1.0f;
        for (int i = 0; i < K && i < N; i++) h_B[i + i * K] = 2.0f;
        
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, M * K * sizeof(float));
        cudaMalloc(&d_B, K * N * sizeof(float));
        cudaMalloc(&d_C, M * N * sizeof(float));
        
        cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice);
        
        float alpha = 1.0f, beta = 0.0f;
        
        // GEMM: C = alpha * A * B + beta * C
        CHECK_CUBLAS(cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,  // No transpose
            M, N, K,                    // Dimensions
            &alpha,
            d_A, M,                     // A and its leading dimension
            d_B, K,                     // B and its leading dimension
            &beta,
            d_C, M));                   // C and its leading dimension
        
        cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        
        printf("   A = I (identity), B = 2*I\n");
        printf("   C = A * B:\n");
        for (int i = 0; i < M; i++) {
            printf("   [");
            for (int j = 0; j < N; j++) {
                printf("%.0f ", h_C[i + j * M]);  // Column-major access
            }
            printf("]\n");
        }
        printf("\n");
        
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }
    
    // ========================================================================
    // Part 3: Performance - cuBLAS vs Custom
    // ========================================================================
    {
        printf("3. Performance Benchmark\n");
        printf("─────────────────────────────────────────\n");
        
        const int M = 1024, N = 1024, K = 1024;
        
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, M * K * sizeof(float));
        cudaMalloc(&d_B, K * N * sizeof(float));
        cudaMalloc(&d_C, M * N * sizeof(float));
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        float alpha = 1.0f, beta = 0.0f;
        
        // Warmup
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                    &alpha, d_A, M, d_B, K, &beta, d_C, M);
        cudaDeviceSynchronize();
        
        // Benchmark
        cudaEventRecord(start);
        for (int i = 0; i < 100; i++) {
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                        &alpha, d_A, M, d_B, K, &beta, d_C, M);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        
        // Calculate TFLOPS
        // GEMM does 2*M*N*K FLOPs
        double flops = 2.0 * M * N * K * 100;
        double tflops = flops / (ms / 1000.0) / 1e12;
        
        printf("   Matrix size: %dx%d x %dx%d\n", M, K, K, N);
        printf("   Time: %.4f ms per GEMM\n", ms / 100);
        printf("   Performance: %.2f TFLOPS\n\n", tflops);
        
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    printf("=== Key Points ===\n\n");
    printf("1. Always create/destroy cuBLAS handle\n");
    printf("2. cuBLAS uses column-major order!\n");
    printf("3. Level 1: vectors, Level 2: matrix-vector, Level 3: matrix-matrix\n");
    printf("4. GEMM is highly optimized - use it!\n");
    
    cublasDestroy(handle);
    return 0;
}
