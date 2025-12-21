/**
 * cublas_advanced.cu - Advanced cuBLAS features
 * 
 * Learning objectives:
 * - Batched operations
 * - Tensor Core acceleration
 * - Mixed precision
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <vector>

#define CHECK_CUBLAS(call) { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        printf("cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
        exit(1); \
    } \
}

int main() {
    printf("=== cuBLAS Advanced Demo ===\n\n");
    
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // ========================================================================
    // Part 1: Batched GEMM
    // ========================================================================
    {
        printf("1. Batched GEMM\n");
        printf("─────────────────────────────────────────\n");
        
        const int M = 64, N = 64, K = 64;
        const int batch_count = 1000;
        
        // Allocate batch of matrices (strided layout)
        float *d_A, *d_B, *d_C;
        size_t stride_A = M * K;
        size_t stride_B = K * N;
        size_t stride_C = M * N;
        
        cudaMalloc(&d_A, batch_count * stride_A * sizeof(float));
        cudaMalloc(&d_B, batch_count * stride_B * sizeof(float));
        cudaMalloc(&d_C, batch_count * stride_C * sizeof(float));
        
        float alpha = 1.0f, beta = 0.0f;
        
        // Warmup
        cublasSgemmStridedBatched(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K,
            &alpha,
            d_A, M, stride_A,
            d_B, K, stride_B,
            &beta,
            d_C, M, stride_C,
            batch_count);
        cudaDeviceSynchronize();
        
        // Benchmark
        cudaEventRecord(start);
        for (int i = 0; i < 100; i++) {
            cublasSgemmStridedBatched(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                M, N, K,
                &alpha,
                d_A, M, stride_A,
                d_B, K, stride_B,
                &beta,
                d_C, M, stride_C,
                batch_count);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        
        // TFLOPS: 2*M*N*K * batch * iterations
        double flops = 2.0 * M * N * K * batch_count * 100;
        double tflops = flops / (ms / 1000.0) / 1e12;
        
        printf("   Batch size: %d matrices of %dx%d\n", batch_count, M, N);
        printf("   Time: %.4f ms per batch\n", ms / 100);
        printf("   Performance: %.2f TFLOPS\n\n", tflops);
        
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }
    
    // ========================================================================
    // Part 2: Tensor Cores with GemmEx
    // ========================================================================
    {
        printf("2. Tensor Core GEMM (FP16)\n");
        printf("─────────────────────────────────────────\n");
        
        const int M = 4096, N = 4096, K = 4096;
        
        // Allocate FP16 inputs
        half *d_A_fp16, *d_B_fp16;
        float *d_C_fp32;
        
        cudaMalloc(&d_A_fp16, M * K * sizeof(half));
        cudaMalloc(&d_B_fp16, K * N * sizeof(half));
        cudaMalloc(&d_C_fp32, M * N * sizeof(float));
        
        float alpha = 1.0f, beta = 0.0f;
        
        // Enable tensor ops
        CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
        
        // Warmup
        cublasGemmEx(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K,
            &alpha,
            d_A_fp16, CUDA_R_16F, M,
            d_B_fp16, CUDA_R_16F, K,
            &beta,
            d_C_fp32, CUDA_R_32F, M,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cudaDeviceSynchronize();
        
        // Benchmark
        cudaEventRecord(start);
        for (int i = 0; i < 100; i++) {
            cublasGemmEx(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                M, N, K,
                &alpha,
                d_A_fp16, CUDA_R_16F, M,
                d_B_fp16, CUDA_R_16F, K,
                &beta,
                d_C_fp32, CUDA_R_32F, M,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms_tc;
        cudaEventElapsedTime(&ms_tc, start, stop);
        
        double flops = 2.0 * M * N * K * 100;
        double tflops_tc = flops / (ms_tc / 1000.0) / 1e12;
        
        printf("   Matrix size: %dx%d (FP16 inputs, FP32 output)\n", M, N);
        printf("   Tensor Core time: %.4f ms per GEMM\n", ms_tc / 100);
        printf("   Tensor Core perf: %.2f TFLOPS\n", tflops_tc);
        
        // Compare with regular FP32
        float *d_A_fp32, *d_B_fp32;
        cudaMalloc(&d_A_fp32, M * K * sizeof(float));
        cudaMalloc(&d_B_fp32, K * N * sizeof(float));
        
        CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
        
        cudaEventRecord(start);
        for (int i = 0; i < 100; i++) {
            cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                M, N, K,
                &alpha,
                d_A_fp32, M,
                d_B_fp32, K,
                &beta,
                d_C_fp32, M);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms_fp32;
        cudaEventElapsedTime(&ms_fp32, start, stop);
        double tflops_fp32 = flops / (ms_fp32 / 1000.0) / 1e12;
        
        printf("   FP32 time: %.4f ms per GEMM\n", ms_fp32 / 100);
        printf("   FP32 perf: %.2f TFLOPS\n", tflops_fp32);
        printf("   Tensor Core speedup: %.2fx\n\n", ms_fp32 / ms_tc);
        
        cudaFree(d_A_fp16);
        cudaFree(d_B_fp16);
        cudaFree(d_A_fp32);
        cudaFree(d_B_fp32);
        cudaFree(d_C_fp32);
    }
    
    // ========================================================================
    // Part 3: cuBLAS with Streams
    // ========================================================================
    {
        printf("3. cuBLAS with Multiple Streams\n");
        printf("─────────────────────────────────────────\n");
        
        const int M = 512, N = 512, K = 512;
        const int num_streams = 4;
        
        cudaStream_t streams[num_streams];
        float *d_A[num_streams], *d_B[num_streams], *d_C[num_streams];
        
        for (int i = 0; i < num_streams; i++) {
            cudaStreamCreate(&streams[i]);
            cudaMalloc(&d_A[i], M * K * sizeof(float));
            cudaMalloc(&d_B[i], K * N * sizeof(float));
            cudaMalloc(&d_C[i], M * N * sizeof(float));
        }
        
        float alpha = 1.0f, beta = 0.0f;
        
        // Single stream baseline
        cudaEventRecord(start);
        for (int iter = 0; iter < 100; iter++) {
            for (int i = 0; i < num_streams; i++) {
                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    M, N, K, &alpha,
                    d_A[i], M, d_B[i], K, &beta, d_C[i], M);
            }
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms_single;
        cudaEventElapsedTime(&ms_single, start, stop);
        
        // Multi-stream
        cudaEventRecord(start);
        for (int iter = 0; iter < 100; iter++) {
            for (int i = 0; i < num_streams; i++) {
                cublasSetStream(handle, streams[i]);
                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    M, N, K, &alpha,
                    d_A[i], M, d_B[i], K, &beta, d_C[i], M);
            }
        }
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms_multi;
        cudaEventElapsedTime(&ms_multi, start, stop);
        
        printf("   %d independent %dx%d GEMMs per iteration\n", num_streams, M, N);
        printf("   Single stream: %.4f ms\n", ms_single / 100);
        printf("   Multi-stream:  %.4f ms\n", ms_multi / 100);
        printf("   Speedup: %.2fx\n\n", ms_single / ms_multi);
        
        cublasSetStream(handle, 0);  // Reset to default
        
        for (int i = 0; i < num_streams; i++) {
            cudaStreamDestroy(streams[i]);
            cudaFree(d_A[i]);
            cudaFree(d_B[i]);
            cudaFree(d_C[i]);
        }
    }
    
    printf("=== Key Points ===\n\n");
    printf("1. Use StridedBatched for contiguous batch layouts\n");
    printf("2. Enable CUBLAS_TENSOR_OP_MATH for Tensor Cores\n");
    printf("3. Use cublasGemmEx for mixed precision\n");
    printf("4. Use cublasSetStream for concurrent operations\n");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);
    
    return 0;
}
