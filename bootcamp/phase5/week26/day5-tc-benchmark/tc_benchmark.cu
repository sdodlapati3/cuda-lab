/**
 * Week 26, Day 5: Tensor Core Benchmarking
 */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>

#define CHECK_CUDA(call) { cudaError_t e = call; if(e) exit(1); }
#define CHECK_CUBLAS(call) { cublasStatus_t s = call; if(s) exit(1); }

int main() {
    printf("Week 26 Day 5: Tensor Core Benchmarking\n\n");
    
    const int M = 4096, N = 4096, K = 4096;
    double gflops = 2.0 * M * N * K / 1e9;
    
    // Allocate FP32 and FP16 matrices
    float *dA32, *dB32, *dC32;
    half *dA16, *dB16;
    float *dC16;
    
    CHECK_CUDA(cudaMalloc(&dA32, M*K*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB32, K*N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC32, M*N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dA16, M*K*sizeof(half)));
    CHECK_CUDA(cudaMalloc(&dB16, K*N*sizeof(half)));
    CHECK_CUDA(cudaMalloc(&dC16, M*N*sizeof(float)));
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int iterations = 20;
    
    // FP32 SGEMM
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB32, N, dA32, K, &beta, dC32, N);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB32, N, dA32, K, &beta, dC32, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float fp32Ms;
    cudaEventElapsedTime(&fp32Ms, start, stop);
    fp32Ms /= iterations;
    
    // FP16 Tensor Core GEMM (via cuBLAS)
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                 &alpha, dB16, CUDA_R_16F, N, dA16, CUDA_R_16F, K,
                 &beta, dC16, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                     &alpha, dB16, CUDA_R_16F, N, dA16, CUDA_R_16F, K,
                     &beta, dC16, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float tcMs;
    cudaEventElapsedTime(&tcMs, start, stop);
    tcMs /= iterations;
    
    printf("Matrix: %d × %d × %d (%.1f GFLOP)\n\n", M, N, K, gflops);
    printf("┌───────────────────┬──────────┬──────────┐\n");
    printf("│ Mode              │ Time(ms) │ TFLOPS   │\n");
    printf("├───────────────────┼──────────┼──────────┤\n");
    printf("│ FP32 (cuBLAS)     │ %8.3f │ %8.2f │\n", fp32Ms, gflops/fp32Ms);
    printf("│ FP16 Tensor Core  │ %8.3f │ %8.2f │\n", tcMs, gflops/tcMs);
    printf("└───────────────────┴──────────┴──────────┘\n\n");
    printf("Tensor Core speedup: %.2fx\n", fp32Ms/tcMs);
    
    cublasDestroy(handle);
    cudaFree(dA32); cudaFree(dB32); cudaFree(dC32);
    cudaFree(dA16); cudaFree(dB16); cudaFree(dC16);
    
    return 0;
}
