/**
 * Week 29, Day 5: Quantized GEMM
 * INT8 matrix multiplication.
 */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>

#define CHECK_CUDA(call) { cudaError_t e = call; if(e) exit(1); }
#define CHECK_CUBLAS(call) { cublasStatus_t s = call; if(s) exit(1); }

// INT8 GEMM: C_fp32 = A_int8 × B_int8 × scale_a × scale_b
// Then dequantize to output

__global__ void quantizeMatrixKernel(const float* input, int8_t* output, 
                                      float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int32_t q = __float2int_rn(input[idx] / scale);
        output[idx] = static_cast<int8_t>(max(-128, min(127, q)));
    }
}

int main() {
    printf("Week 29 Day 5: Quantized GEMM\n\n");
    
    const int M = 1024, N = 1024, K = 1024;
    
    printf("INT8 GEMM Formula:\n");
    printf("  C_fp32 = (A_int8 × B_int8) × scale_a × scale_b\n\n");
    
    // Allocate FP32 matrices
    float *dA_fp32, *dB_fp32, *dC_fp32;
    CHECK_CUDA(cudaMalloc(&dA_fp32, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB_fp32, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC_fp32, M * N * sizeof(float)));
    
    // Allocate INT8 matrices
    int8_t *dA_int8, *dB_int8;
    int32_t *dC_int32;
    CHECK_CUDA(cudaMalloc(&dA_int8, M * K * sizeof(int8_t)));
    CHECK_CUDA(cudaMalloc(&dB_int8, K * N * sizeof(int8_t)));
    CHECK_CUDA(cudaMalloc(&dC_int32, M * N * sizeof(int32_t)));
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Benchmark FP32 GEMM
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, 
                &alpha, dB_fp32, N, dA_fp32, K, &beta, dC_fp32, N);
    cudaDeviceSynchronize();
    
    int iterations = 20;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                    &alpha, dB_fp32, N, dA_fp32, K, &beta, dC_fp32, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float fp32Ms;
    cudaEventElapsedTime(&fp32Ms, start, stop);
    fp32Ms /= iterations;
    
    // Benchmark INT8 GEMM via cublasGemmEx
    int32_t alpha_i = 1, beta_i = 0;
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                 &alpha_i, dB_int8, CUDA_R_8I, N, dA_int8, CUDA_R_8I, K,
                 &beta_i, dC_int32, CUDA_R_32I, N, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                     &alpha_i, dB_int8, CUDA_R_8I, N, dA_int8, CUDA_R_8I, K,
                     &beta_i, dC_int32, CUDA_R_32I, N, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float int8Ms;
    cudaEventElapsedTime(&int8Ms, start, stop);
    int8Ms /= iterations;
    
    double gflops = 2.0 * M * N * K / 1e9;
    printf("Performance Comparison (%d × %d × %d):\n", M, N, K);
    printf("┌───────────┬──────────┬──────────┐\n");
    printf("│ Precision │ Time(ms) │ TFLOPS   │\n");
    printf("├───────────┼──────────┼──────────┤\n");
    printf("│ FP32      │ %8.3f │ %8.2f │\n", fp32Ms, gflops/fp32Ms);
    printf("│ INT8      │ %8.3f │ %8.2f │\n", int8Ms, gflops/int8Ms);
    printf("└───────────┴──────────┴──────────┘\n");
    printf("\nINT8 Speedup: %.2fx\n", fp32Ms/int8Ms);
    
    printf("\nA100 INT8 Tensor Core Peak: 624 TOPS\n");
    
    cublasDestroy(handle);
    cudaFree(dA_fp32); cudaFree(dB_fp32); cudaFree(dC_fp32);
    cudaFree(dA_int8); cudaFree(dB_int8); cudaFree(dC_int32);
    
    return 0;
}
