/**
 * Week 28, Day 4: Batched GEMM
 * Multiple matrix multiplications.
 */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>

#define CHECK_CUDA(call) { cudaError_t e = call; if(e) exit(1); }
#define CHECK_CUBLAS(call) { cublasStatus_t s = call; if(s) exit(1); }

int main() {
    printf("Week 28 Day 4: Batched GEMM\n\n");
    
    const int M = 512, N = 512, K = 512;
    const int batchCount = 64;
    
    printf("Problem: %d × (%d × %d × %d)\n", batchCount, M, N, K);
    printf("Total ops: %.2f GFLOP\n\n", 2.0 * batchCount * M * N * K / 1e9);
    
    // Allocate batched matrices
    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, batchCount * M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB, batchCount * K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC, batchCount * M * N * sizeof(float)));
    
    // Create array of pointers
    float **dAarray, **dBarray, **dCarray;
    CHECK_CUDA(cudaMalloc(&dAarray, batchCount * sizeof(float*)));
    CHECK_CUDA(cudaMalloc(&dBarray, batchCount * sizeof(float*)));
    CHECK_CUDA(cudaMalloc(&dCarray, batchCount * sizeof(float*)));
    
    float **hAarray = new float*[batchCount];
    float **hBarray = new float*[batchCount];
    float **hCarray = new float*[batchCount];
    for (int i = 0; i < batchCount; i++) {
        hAarray[i] = dA + i * M * K;
        hBarray[i] = dB + i * K * N;
        hCarray[i] = dC + i * M * N;
    }
    CHECK_CUDA(cudaMemcpy(dAarray, hAarray, batchCount*sizeof(float*), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dBarray, hBarray, batchCount*sizeof(float*), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dCarray, hCarray, batchCount*sizeof(float*), cudaMemcpyHostToDevice));
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    float alpha = 1.0f, beta = 0.0f;
    int iterations = 20;
    
    // Warmup
    cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                       N, M, K, &alpha,
                       dBarray, N, dAarray, K,
                       &beta, dCarray, N, batchCount);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                           N, M, K, &alpha,
                           dBarray, N, dAarray, K,
                           &beta, dCarray, N, batchCount);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= iterations;
    
    double gflops = 2.0 * batchCount * M * N * K / 1e9;
    printf("Batched GEMM (cuBLAS):\n");
    printf("  Time: %.3f ms\n", ms);
    printf("  Performance: %.2f TFLOPS\n", gflops / ms);
    
    printf("\nBatched GEMM Modes:\n");
    printf("  - Batched: Array of pointers (varying locations)\n");
    printf("  - Strided: Fixed stride between matrices\n");
    printf("  - CUTLASS GroupedGemm: Different sizes per batch\n");
    
    cublasDestroy(handle);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cudaFree(dAarray); cudaFree(dBarray); cudaFree(dCarray);
    delete[] hAarray; delete[] hBarray; delete[] hCarray;
    
    return 0;
}
