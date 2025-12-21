/**
 * Week 26, Day 1: Tensor Core Introduction
 * 
 * First exposure to Tensor Cores using WMMA API.
 * Simple 16x16x16 matrix multiplication.
 */

#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>

using namespace nvcuda;

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// WMMA dimensions
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

// Simple Tensor Core GEMM using WMMA
// Each warp computes one 16x16 output tile
__global__ void wmmaGemmBasic(const half* A, const half* B, float* C,
                               int M, int N, int K) {
    // Warp and lane indices
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y);
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x);
    
    // Declare fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    // Initialize accumulator to zero
    wmma::fill_fragment(c_frag, 0.0f);
    
    // Output tile position
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    
    // Loop over K dimension
    for (int k = 0; k < K; k += WMMA_K) {
        int aRow = cRow;
        int aCol = k;
        int bRow = k;
        int bCol = cCol;
        
        // Bounds check
        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            // Load matrices
            wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
            wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);
            
            // Perform matrix multiply-accumulate
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }
    
    // Store result
    if (cRow < M && cCol < N) {
        wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
    }
}

// Convert FP32 to FP16
__global__ void convertFp32ToFp16(half* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(in[idx]);
    }
}

void initMatrix(float* mat, int size) {
    for (int i = 0; i < size; i++) {
        mat[i] = (rand() / (float)RAND_MAX - 0.5f) * 2.0f;
    }
}

int main() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║        Week 26, Day 1: Tensor Core Introduction              ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    // Print device info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    
    if (prop.major < 7) {
        printf("ERROR: Tensor Cores require compute capability >= 7.0\n");
        return 1;
    }
    printf("Tensor Cores: Supported!\n\n");
    
    // Matrix dimensions (must be multiples of 16 for this simple version)
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;
    
    printf("Matrix dimensions: %d × %d × %d\n", M, N, K);
    printf("WMMA tile: %d × %d × %d\n", WMMA_M, WMMA_N, WMMA_K);
    double gflops = 2.0 * M * N * K / 1e9;
    printf("Operations: %.2f GFLOP\n\n", gflops);
    
    // Allocate host memory
    float *hA = new float[M * K];
    float *hB = new float[K * N];
    float *hC = new float[M * N];
    float *hRef = new float[M * N];
    
    srand(42);
    initMatrix(hA, M * K);
    initMatrix(hB, K * N);
    
    // Allocate device memory
    float *dA_fp32, *dB_fp32, *dC;
    half *dA_fp16, *dB_fp16;
    
    CHECK_CUDA(cudaMalloc(&dA_fp32, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB_fp32, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dA_fp16, M * K * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&dB_fp16, K * N * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&dC, M * N * sizeof(float)));
    
    CHECK_CUDA(cudaMemcpy(dA_fp32, hA, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB_fp32, hB, K * N * sizeof(float), cudaMemcpyHostToDevice));
    
    // Convert to FP16
    int threads = 256;
    convertFp32ToFp16<<<(M * K + threads - 1) / threads, threads>>>(dA_fp16, dA_fp32, M * K);
    convertFp32ToFp16<<<(K * N + threads - 1) / threads, threads>>>(dB_fp16, dB_fp32, K * N);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int iterations = 20;
    
    // cuBLAS FP16 reference
    cublasHandle_t handle;
    cublasCreate(&handle);
    half alpha_h = __float2half(1.0f);
    half beta_h = __float2half(0.0f);
    
    // Use cublasGemmEx for FP16
    half *dC_fp16;
    CHECK_CUDA(cudaMalloc(&dC_fp16, M * N * sizeof(half)));
    
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                 &alpha_h, dB_fp16, CUDA_R_16F, N,
                 dA_fp16, CUDA_R_16F, K,
                 &beta_h, dC_fp16, CUDA_R_16F, N,
                 CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                     &alpha_h, dB_fp16, CUDA_R_16F, N,
                     dA_fp16, CUDA_R_16F, K,
                     &beta_h, dC_fp16, CUDA_R_16F, N,
                     CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float cublasMs;
    cudaEventElapsedTime(&cublasMs, start, stop);
    cublasMs /= iterations;
    
    printf("Performance:\n");
    printf("┌─────────────────────────┬──────────┬──────────┬──────────┐\n");
    printf("│ Implementation          │ Time(ms) │ TFLOPS   │ %%cuBLAS  │\n");
    printf("├─────────────────────────┼──────────┼──────────┼──────────┤\n");
    printf("│ cuBLAS Tensor Core      │ %8.3f │ %8.2f │   100.0%% │\n",
           cublasMs, gflops / cublasMs);
    
    // WMMA kernel
    // Each warp computes 16x16, need M/16 × N/16 warps
    // One warp = 32 threads, organized as 1 row in x
    dim3 block(1, 4);  // 4 warps per block (each warp is 1 "thread" in this simple mapping)
    dim3 grid((N / WMMA_N + block.x - 1) / block.x, 
              (M / WMMA_M + block.y - 1) / block.y);
    
    wmmaGemmBasic<<<grid, block>>>(dA_fp16, dB_fp16, dC, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        wmmaGemmBasic<<<grid, block>>>(dA_fp16, dB_fp16, dC, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float wmmaMs;
    cudaEventElapsedTime(&wmmaMs, start, stop);
    wmmaMs /= iterations;
    
    printf("│ WMMA Basic              │ %8.3f │ %8.2f │ %7.1f%% │\n",
           wmmaMs, gflops / wmmaMs, 100.0f * cublasMs / wmmaMs);
    printf("└─────────────────────────┴──────────┴──────────┴──────────┘\n\n");
    
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║                    Key Takeaways                             ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║ 1. Tensor Cores use WMMA API for 16×16×16 operations         ║\n");
    printf("║ 2. Fragments represent matrix tiles in registers             ║\n");
    printf("║ 3. mma_sync performs the actual Tensor Core operation        ║\n");
    printf("║ 4. Basic version is far from optimal (needs tiling!)         ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    
    // Cleanup
    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dA_fp32);
    cudaFree(dB_fp32);
    cudaFree(dA_fp16);
    cudaFree(dB_fp16);
    cudaFree(dC);
    cudaFree(dC_fp16);
    delete[] hA;
    delete[] hB;
    delete[] hC;
    delete[] hRef;
    
    return 0;
}
