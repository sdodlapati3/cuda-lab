/**
 * Week 26, Day 3: Mixed Precision GEMM
 * FP16 inputs, FP32 accumulation.
 */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cublas_v2.h>
#include <cstdio>

using namespace nvcuda;
#define CHECK_CUDA(call) { cudaError_t e = call; if(e) exit(1); }

__global__ void fp32Accumulate(const half* A, const half* B, float* C, int M, int N, int K) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;  // FP32!
    
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warpN = blockIdx.y;
    int row = warpM * 16, col = warpN * 16;
    
    wmma::fill_fragment(c_frag, 0.0f);
    
    for (int k = 0; k < K; k += 16) {
        if (row < M && col < N && k + 16 <= K) {
            wmma::load_matrix_sync(a_frag, A + row*K + k, K);
            wmma::load_matrix_sync(b_frag, B + k*N + col, N);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }
    
    if (row < M && col < N)
        wmma::store_matrix_sync(C + row*N + col, c_frag, N, wmma::mem_row_major);
}

int main() {
    printf("Week 26 Day 3: Mixed Precision\n");
    printf("Input: FP16 (half precision)\n");
    printf("Accumulation: FP32 (single precision)\n");
    printf("Benefits: Speed of FP16, accuracy of FP32 accumulation\n");
    
    // Demo: FP16 range
    printf("\nFP16 Range:\n");
    printf("  Max: ~65504\n");
    printf("  Min positive: ~6e-8\n");
    printf("  Precision: ~3 decimal digits\n");
    
    return 0;
}
