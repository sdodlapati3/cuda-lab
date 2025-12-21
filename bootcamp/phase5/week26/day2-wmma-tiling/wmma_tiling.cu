/**
 * Week 26, Day 2: WMMA Tiling
 * Tiled GEMM using Tensor Cores.
 */
#include <cuda_runtime.h>
#include <mma.h>
#include <cublas_v2.h>
#include <cstdio>

using namespace nvcuda;
#define CHECK_CUDA(call) { cudaError_t e = call; if(e) exit(1); }

// WMMA dimensions
constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
constexpr int BM = 64, BN = 64, BK = 16;

__global__ void wmmaTiledGemm(const half* A, const half* B, float* C, int M, int N, int K) {
    __shared__ half As[BK][BM], Bs[BK][BN];
    
    int warpId = (threadIdx.x + threadIdx.y * blockDim.x) / 32;
    int warpM = warpId / (BN/WMMA_N);
    int warpN = warpId % (BN/WMMA_N);
    
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);
    
    int aRow = blockIdx.y * BM;
    int bCol = blockIdx.x * BN;
    
    for (int t = 0; t < K; t += BK) {
        // Load tiles to shared memory (simplified)
        int tid = threadIdx.x + threadIdx.y * blockDim.x;
        for (int i = tid; i < BK * BM; i += blockDim.x * blockDim.y) {
            int k = i / BM, m = i % BM;
            As[k][m] = (t+k < K && aRow+m < M) ? A[(aRow+m)*K + t+k] : __float2half(0.0f);
        }
        for (int i = tid; i < BK * BN; i += blockDim.x * blockDim.y) {
            int k = i / BN, n = i % BN;
            Bs[k][n] = (t+k < K && bCol+n < N) ? B[(t+k)*N + bCol+n] : __float2half(0.0f);
        }
        __syncthreads();
        
        // WMMA operations
        wmma::load_matrix_sync(a_frag, &As[0][warpM*WMMA_M], BM);
        wmma::load_matrix_sync(b_frag, &Bs[0][warpN*WMMA_N], BN);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }
    
    // Store result
    int cRow = aRow + warpM * WMMA_M;
    int cCol = bCol + warpN * WMMA_N;
    if (cRow < M && cCol < N) {
        wmma::store_matrix_sync(&C[cRow*N + cCol], c_frag, N, wmma::mem_row_major);
    }
}

int main() {
    printf("Week 26 Day 2: WMMA Tiling\n");
    printf("Block tile: %dx%d, WMMA tile: %dx%dx%d\n", BM, BN, WMMA_M, WMMA_N, WMMA_K);
    printf("Warps per block: %d\n", (BM/WMMA_M) * (BN/WMMA_N));
    return 0;
}
