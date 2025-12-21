/**
 * Week 27, Day 1: Double Buffering with Tensor Cores
 * Pipeline async loads with WMMA compute.
 */
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <mma.h>
#include <cstdio>

using namespace nvcuda;
#define CHECK_CUDA(call) { cudaError_t e = call; if(e) exit(1); }

constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
constexpr int BM = 64, BN = 64, BK = 16;

__global__ void doubleBufferTC(const half* A, const half* B, float* C, int M, int N, int K) {
    __shared__ half As[2][BK][BM + 8];  // +8 for bank conflict avoidance
    __shared__ half Bs[2][BK][BN + 8];
    
    int warpId = threadIdx.x / 32;
    int warpM = warpId / (BN/WMMA_N);
    int warpN = warpId % (BN/WMMA_N);
    
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);
    
    int aRow = blockIdx.y * BM;
    int bCol = blockIdx.x * BN;
    int numTiles = K / BK;
    int buf = 0;
    
    // Preload first tile
    for (int i = threadIdx.x; i < BK * BM; i += blockDim.x) {
        int k = i / BM, m = i % BM;
        __pipeline_memcpy_async(&As[0][k][m], &A[(aRow+m)*K + k], sizeof(half));
    }
    for (int i = threadIdx.x; i < BK * BN; i += blockDim.x) {
        int k = i / BN, n = i % BN;
        __pipeline_memcpy_async(&Bs[0][k][n], &B[k*N + bCol + n], sizeof(half));
    }
    __pipeline_commit();
    
    for (int t = 0; t < numTiles; t++) {
        int nextBuf = 1 - buf;
        
        // Start loading next tile (if exists)
        if (t + 1 < numTiles) {
            int nextK = (t + 1) * BK;
            for (int i = threadIdx.x; i < BK * BM; i += blockDim.x) {
                int k = i / BM, m = i % BM;
                __pipeline_memcpy_async(&As[nextBuf][k][m], &A[(aRow+m)*K + nextK + k], sizeof(half));
            }
            for (int i = threadIdx.x; i < BK * BN; i += blockDim.x) {
                int k = i / BN, n = i % BN;
                __pipeline_memcpy_async(&Bs[nextBuf][k][n], &B[(nextK+k)*N + bCol + n], sizeof(half));
            }
            __pipeline_commit();
        }
        
        // Wait for current tile
        __pipeline_wait_prior(1);
        __syncthreads();
        
        // Compute with current tile
        wmma::load_matrix_sync(a_frag, &As[buf][0][warpM*WMMA_M], BM + 8);
        wmma::load_matrix_sync(b_frag, &Bs[buf][0][warpN*WMMA_N], BN + 8);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        
        buf = nextBuf;
    }
    
    // Store result
    int cRow = aRow + warpM * WMMA_M;
    int cCol = bCol + warpN * WMMA_N;
    if (cRow < M && cCol < N)
        wmma::store_matrix_sync(&C[cRow*N + cCol], c_frag, N, wmma::mem_row_major);
}

int main() {
    printf("Week 27 Day 1: Double Buffering with Tensor Cores\n");
    printf("Pipeline: Load tile N+1 while computing tile N\n");
    printf("Benefit: Hide global memory latency\n");
    return 0;
}
