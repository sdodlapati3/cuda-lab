/**
 * Week 27, Day 2: Multi-Stage Pipelining
 * 3-4 stage pipeline for hiding latency.
 */
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <mma.h>
#include <cstdio>

using namespace nvcuda;

constexpr int STAGES = 3;
constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
constexpr int BM = 64, BN = 64, BK = 16;

__global__ void multiStageTC(const half* A, const half* B, float* C, int M, int N, int K) {
    __shared__ half As[STAGES][BK][BM + 8];
    __shared__ half Bs[STAGES][BK][BN + 8];
    
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
    
    // Prefetch first STAGES tiles
    for (int s = 0; s < min(STAGES, numTiles); s++) {
        int tileK = s * BK;
        for (int i = threadIdx.x; i < BK * BM; i += blockDim.x) {
            int k = i / BM, m = i % BM;
            __pipeline_memcpy_async(&As[s][k][m], &A[(aRow+m)*K + tileK + k], sizeof(half));
        }
        for (int i = threadIdx.x; i < BK * BN; i += blockDim.x) {
            int k = i / BN, n = i % BN;
            __pipeline_memcpy_async(&Bs[s][k][n], &B[(tileK+k)*N + bCol + n], sizeof(half));
        }
        __pipeline_commit();
    }
    
    for (int t = 0; t < numTiles; t++) {
        int readStage = t % STAGES;
        
        // Start loading future tile
        if (t + STAGES < numTiles) {
            int futureK = (t + STAGES) * BK;
            int writeStage = (t + STAGES) % STAGES;
            for (int i = threadIdx.x; i < BK * BM; i += blockDim.x) {
                int k = i / BM, m = i % BM;
                __pipeline_memcpy_async(&As[writeStage][k][m], &A[(aRow+m)*K + futureK + k], sizeof(half));
            }
            for (int i = threadIdx.x; i < BK * BN; i += blockDim.x) {
                int k = i / BN, n = i % BN;
                __pipeline_memcpy_async(&Bs[writeStage][k][n], &B[(futureK+k)*N + bCol + n], sizeof(half));
            }
            __pipeline_commit();
        }
        
        __pipeline_wait_prior(STAGES - 1);
        __syncthreads();
        
        // Compute
        wmma::load_matrix_sync(a_frag, &As[readStage][0][warpM*WMMA_M], BM + 8);
        wmma::load_matrix_sync(b_frag, &Bs[readStage][0][warpN*WMMA_N], BN + 8);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        
        __syncthreads();
    }
    
    int cRow = aRow + warpM * WMMA_M;
    int cCol = bCol + warpN * WMMA_N;
    if (cRow < M && cCol < N)
        wmma::store_matrix_sync(&C[cRow*N + cCol], c_frag, N, wmma::mem_row_major);
}

int main() {
    printf("Week 27 Day 2: Multi-Stage Pipelining\n");
    printf("Stages: %d (buffers in flight)\n", STAGES);
    printf("Benefit: Better latency hiding for high-latency memory\n");
    return 0;
}
