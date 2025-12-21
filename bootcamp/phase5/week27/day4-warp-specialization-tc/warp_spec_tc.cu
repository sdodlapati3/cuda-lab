/**
 * Week 27, Day 4: Warp Specialization for Tensor Cores
 * Separate warps for async loads vs TC compute.
 */
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <mma.h>
#include <cstdio>

using namespace nvcuda;

constexpr int BM = 128, BN = 128, BK = 32;
constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
constexpr int NUM_WARPS = 8;
constexpr int LOADER_WARPS = 2;
constexpr int COMPUTE_WARPS = NUM_WARPS - LOADER_WARPS;

__global__ void warpSpecializedTC(const half* A, const half* B, float* C, int M, int N, int K) {
    __shared__ half As[2][BK][BM];
    __shared__ half Bs[2][BK][BN];
    __shared__ int readyFlag[2];
    
    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;
    bool isLoader = (warpId < LOADER_WARPS);
    
    int aRow = blockIdx.y * BM;
    int bCol = blockIdx.x * BN;
    
    if (isLoader) {
        // Loader warps: async copy tiles
        for (int t = 0; t < K / BK; t++) {
            int buf = t % 2;
            int tileK = t * BK;
            
            // Each loader warp handles part of the tile
            if (warpId == 0) {
                for (int i = laneId; i < BK * BM / 2; i += 32) {
                    int k = i / BM, m = i % BM;
                    if (aRow + m < M && tileK + k < K)
                        __pipeline_memcpy_async(&As[buf][k][m], &A[(aRow+m)*K + tileK + k], sizeof(half));
                }
            } else {
                for (int i = laneId; i < BK * BN / 2; i += 32) {
                    int k = i / BN, n = i % BN;
                    if (tileK + k < K && bCol + n < N)
                        __pipeline_memcpy_async(&Bs[buf][k][n], &B[(tileK+k)*N + bCol + n], sizeof(half));
                }
            }
            __pipeline_commit();
            __pipeline_wait_prior(0);
            
            if (laneId == 0) atomicExch(&readyFlag[buf], t + 1);
            __threadfence_block();
        }
    } else {
        // Compute warps: WMMA operations
        int computeWarp = warpId - LOADER_WARPS;
        int warpM = computeWarp / (BN/WMMA_N);
        int warpN = computeWarp % (BN/WMMA_N);
        
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);
        
        for (int t = 0; t < K / BK; t++) {
            int buf = t % 2;
            
            // Wait for loader
            while (atomicAdd(&readyFlag[buf], 0) < t + 1) {}
            
            // Compute with WMMA
            for (int kk = 0; kk < BK; kk += WMMA_K) {
                wmma::load_matrix_sync(a_frag, &As[buf][kk][warpM*WMMA_M], BM);
                wmma::load_matrix_sync(b_frag, &Bs[buf][kk][warpN*WMMA_N], BN);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
        }
        
        // Store result
        int cRow = aRow + warpM * WMMA_M;
        int cCol = bCol + warpN * WMMA_N;
        if (cRow < M && cCol < N)
            wmma::store_matrix_sync(&C[cRow*N + cCol], c_frag, N, wmma::mem_row_major);
    }
}

int main() {
    printf("Week 27 Day 4: Warp Specialization for Tensor Cores\n");
    printf("Configuration:\n");
    printf("  Total warps: %d\n", NUM_WARPS);
    printf("  Loader warps: %d\n", LOADER_WARPS);
    printf("  Compute warps: %d\n", COMPUTE_WARPS);
    printf("Benefit: Overlapped loading and computing\n");
    return 0;
}
