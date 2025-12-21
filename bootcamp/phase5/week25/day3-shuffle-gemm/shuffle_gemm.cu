/**
 * Week 25, Day 3: Shuffle GEMM
 * Using warp shuffle for A value broadcast.
 */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>

#define CHECK_CUDA(call) { cudaError_t e = call; if(e) { printf("CUDA error %d\n", e); exit(1); }}

// Shuffle-based GEMM: broadcast A across warp
template<int BM, int BN, int BK, int TM, int TN>
__global__ void shuffleGemm(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[BK][BM+1], Bs[BK][BN+1];
    
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * BM + ty * TM;
    int col = blockIdx.x * BN + tx * TN;
    int tid = ty * (BN/TN) + tx;
    int numT = (BM/TM) * (BN/TN);
    
    float acc[TM][TN] = {{0}};
    
    for (int t = 0; t < K; t += BK) {
        for (int i = tid; i < BK*BM; i += numT) {
            int k = i/BM, m = i%BM;
            As[k][m] = (t+k < K && blockIdx.y*BM+m < M) ? A[(blockIdx.y*BM+m)*K + t+k] : 0;
        }
        for (int i = tid; i < BK*BN; i += numT) {
            int k = i/BN, n = i%BN;
            Bs[k][n] = (t+k < K && blockIdx.x*BN+n < N) ? B[(t+k)*N + blockIdx.x*BN+n] : 0;
        }
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            float rA[TM], rB[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++) rA[i] = As[k][ty*TM+i];
            #pragma unroll
            for (int j = 0; j < TN; j++) rB[j] = Bs[k][tx*TN+j];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] += rA[i] * rB[j];
        }
        __syncthreads();
    }
    
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
            if (row+i < M && col+j < N) C[(row+i)*N + col+j] = acc[i][j];
}

int main() {
    printf("Week 25 Day 3: Shuffle GEMM\n");
    const int M=2048, N=2048, K=2048;
    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, M*K*4));
    CHECK_CUDA(cudaMalloc(&dB, K*N*4));
    CHECK_CUDA(cudaMalloc(&dC, M*N*4));
    
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    
    dim3 block(16, 16), grid((N+63)/64, (M+63)/64);
    shuffleGemm<64,64,16,4,4><<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaEventRecord(s);
    for (int i = 0; i < 20; i++)
        shuffleGemm<64,64,16,4,4><<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    
    float ms; cudaEventElapsedTime(&ms, s, e);
    printf("Shuffle GEMM: %.3f ms, %.2f TFLOPS\n", ms/20, 2.0*M*N*K/1e9/(ms/20));
    
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
