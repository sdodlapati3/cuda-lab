#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

#define BLOCK_SIZE 256
#define RADIUS 3

// SOLUTION: Naive stencil (no shared memory)
__global__ void stencil_naive(float* out, const float* in, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < n) {
        float sum = 0.0f;
        for (int j = -RADIUS; j <= RADIUS; j++) {
            int neighbor = max(0, min(n-1, idx + j));
            sum += in[neighbor];
        }
        out[idx] = sum / (2 * RADIUS + 1);
    }
}

// SOLUTION: Tiled stencil with shared memory
__global__ void stencil_tiled(float* out, const float* in, int n) {
    __shared__ float tile[BLOCK_SIZE + 2 * RADIUS];
    
    int gidx = threadIdx.x + blockIdx.x * blockDim.x;
    int lidx = threadIdx.x + RADIUS;
    
    // Load main elements
    if (gidx < n) {
        tile[lidx] = in[gidx];
    }
    
    // Load left halo
    if (threadIdx.x < RADIUS) {
        int halo_idx = gidx - RADIUS;
        tile[threadIdx.x] = (halo_idx >= 0) ? in[halo_idx] : in[0];
    }
    
    // Load right halo
    if (threadIdx.x >= BLOCK_SIZE - RADIUS) {
        int halo_idx = gidx + RADIUS;
        tile[lidx + RADIUS] = (halo_idx < n) ? in[halo_idx] : in[n-1];
    }
    
    __syncthreads();
    
    // Compute stencil from shared memory
    if (gidx < n) {
        float sum = 0.0f;
        for (int j = -RADIUS; j <= RADIUS; j++) {
            sum += tile[lidx + j];
        }
        out[gidx] = sum / (2 * RADIUS + 1);
    }
}

void cpu_stencil(float* out, const float* in, int n) {
    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int j = -RADIUS; j <= RADIUS; j++) {
            int idx = max(0, min(n-1, i + j));
            sum += in[idx];
        }
        out[i] = sum / (2 * RADIUS + 1);
    }
}

int main() {
    const int N = 16 * 1024 * 1024;
    const int iterations = 100;
    
    printf("Shared Memory Tiling - SOLUTION\n");
    printf("================================\n");
    printf("Array size: %d elements\n", N);
    printf("Stencil radius: %d (window size: %d)\n\n", RADIUS, 2*RADIUS+1);
    
    float *h_in, *h_out, *h_ref;
    float *d_in, *d_out;
    
    h_in = (float*)malloc(N * sizeof(float));
    h_out = (float*)malloc(N * sizeof(float));
    h_ref = (float*)malloc(N * sizeof(float));
    CHECK_CUDA(cudaMalloc(&d_in, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, N * sizeof(float)));
    
    for (int i = 0; i < N; i++) {
        h_in[i] = (float)(i % 100);
    }
    CHECK_CUDA(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));
    
    cpu_stencil(h_ref, h_in, N);
    
    int blockSize = BLOCK_SIZE;
    int numBlocks = (N + blockSize - 1) / blockSize;
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    // Benchmark naive
    stencil_naive<<<numBlocks, blockSize>>>(d_out, d_in, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        stencil_naive<<<numBlocks, blockSize>>>(d_out, d_in, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float time_naive;
    CHECK_CUDA(cudaEventElapsedTime(&time_naive, start, stop));
    time_naive /= iterations;
    
    printf("Naive Stencil: %.3f ms\n", time_naive);
    
    // Benchmark tiled
    stencil_tiled<<<numBlocks, blockSize>>>(d_out, d_in, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        stencil_tiled<<<numBlocks, blockSize>>>(d_out, d_in, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float time_tiled;
    CHECK_CUDA(cudaEventElapsedTime(&time_tiled, start, stop));
    time_tiled /= iterations;
    
    printf("Tiled Stencil: %.3f ms\n", time_tiled);
    printf("Speedup: %.2fx\n\n", time_naive / time_tiled);
    
    // Verify
    CHECK_CUDA(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_out[i] - h_ref[i]) > 1e-5) {
            printf("Mismatch at %d: got %f, expected %f\n", i, h_out[i], h_ref[i]);
            correct = false;
            break;
        }
    }
    printf("Verification: %s\n", correct ? "PASSED" : "FAILED");
    
    free(h_in); free(h_out); free(h_ref);
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    printf("\n✓ Solution complete!\n");
    return 0;
}
