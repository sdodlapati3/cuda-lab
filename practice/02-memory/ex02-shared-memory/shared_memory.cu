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
#define RADIUS 3  // Stencil radius (3-point = radius 1, 7-point = radius 3)

// =============================================================================
// TODO 1: Implement naive stencil (no shared memory)
// Each thread reads (2*RADIUS+1) values from global memory
// Compute average of neighbors: out[i] = (in[i-R] + ... + in[i] + ... + in[i+R]) / (2R+1)
// =============================================================================
__global__ void stencil_naive(float* out, const float* in, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < n) {
        // TODO: Compute stencil from global memory
        // Handle boundary conditions (clamp to valid range)
    }
}

// =============================================================================
// TODO 2: Implement tiled stencil with shared memory
// Load tile + halo to shared memory, then compute
// =============================================================================
__global__ void stencil_tiled(float* out, const float* in, int n) {
    // TODO: Declare shared memory array with size BLOCK_SIZE + 2*RADIUS
    // __shared__ float tile[???];
    
    int gidx = threadIdx.x + blockIdx.x * blockDim.x;  // Global index
    int lidx = threadIdx.x + RADIUS;  // Local index in tile (offset by RADIUS)
    
    // TODO: Load main elements to shared memory
    
    // TODO: Load halo elements (left and right boundaries)
    // First RADIUS threads load left halo
    // Last RADIUS threads load right halo
    
    // TODO: Synchronize threads
    
    // TODO: Compute stencil from shared memory
    
    // TODO: Write result to global memory
}

// =============================================================================
// Verification and benchmarking
// =============================================================================
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
    const int N = 16 * 1024 * 1024;  // 16M elements
    const int iterations = 100;
    
    printf("Shared Memory Tiling Exercise\n");
    printf("==============================\n");
    printf("Array size: %d elements\n", N);
    printf("Stencil radius: %d (window size: %d)\n\n", RADIUS, 2*RADIUS+1);
    
    // Allocate memory
    float *h_in, *h_out, *h_ref;
    float *d_in, *d_out;
    
    h_in = (float*)malloc(N * sizeof(float));
    h_out = (float*)malloc(N * sizeof(float));
    h_ref = (float*)malloc(N * sizeof(float));
    CHECK_CUDA(cudaMalloc(&d_in, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, N * sizeof(float)));
    
    // Initialize
    for (int i = 0; i < N; i++) {
        h_in[i] = (float)(i % 100);
    }
    CHECK_CUDA(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));
    
    // CPU reference
    cpu_stencil(h_ref, h_in, N);
    
    // Launch configuration
    int blockSize = BLOCK_SIZE;
    int numBlocks = (N + blockSize - 1) / blockSize;
    
    // Benchmark naive
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    stencil_naive<<<numBlocks, blockSize>>>(d_out, d_in, N);  // Warmup
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
    stencil_tiled<<<numBlocks, blockSize>>>(d_out, d_in, N);  // Warmup
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
    
    // Cleanup
    free(h_in); free(h_out); free(h_ref);
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    return 0;
}
