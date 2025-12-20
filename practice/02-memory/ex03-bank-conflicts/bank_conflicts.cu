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

#define TILE_DIM 32

// =============================================================================
// TODO 1: Transpose with bank conflicts
// Load tile to shared memory, write transposed
// Use __shared__ float tile[TILE_DIM][TILE_DIM]
// =============================================================================
__global__ void transpose_bank_conflicts(float* out, const float* in, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM];  // Has bank conflicts!
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // TODO: Load tile (coalesced read from input)
    // TODO: __syncthreads()
    // TODO: Write transposed (has bank conflicts when reading columns)
}

// =============================================================================
// TODO 2: Transpose without bank conflicts
// Add +1 padding to shared memory to avoid conflicts
// Use __shared__ float tile[TILE_DIM][TILE_DIM + 1]
// =============================================================================
__global__ void transpose_no_conflicts(float* out, const float* in, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // +1 padding eliminates conflicts
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // TODO: Same logic as above, but now conflict-free
}

int main() {
    const int WIDTH = 4096;
    const int HEIGHT = 4096;
    const int iterations = 100;
    
    printf("Bank Conflicts Exercise\n");
    printf("=======================\n");
    printf("Matrix size: %dx%d\n\n", WIDTH, HEIGHT);
    
    size_t bytes = WIDTH * HEIGHT * sizeof(float);
    
    float *h_in, *h_out;
    float *d_in, *d_out;
    
    h_in = (float*)malloc(bytes);
    h_out = (float*)malloc(bytes);
    CHECK_CUDA(cudaMalloc(&d_in, bytes));
    CHECK_CUDA(cudaMalloc(&d_out, bytes));
    
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        h_in[i] = (float)i;
    }
    CHECK_CUDA(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid(WIDTH / TILE_DIM, HEIGHT / TILE_DIM);
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    // Benchmark with conflicts
    transpose_bank_conflicts<<<grid, block>>>(d_out, d_in, WIDTH, HEIGHT);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        transpose_bank_conflicts<<<grid, block>>>(d_out, d_in, WIDTH, HEIGHT);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float time_conflicts;
    CHECK_CUDA(cudaEventElapsedTime(&time_conflicts, start, stop));
    time_conflicts /= iterations;
    float bw_conflicts = 2 * bytes / (time_conflicts * 1e6);
    
    printf("With Bank Conflicts:\n");
    printf("  Time: %.3f ms, Bandwidth: %.1f GB/s\n\n", time_conflicts, bw_conflicts);
    
    // Benchmark without conflicts
    transpose_no_conflicts<<<grid, block>>>(d_out, d_in, WIDTH, HEIGHT);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        transpose_no_conflicts<<<grid, block>>>(d_out, d_in, WIDTH, HEIGHT);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float time_no_conflicts;
    CHECK_CUDA(cudaEventElapsedTime(&time_no_conflicts, start, stop));
    time_no_conflicts /= iterations;
    float bw_no_conflicts = 2 * bytes / (time_no_conflicts * 1e6);
    
    printf("Without Bank Conflicts:\n");
    printf("  Time: %.3f ms, Bandwidth: %.1f GB/s\n\n", time_no_conflicts, bw_no_conflicts);
    
    printf("Speedup: %.2fx\n", time_conflicts / time_no_conflicts);
    
    free(h_in); free(h_out);
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    
    return 0;
}
